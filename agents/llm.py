"""
Model backend for the agents.

  MATCHROOM_BACKEND=offline   deterministic templates, no cloud needed (default)
  MATCHROOM_BACKEND=foundry   calls the model deployed in Microsoft Foundry directly (chat completions)
  MATCHROOM_BACKEND=agents    calls versioned Foundry Agent Service agents (matchroom-narrator,
                              matchroom-recap-writer, matchroom-localiser). Falls back to the direct
                              model if the agents can't be reached.

Settings (local .env file, never committed; the Azure deployment sets them for you):
  AZURE_OPENAI_ENDPOINT      https://<resource>.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT    gpt-5.4-mini
  AZURE_OPENAI_API_VERSION   defaults to 2025-04-01-preview
  AZURE_OPENAI_API_KEY       optional; if absent, Entra ID (az/azd login locally, managed identity in Azure)
  AZURE_AI_PROJECT_ENDPOINT  https://<resource>.services.ai.azure.com/api/projects/<project>  (agents backend)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from agents import telemetry


def _load_dotenv():
    p = Path(".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()


def _use_system_certificates():
    """Trust the operating system's certificate store (Windows/macOS/Linux) instead of only Python's bundle.

    Company networks often inspect HTTPS with their own root certificate, which Windows trusts but Python
    doesn't, causing CERTIFICATE_VERIFY_FAILED. This keeps full TLS verification; it just uses the same
    trusted roots as the browser. Set MATCHROOM_SYSTEM_CERTS=0 to turn it off.
    """
    if os.getenv("MATCHROOM_SYSTEM_CERTS", "1") == "0":
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass  # truststore not installed: Python's default certificates are used


_use_system_certificates()


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content or "", re.S)  # tolerate text around the JSON object
        if not m:
            raise
        return json.loads(m.group(0))


class LLM:
    def __init__(self, backend: str | None = None):
        self.backend = (backend or os.getenv("MATCHROOM_BACKEND", "offline")).lower()
        self.calls = 0
        self.agent_calls = 0
        self.latency_ms = 0.0
        self.failures = 0
        self.last_error = None
        self.latencies: list[float] = []
        self.throttled = 0  # HTTP 429 responses (quota / rate limit)
        self.transient = 0  # 5xx / connection / timeout errors that were retried
        self._tripped_at = 0.0
        self._lock = threading.Lock()
        self._dropped: set[str] = set()   # request params this model rejected
        self.client = None
        self.project = None
        self.agents_ok = False
        self.agent_versions: dict[str, str] = {}
        self._agent_clients: dict[str, object] = {}
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")

        if self.backend in ("foundry", "agents") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.client = self._chat_client()
        if self.backend == "agents":
            self._setup_agents()
            if not self.agents_ok:
                if self.client is None:
                    raise RuntimeError("Foundry agents unavailable and AZURE_OPENAI_ENDPOINT not set for fallback")
                print("[agents] falling back to calling the model directly")
        if self.backend in ("foundry", "agents"):
            telemetry.setup(self.project)

    # ------------------------------------------------------------- clients
    def _credential(self):
        from azure.identity import DefaultAzureCredential  # covers az login, azd login, managed identity
        return DefaultAzureCredential()

    def _chat_client(self):
        from openai import AzureOpenAI

        kwargs = dict(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                      api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
                      max_retries=0,  # we retry ourselves so slow-downs are visible, not silent
                      timeout=float(os.getenv("MATCHROOM_TIMEOUT", "60")))
        if os.getenv("AZURE_OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
        else:
            from azure.identity import get_bearer_token_provider
            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                self._credential(), "https://cognitiveservices.azure.com/.default")
        return AzureOpenAI(**kwargs)

    def _setup_agents(self):
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        if not endpoint:
            print("[agents] AZURE_AI_PROJECT_ENDPOINT is not set")
            return
        try:
            from azure.ai.projects import AIProjectClient

            from agents.foundry import ensure_agents

            self.project = AIProjectClient(endpoint=endpoint, credential=self._credential())
            self.agent_versions = ensure_agents(self.project, self.deployment)
            self.agents_ok = True
        except Exception as ex:
            print(f"[agents] setup failed: {ex.__class__.__name__}: {str(ex)[:300]}")

    def _agent_client(self, name: str):
        with self._lock:
            if name not in self._agent_clients:
                self._agent_clients[name] = self.project.get_openai_client(agent_name=name)
            return self._agent_clients[name]

    # ------------------------------------------------------------- status
    @property
    def online(self) -> bool:
        # circuit breaker: after 3 failures in a row, pause model calls for 30 s, then try again
        if self.backend not in ("foundry", "agents"):
            return False
        return self.failures < 3 or time.time() - self._tripped_at > 30

    @property
    def label(self) -> str:
        if self.backend == "agents" and self.agents_ok:
            return "Foundry Agent Service"
        if self.backend in ("foundry", "agents"):
            return "Microsoft Foundry model"
        return "Offline templates"

    def _params(self, max_tokens: int) -> dict:
        reasoning = self.deployment.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        p = {"response_format": {"type": "json_object"}}
        if reasoning:
            # reasoning models: no temperature; output budget must also cover hidden reasoning tokens
            p["max_completion_tokens"] = max_tokens * 4
            p["reasoning_effort"] = "minimal"  # tested: ~3 s per reply vs ~110 s at "low"
        else:
            p["temperature"] = 0.4
            p["max_tokens"] = max_tokens
        return {k: v for k, v in p.items() if k not in self._dropped}

    # --------------------------------------------------------------- calls
    def _invoke(self, system, payload, max_tokens, agent):
        """One model call. Returns (content, input_tokens, output_tokens, via_agent)."""
        if agent and self.agents_ok:
            # JSON output mode requires the word "json" in the input itself, not only in the agent's instructions
            text = "Request as JSON (reply with a JSON object):\n" + json.dumps(payload, ensure_ascii=False)
            r = self._agent_client(agent).responses.create(input=text)
            u = getattr(r, "usage", None)
            return r.output_text, getattr(u, "input_tokens", None), getattr(u, "output_tokens", None), True
        messages = [{"role": "system", "content": system + "\nRespond with a JSON object only."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        r = self.client.chat.completions.create(model=self.deployment, messages=messages, **self._params(max_tokens))
        u = getattr(r, "usage", None)
        return (r.choices[0].message.content or "", getattr(u, "prompt_tokens", None),
                getattr(u, "completion_tokens", None), False)

    def json(self, system: str, payload: dict, max_tokens: int = 300, agent: str | None = None) -> dict:
        """Call the agent (or model) and return a parsed JSON object."""
        import openai

        use_agent = bool(agent and self.agents_ok)
        name = f"invoke_agent {agent}" if use_agent else f"chat {self.deployment}"
        with telemetry.span(name, **{
                "gen_ai.operation.name": "invoke_agent" if use_agent else "chat",
                "gen_ai.provider.name": "azure.ai.foundry" if use_agent else "azure.ai.openai",
                "gen_ai.agent.name": agent if use_agent else None,
                "gen_ai.agent.version": self.agent_versions.get(agent) if use_agent else None,
                "gen_ai.request.model": self.deployment}) as sp:
            for attempt in range(6):
                t0 = time.time()
                try:
                    content, tin, tout, via_agent = self._invoke(system, payload, max_tokens, agent)
                    out = _parse_json(content)
                    ms = (time.time() - t0) * 1000
                    with self._lock:
                        self.calls += 1
                        self.agent_calls += int(via_agent)
                        self.failures = 0
                        self.latency_ms += ms
                        self.latencies.append(ms)
                    telemetry.set_attrs(sp, **{"gen_ai.usage.input_tokens": tin, "gen_ai.usage.output_tokens": tout,
                                               "matchroom.latency_ms": round(ms), "matchroom.attempts": attempt + 1})
                    return out
                except openai.RateLimitError as ex:
                    with self._lock:
                        self.throttled += 1
                    wait = 2.0
                    try:
                        wait = float(ex.response.headers.get("retry-after", wait))
                    except Exception:
                        pass
                    time.sleep(min(wait, 10))
                    continue
                except (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as ex:
                    with self._lock:
                        self.transient += 1
                    if attempt < 2:  # Azure occasionally returns 500/503; a short retry almost always succeeds
                        time.sleep(0.8 * (attempt + 1))
                        continue
                    self._fail(ex)
                    raise
                except (openai.NotFoundError, openai.PermissionDeniedError, openai.AuthenticationError) as ex:
                    if agent and self.agents_ok and self.client is not None:
                        # agent missing or no permission: keep the broadcast going on the direct model
                        print(f"[agents] {ex.__class__.__name__} calling {agent}; switching to the direct model")
                        self.agents_ok = False
                        continue
                    self._fail(ex)
                    raise
                except openai.BadRequestError as ex:
                    msg = str(ex)
                    rejected_effort = "reasoning" in msg or ("Supported values" in msg and "'none'" in msg)
                    if agent and self.agents_ok and rejected_effort and not getattr(self, "_repaired", False):
                        # the model rejected the agent's reasoning effort: re-version the agents with one it lists
                        self._repaired = True
                        effort = "none" if "'none'" in msg else "low"
                        from agents.foundry import ensure_agents
                        print(f"[agents] model rejected the reasoning setting; updating agents to effort '{effort}'")
                        self.agent_versions = ensure_agents(self.project, self.deployment, effort=effort)
                        with self._lock:
                            self._agent_clients.clear()
                        continue
                    bad = next((k for k in ("reasoning_effort", "temperature", "response_format",
                                            "max_completion_tokens", "max_tokens") if k in msg), None)
                    if bad and bad not in self._dropped and not (agent and self.agents_ok):
                        self._dropped.add(bad)   # model doesn't support this setting: retry without it
                        continue
                    self._fail(ex)
                    raise
                except Exception as ex:
                    self._fail(ex)
                    raise
            raise RuntimeError("model call failed after retries")

    def summary(self) -> str:
        if not self.latencies:
            return f"{self.label}: no successful calls"
        xs = sorted(self.latencies)
        p50, p95 = xs[len(xs) // 2], xs[min(len(xs) - 1, int(len(xs) * 0.95))]
        msg = f"{self.label}: {self.calls} calls"
        if self.agent_calls:
            msg += f" ({self.agent_calls} via agents)"
        msg += f", median {p50 / 1000:.1f} s, p95 {p95 / 1000:.1f} s"
        if self.transient:
            msg += f", {self.transient} transient Azure errors retried"
        if self.throttled:
            msg += (f", {self.throttled} rate-limited (HTTP 429): raise the deployment's "
                    f"tokens-per-minute limit in Foundry")
        return msg

    def _fail(self, ex):
        self.failures += 1
        if self.last_error is None:
            print(f"\n[model] first error: {ex.__class__.__name__}: {str(ex)[:400]}\n")
        self.last_error = ex
        if self.failures == 3:
            self._tripped_at = time.time()
            print("[model] 3 failures in a row: using safe templates for 30 s, then retrying the model\n")
