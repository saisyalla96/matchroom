"""
Model backend for the agents.

  MATCHROOM_BACKEND=offline   deterministic templates, no cloud needed (default)
  MATCHROOM_BACKEND=foundry   Azure OpenAI model deployed in Microsoft Foundry

Foundry settings (put them in a local .env file, never commit it):
  AZURE_OPENAI_ENDPOINT      https://<resource>.openai.azure.com/  (or the Foundry project's OpenAI endpoint)
  AZURE_OPENAI_DEPLOYMENT    matchroom-chat
  AZURE_OPENAI_API_VERSION   defaults to 2025-04-01-preview
  AZURE_OPENAI_API_KEY       optional; if absent, Entra ID (DefaultAzureCredential / managed identity) is used
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _load_dotenv():
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()


class LLM:
    def __init__(self, backend: str | None = None):
        self.backend = (backend or os.getenv("MATCHROOM_BACKEND", "offline")).lower()
        self.calls = 0
        self.latency_ms = 0.0
        if self.backend == "foundry":
            from openai import AzureOpenAI  # pip install openai azure-identity

            kwargs = dict(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                          api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"))
            if os.getenv("AZURE_OPENAI_API_KEY"):
                kwargs["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
            else:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            self.client = AzureOpenAI(**kwargs)
            self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")

        self.failures = 0
        self.last_error = None
        self._dropped: set[str] = set()   # request params this model rejected

    @property
    def online(self) -> bool:
        # after 3 consecutive failures, stop calling the model for this run (live broadcast safety)
        return self.backend == "foundry" and self.failures < 3

    def _params(self, max_tokens: int) -> dict:
        reasoning = self.deployment.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        p = {"response_format": {"type": "json_object"}}
        if reasoning:
            # reasoning models: no temperature; output budget must also cover hidden reasoning tokens
            p["max_completion_tokens"] = max_tokens * 4
            p["reasoning_effort"] = "low"
        else:
            p["temperature"] = 0.4
            p["max_tokens"] = max_tokens
        return {k: v for k, v in p.items() if k not in self._dropped}

    def json(self, system: str, payload: dict, max_tokens: int = 300) -> dict:
        """Call the model and return a parsed JSON object."""
        import openai

        messages = [{"role": "system", "content": system + "\nRespond with a JSON object only."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for _ in range(4):
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(model=self.deployment, messages=messages,
                                                        **self._params(max_tokens))
                content = r.choices[0].message.content or ""
                out = json.loads(content)
                self.calls += 1
                self.failures = 0
                self.latency_ms += (time.time() - t0) * 1000
                return out
            except openai.BadRequestError as ex:
                msg = str(ex)
                bad = next((k for k in ("reasoning_effort", "temperature", "response_format",
                                        "max_completion_tokens", "max_tokens") if k in msg), None)
                if bad and bad not in self._dropped:
                    self._dropped.add(bad)   # model doesn't support this setting: retry without it
                    continue
                self._fail(ex)
                raise
            except Exception as ex:
                self._fail(ex)
                raise
        raise RuntimeError("model call failed after parameter retries")

    def _fail(self, ex):
        self.failures += 1
        if self.last_error is None:
            print(f"\n[model] first error: {ex.__class__.__name__}: {str(ex)[:400]}\n")
        self.last_error = ex
        if self.failures == 3:
            print("[model] 3 failures in a row: switching to safe templates for the rest of this run\n")
