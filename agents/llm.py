"""
Model backend for the agents.

  MATCHROOM_BACKEND=offline   deterministic templates, no cloud needed (default)
  MATCHROOM_BACKEND=foundry   Azure OpenAI model deployed in Microsoft Foundry

Foundry settings (put them in a local .env file, never commit it):
  AZURE_OPENAI_ENDPOINT      https://<resource>.openai.azure.com/  (or the Foundry project's OpenAI endpoint)
  AZURE_OPENAI_DEPLOYMENT    matchroom-chat
  AZURE_OPENAI_API_VERSION   defaults to 2024-10-21
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
                          api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))
            if os.getenv("AZURE_OPENAI_API_KEY"):
                kwargs["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
            else:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            self.client = AzureOpenAI(**kwargs)
            self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "matchroom-chat")

    @property
    def online(self) -> bool:
        return self.backend == "foundry"

    def json(self, system: str, payload: dict, max_tokens: int = 300) -> dict:
        """Call the model and return a parsed JSON object."""
        t0 = time.time()
        r = self.client.chat.completions.create(
            model=self.deployment, temperature=0.4, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        self.calls += 1
        self.latency_ms += (time.time() - t0) * 1000
        return json.loads(r.choices[0].message.content)
