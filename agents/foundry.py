"""
Register Matchroom's AI agents in Microsoft Foundry Agent Service, and test them.

    python -m agents.foundry            # create/update the agents, then send one test request to each

Agents (versioned in Foundry, visible under Build > Agents):
  matchroom-narrator       live on-screen lines, per viewer mode, language and club perspective
  matchroom-recap-writer   half-time and full-time recaps
  matchroom-localiser      translation that keeps names and numbers exact

The Personaliser and Verifier stay as deterministic code on purpose: the guardrail must not be able
to hallucinate. A new agent version is only created when its instructions or model change, so this is
safe to run on every start.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

JSON_RULE = "\nRespond with a JSON object only."


def agent_specs() -> dict[str, tuple[str, str]]:
    from agents.agents import LOCALISE_SYSTEM, NARRATOR_SYSTEM, RECAP_SYSTEM

    return {
        "matchroom-narrator": (NARRATOR_SYSTEM, "Writes grounded live broadcast lines for one viewer."),
        "matchroom-recap-writer": (RECAP_SYSTEM, "Writes half-time and full-time recaps from verified facts."),
        "matchroom-localiser": (LOCALISE_SYSTEM, "Translates broadcast text, keeping names and numbers exact."),
    }


def ensure_agents(project, model: str, effort: str | None = None, verbose: bool = True) -> dict[str, str]:
    """Create a new version of each agent only when its definition changed. Returns {name: version}."""
    from azure.ai.projects.models import (PromptAgentDefinition, PromptAgentDefinitionTextOptions, Reasoning,
                                          TextResponseFormatJsonObject)

    # "none" = fastest; gpt-5.4-mini rejects "minimal" through the Agent Service (tested Oct 2026)
    effort = effort or os.getenv("MATCHROOM_AGENT_REASONING", "none")
    versions = {}
    for name, (prompt, description) in agent_specs().items():
        instructions = prompt + JSON_RULE
        sha = hashlib.sha256(f"{model}|{effort}|{instructions}".encode()).hexdigest()[:16]
        latest = None
        try:
            latest = next(iter(project.agents.list_versions(name, limit=1, order="desc")), None)
        except Exception:
            latest = None  # agent doesn't exist yet
        if latest is not None and (getattr(latest, "metadata", None) or {}).get("prompt_sha") == sha:
            versions[name] = str(latest.version)
            if verbose:
                print(f"[agents] {name} v{latest.version} is up to date")
            continue

        def definition(with_reasoning=True, with_json=True):
            kw = dict(model=model, instructions=instructions)
            if with_reasoning:
                kw["reasoning"] = Reasoning(effort=effort)
            if with_json:
                kw["text"] = PromptAgentDefinitionTextOptions(format=TextResponseFormatJsonObject())
            return PromptAgentDefinition(**kw)

        created, last_err = None, None
        for opts in ((True, True), (False, True), (False, False)):  # drop settings the model rejects
            try:
                created = project.agents.create_version(
                    agent_name=name, definition=definition(*opts), description=description,
                    metadata={"app": "matchroom", "prompt_sha": sha})
                break
            except Exception as ex:
                last_err = ex
        if created is None:
            raise last_err
        versions[name] = str(created.version)
        if verbose:
            print(f"[agents] {name} v{created.version} created")
    return versions


def main():
    from agents.llm import LLM

    llm = LLM("agents")
    if not llm.agents_ok:
        print("\nFoundry agents are NOT available. Check AZURE_AI_PROJECT_ENDPOINT in .env, that you are signed in "
              "(azd auth login), and that your account has the 'Azure AI User' role on the Foundry resource.")
        raise SystemExit(1)
    sample = {
        "matchroom-narrator": {"viewer": {"mode": "casual", "clubs": [], "metric": None}, "language": "es",
                               "perspective": "neutral", "teams": {"RVA": "Riverside", "NGC": "Northgate"},
                               "facts": {"kind": "big_chance", "clock": "21'", "team": "Riverside",
                                         "player_name": "Ezra Quill", "headline": "Big chance: Ezra Quill (0.24 xG), saved",
                                         "data": {"xg": 0.24, "distance_m": 9.8, "outcome": "saved"},
                                         "why": "A shot from 9.8 m is converted about 24% of the time."}},
        "matchroom-recap-writer": {"label": "Half-time", "viewer": {"mode": "casual", "clubs": []}, "language": "en",
                                   "facts": {"score": {"RVA": 0, "NGC": 1}, "teams": {"RVA": "Riverside", "NGC": "Northgate"},
                                             "key_moments": [{"clock": "24'", "kind": "goal",
                                                              "headline": "GOAL Northgate: Elias Castell | 0-1",
                                                              "why": "A 6-pass move finished from a 0.04 xG chance."}]}},
        "matchroom-localiser": {"lang": "fr", "text": "Ezra Quill's shot flew at 106.1 km/h."},
    }
    print()
    for name, payload in sample.items():
        t0 = time.time()
        out = llm.json("", payload, agent=name)
        print(f"{name} (v{llm.agent_versions.get(name)}, {time.time() - t0:.1f} s): {out.get('text')}")
    print(f"\nOK: {llm.summary()}")
    print("Tracing:", "on (see Foundry > Agents > Traces, or Application Insights)"
          if __import__("agents.telemetry", fromlist=["x"])._enabled else
          "off (set APPLICATIONINSIGHTS_CONNECTION_STRING in .env, or connect App Insights to the Foundry project)")


if __name__ == "__main__":
    main()
