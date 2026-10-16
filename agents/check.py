"""Foundry connectivity check:  python -m agents.check"""
import os
import sys

from agents.llm import LLM


def main():
    ep = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    dep = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    print(f"endpoint   : {ep or '(missing)'}")
    print(f"deployment : {dep or '(missing)'}")
    print(f"api key    : {'set' if os.getenv('AZURE_OPENAI_API_KEY') else 'not set (using Entra ID / az login)'}")
    problems = []
    if not ep:
        problems.append("AZURE_OPENAI_ENDPOINT is missing from .env")
    elif "/api/projects" in ep or "/openai/" in ep:
        problems.append("endpoint should stop after '.com/' (remove /openai/... or /api/projects/...)")
    if not dep:
        problems.append("AZURE_OPENAI_DEPLOYMENT is missing (use the Name column on the Deployments page)")
    if problems:
        print("\nFix these in .env first:\n - " + "\n - ".join(problems))
        sys.exit(1)
    llm = LLM("foundry")
    try:
        out = llm.json("You are a football commentator. Return {\"text\": \"...\"}.",
                       {"task": "one short line about a 30 metre through ball"}, max_tokens=80)
        print(f"\nOK, model replied in {llm.latency_ms:.0f} ms: {out.get('text', out)}")
        if llm._dropped:
            print(f"(automatically skipped unsupported settings: {', '.join(sorted(llm._dropped))})")
    except Exception as ex:
        print(f"\nFAILED: {ex.__class__.__name__}: {str(ex)[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
