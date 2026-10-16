"""
Fault injection: prove the Verifier works by corrupting real AI output on purpose.

    python -m ui.server --backend agents --inject-faults        # 1 in 5 narrator lines corrupted
    python -m agents.run data/match_001 --inject-faults

The model writes as normal; then every Nth narrator line is altered in one of three ways a real model
can fail: an invented number, a player who isn't in the moment, or words in the wrong script. The UI
labels the mode clearly. Off by default and never enabled in the deployed demo.
"""
from __future__ import annotations

import itertools
import random
import threading

FAULTS = ("number", "player", "script")


class FaultInjector:
    def __init__(self, llm, every: int = 5, seed: int = 11):
        self._llm = llm
        self.every = max(2, every)
        self.roster: list[str] = []
        self.injected = 0
        self._n = itertools.count(1)
        self._kind = itertools.cycle(FAULTS)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def __getattr__(self, name):  # behave exactly like the wrapped backend otherwise
        return getattr(self._llm, name)

    def _corrupt(self, text: str, payload: dict) -> str:
        kind = next(self._kind)
        facts = payload.get("facts", {})
        if kind == "number":
            return text.rstrip(".!") + ", after a 97 metre solo run!"
        if kind == "player":
            others = [n for n in self.roster if n not in str(facts)]
            if others:
                return f"{self._rng.choice(others)} was involved too. " + text
            kind = "script"
        return text.rstrip(".!") + " शानदार!"

    def json(self, system, payload, max_tokens=300, agent=None):
        out = self._llm.json(system, payload, max_tokens=max_tokens, agent=agent)
        if agent == "matchroom-narrator" and "text" in out:
            with self._lock:
                hit = next(self._n) % self.every == 0
                if hit:
                    self.injected += 1
            if hit:
                out = dict(out, text=self._corrupt(out["text"], payload))
        return out
