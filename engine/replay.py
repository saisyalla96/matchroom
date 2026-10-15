"""
Replay a synthetic match through the insights engine, as if live.

    python -m engine.replay data/match_001                # as fast as possible
    python -m engine.replay data/match_001 --speed 30     # 30x real time, prints cues live

Writes cues.jsonl, stats.json and timeline.json next to the input, and checks the
explainability layer against the simulator's hidden tactical ground truth.
"""
from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

from engine.insights import MatchEngine


def stream(folder: Path):
    """Merge events and tracking frames into one time-ordered stream (stage a: ingest)."""
    ev = ((e["t"], 1, "event", e) for e in map(json.loads, open(folder / "events.jsonl")))
    fr = ((f["t"], 0, "frame", f) for f in map(json.loads, open(folder / "tracking.jsonl")))
    for t, _, kind, obj in heapq.merge(ev, fr, key=lambda x: (x[0], x[1])):
        yield t, kind, obj


def evaluate(cues, truth_path: Path):
    """How often do pressure-surge cues fall inside a true high-press phase?"""
    if not truth_path.exists():
        return None
    phases = json.load(open(truth_path))["tactical_phases"]
    surges = [c for c in cues if c["kind"] == "pressure_surge"]
    hits = sum(1 for c in surges if any(p["team"] == c["team"] and p["phase"] == "high_press"
                                        and p["start"] <= c["t_in"] <= p["end"] + 120 for p in phases))
    return {"pressure_surge_cues": len(surges), "inside_true_high_press": hits,
            "precision": round(hits / len(surges), 2) if surges else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--speed", type=float, default=0, help="real-time multiplier; 0 = no waiting")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    folder = Path(a.folder)
    eng = MatchEngine()
    t0, wall0 = None, time.time()
    for t, kind, obj in stream(folder):
        if a.speed:
            t0 = t if t0 is None else t0
            wait = (t - t0) / a.speed - (time.time() - wall0)
            if wait > 0:
                time.sleep(wait)
        new = eng.ingest_event(obj) if kind == "event" else eng.ingest_frame(obj)
        if not a.quiet:
            for c in new:
                print(f"{c['clock']:>4} [{c['kind']:<15}] {c['headline']}\n       why: {c['explain']['why']}")

    with open(folder / "cues.jsonl", "w") as f:
        for c in eng.cues:
            f.write(json.dumps(c) + "\n")
    json.dump(eng.stats(), open(folder / "stats.json", "w"), indent=2)
    json.dump(eng.timeline, open(folder / "timeline.json", "w"))
    s = eng.stats()
    print("\nFINAL", s["score"])
    for t, v in s["teams"].items():
        print(f"  {t}: poss {v['possession_pct']}% | passes {v['passes']} ({v['pass_accuracy_pct']}%) | "
              f"shots {v['shots']} ({v['on_target']} on target) | xG {v['xg']} | pressures {v['pressures']}")
    kinds = {}
    for c in eng.cues:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print("cues:", kinds)
    ev = evaluate(eng.cues, folder / "truth.json")
    if ev:
        print("ground-truth check:", ev)


if __name__ == "__main__":
    main()
