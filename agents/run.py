"""
Matchroom orchestrator: one live match, many personalised feeds.

    # offline (no Azure needed), three preset viewers
    python -m agents.run data/match_001

    # a custom viewer
    python -m agents.run data/match_001 --mode player_focus --player "Elias Castell" --lang es
    python -m agents.run data/match_001 --mode metric --metric xg --club NGC

    # with Microsoft Foundry
    MATCHROOM_BACKEND=foundry python -m agents.run data/match_001 --speed 20

Flow per cue:  engine cue -> Personaliser -> Narrator -> Verifier -> Localiser -> Verifier -> feed
At half/full time: RecapWriter -> Verifier -> Localiser.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.agents import Localiser, Narrator, Personaliser, RecapWriter, Verifier
from agents.llm import LLM
from agents.profiles import PRESETS, Profile
from engine.insights import MatchEngine
from engine.replay import stream


class Orchestrator:
    def __init__(self, profiles: dict[str, Profile], backend: str | None = None):
        self.llm = LLM(backend)
        self.engine = MatchEngine()
        self.profiles = profiles
        self.feeds = {k: [] for k in profiles}
        self.recaps = {k: [] for k in profiles}
        self.ready = False

    def _setup(self):
        e = self.engine
        self.teams = {t: n for t, n in zip(e.teams, (e.meta["home"]["short"], e.meta["away"]["short"]))}
        self.verifier = Verifier(e.names)
        self.narrator = Narrator(self.llm, self.teams)
        self.recapper = RecapWriter(self.llm, self.teams)
        self.localiser = Localiser(self.llm)
        self.pers = {}
        for k, p in self.profiles.items():
            ps = Personaliser(p)
            ps.focus_name = e.names.get(p.player_id, "") if p.player_id else ""
            self.pers[k] = ps
        self.ready = True

    # ---- one grounded utterance
    def _say(self, facts, draft_fn, safe_fn, lang):
        text, fallback, issues = None, False, []
        try:
            text = draft_fn()
            issues = self.verifier.check(text, facts)
        except Exception as ex:  # model/network failure must never break a live broadcast
            issues = [f"generation error: {ex.__class__.__name__}"]
        if issues:
            text, fallback = safe_fn(), True
        out, out_lang = text, "en"
        try:
            out, out_lang = self.localiser.translate(text, lang)
            t_issues = self.verifier.check(out, facts) if out_lang != "en" else []
            if t_issues:
                issues += [f"translation: {i}" for i in t_issues]
                out, out_lang = text, "en"
        except Exception as ex:
            issues.append(f"translation error: {ex.__class__.__name__}")
        return out, out_lang, fallback, issues

    def handle_cue(self, cue):
        for k, p in self.profiles.items():
            if not self.pers[k].wants(cue):
                continue
            self._emit(k, p, cue)

    def _emit(self, k, p, cue):
        facts = self.narrator.facts(cue)
        text, lang, fb, issues = self._say(
            facts, lambda: self.narrator.write(cue, p), lambda: self.narrator.template(cue, p), p.language)
        self.feeds[k].append({
            "t": cue["t_in"], "t_out": cue["t_out"], "clock": cue["clock"], "cue_id": cue["cue_id"],
            "kind": cue["kind"], "priority": cue["priority"], "team": cue.get("team"),
            "player_id": cue.get("player_id"), "text": text, "lang": lang,
            "verified": not issues, "fallback": fb, "issues": issues,
            "evidence_event_ids": cue["explain"]["evidence_event_ids"], "data": cue["data"],
        })

    def handle_recap(self, label):
        stats = self.engine.stats()
        for k, p in self.profiles.items():
            holder = {}

            def draft():
                holder["r"] = self.recapper.write(label, stats, self.engine.cues, p)
                return holder["r"]["text"]

            facts = self.recapper.facts(label, stats, self.engine.cues)
            text, lang, fb, issues = self._say(
                facts, draft, lambda: self.recapper.template(facts, p), p.language)
            self.recaps[k].append({"t": self.engine.t, "label": label, "text": text, "lang": lang,
                                   "verified": not issues, "fallback": fb, "issues": issues})

    def run(self, folder: Path, speed: float = 0, echo: str | None = None):
        t0, wall0 = None, time.time()
        for t, kind, obj in stream(folder):
            if speed:
                t0 = t if t0 is None else t0
                wait = (t - t0) / speed - (time.time() - wall0)
                if wait > 0:
                    time.sleep(wait)
            if kind == "frame":
                new = self.engine.ingest_frame(obj)
            else:
                new = self.engine.ingest_event(obj)
                if obj["type"] == "match_metadata":
                    self._setup()
                    continue
            n_before = {k: len(v) for k, v in self.feeds.items()}
            for c in new:
                self.handle_cue(c)
            if kind == "event" and self.ready:
                for k, ps in self.pers.items():
                    for c in ps.periodic(self.engine.t, self.engine.stats(), self.engine.names):
                        self._emit(k, self.profiles[k], c)
                if obj["type"] in ("half_time", "full_time"):
                    self.handle_recap("Half-time" if obj["type"] == "half_time" else "Full-time")
                    if echo:
                        r = self.recaps[echo][-1]
                        print(f"\n=== {r['label']} recap [{echo}] ===\n{r['text']}\n")
            if echo:
                for item in self.feeds[echo][n_before.get(echo, 0):]:
                    flag = "" if item["verified"] else "  [verifier: " + "; ".join(item["issues"]) + "]"
                    print(f"{item['clock']:>4} {item['text']}{flag}")

    def save(self, folder: Path):
        out = folder / "feeds"
        out.mkdir(exist_ok=True)
        for k in self.profiles:
            with open(out / f"{k}.jsonl", "w", encoding="utf-8") as f:
                for it in self.feeds[k]:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            json.dump(self.recaps[k], open(out / f"{k}_recaps.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--backend", choices=["offline", "foundry"])
    ap.add_argument("--speed", type=float, default=0)
    ap.add_argument("--mode", choices=["casual", "analyst", "player_focus", "metric"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--club", action="append", default=[])
    ap.add_argument("--player", help="player id or full name (player_focus)")
    ap.add_argument("--metric")
    ap.add_argument("--echo", help="profile key to print live (default: first)")
    a = ap.parse_args()
    folder = Path(a.folder)

    if a.mode:
        pid = a.player
        if pid and "-" not in pid:  # resolve a name to an id from match metadata
            meta = json.loads(open(folder / "events.jsonl").readline())
            pid = next(p["id"] for lu in meta["lineups"].values() for p in lu if p["name"].lower() == pid.lower())
        profiles = {"custom": Profile("Custom", a.mode, a.lang, a.club, pid, a.metric)}
    else:
        profiles = dict(PRESETS)
    orch = Orchestrator(profiles, a.backend)
    orch.run(folder, a.speed, a.echo or next(iter(profiles)))
    out = orch.save(folder)
    v = orch.verifier
    print(f"\nFeeds written to {out}")
    for k, items in orch.feeds.items():
        print(f"  {k:<18} {len(items):>3} on-screen items, {len(orch.recaps[k])} recaps")
    print(f"Verifier: {v.checked} checks, {v.blocked} blocked and replaced with safe templates")
    if orch.llm.calls:
        print(f"Model: {orch.llm.calls} calls, avg {orch.llm.latency_ms / orch.llm.calls:.0f} ms")


if __name__ == "__main__":
    main()
