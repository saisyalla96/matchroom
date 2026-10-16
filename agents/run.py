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
        """draft -> verify -> (fallback template -> translate -> verify). Never raises."""
        text, text_lang, fallback, issues = None, "en", False, []
        try:
            text, text_lang = draft_fn()
            issues = self.verifier.check(text, facts, text_lang)
        except Exception as ex:  # model/network failure must never break a live broadcast
            issues = [f"generation error: {ex.__class__.__name__}"]
        if issues:
            text, text_lang, fallback = safe_fn(), "en", True
        if text_lang.split("-")[0] != lang.split("-")[0]:
            try:
                out, out_lang = self.localiser.translate(text, lang)
                t_issues = self.verifier.check(out, facts, out_lang) if out_lang != "en" else []
                if t_issues:
                    issues += [f"translation: {i}" for i in t_issues]
                else:
                    text, text_lang = out, out_lang
            except Exception as ex:
                issues.append(f"translation error: {ex.__class__.__name__}")
        return text, text_lang, fallback, issues

    def _submit(self, fn):
        if self.pool:
            self.pending.append(self.pool.submit(fn))
        else:
            fn()

    def _print(self, line):
        with self._print_lock:
            print(line, flush=True)

    def handle_cue(self, cue):
        for k, p in self.profiles.items():
            if not self.pers[k].wants(cue):
                continue
            self._emit(k, p, cue)

    def _emit(self, k, p, cue):
        facts = self.narrator.facts(cue)
        item = {
            "t": cue["t_in"], "t_out": cue["t_out"], "clock": cue["clock"], "cue_id": cue["cue_id"],
            "kind": cue["kind"], "priority": cue["priority"], "team": cue.get("team"),
            "player_id": cue.get("player_id"), "text": self.narrator.template(cue, p), "lang": "en",
            "verified": None, "fallback": None, "issues": [],
            "evidence_event_ids": cue["explain"]["evidence_event_ids"], "data": cue["data"],
            "why": self.narrator._names(cue["explain"]["why"]), "headline": self.narrator._names(cue["headline"]), "player_name": cue.get("player_name"),
        }
        self.feeds[k].append(item)  # placeholder keeps on-screen order; text is filled in when ready

        def job():
            text, lang, fb, issues = self._say(
                facts, lambda: self.narrator.write(cue, p),
                lambda: self.narrator.template(cue, p), p.language)
            item.update(text=text, lang=lang, verified=not issues, fallback=fb, issues=issues)
            if k == self.echo:
                flag = "" if not issues else "  [verifier: " + "; ".join(issues) + "]"
                self._print(f"{item['clock']:>4} {text}{flag}")

        self._submit(job)

    def handle_recap(self, label):
        stats = self.engine.stats()
        cues = list(self.engine.cues)
        for k, p in self.profiles.items():
            facts = self.recapper.facts(label, stats, cues)
            rec = {"t": self.engine.t, "label": label}
            self.recaps[k].append(rec)

            def job(k=k, p=p, facts=facts, rec=rec):
                def draft():
                    r = self.recapper.write(label, stats, cues, p)
                    return r["text"], r["lang"]

                text, lang, fb, issues = self._say(facts, draft, lambda: self.recapper.template(facts, p),
                                                   p.language)
                rec.update(text=text, lang=lang, verified=not issues, fallback=fb, issues=issues)
                if k == self.echo:
                    self._print(f"\n=== {label} recap [{k}] ===\n{text}\n")

            self._submit(job)

    def run(self, folder: Path, speed: float = 0, echo: str | None = None, workers: int | None = None):
        import os
        from concurrent.futures import ThreadPoolExecutor

        self.echo = echo
        self._print_lock = __import__("threading").Lock()
        self.pending = []
        n = workers or int(os.getenv("MATCHROOM_WORKERS", "8"))
        self.pool = ThreadPoolExecutor(max_workers=n) if (self.llm.online and n > 1) else None
        t0, wall0 = None, time.time()
        try:
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
                for c in new:
                    self.handle_cue(c)
                if kind == "event" and self.ready:
                    for k, ps in self.pers.items():
                        for c in ps.periodic(self.engine.t, self.engine.stats(), self.engine.names):
                            self._emit(k, self.profiles[k], c)
                    if obj["type"] in ("half_time", "full_time"):
                        self.handle_recap("Half-time" if obj["type"] == "half_time" else "Full-time")
        finally:
            if self.pool:
                for f in self.pending:
                    f.result()
                self.pool.shutdown()
        self.wall_seconds = time.time() - wall0

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
    if orch.llm.backend == "foundry":
        print(orch.llm.summary())
        print(f"Total run time: {orch.wall_seconds:.0f} s")


if __name__ == "__main__":
    main()
