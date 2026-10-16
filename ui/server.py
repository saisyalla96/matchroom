"""
Matchroom Studio: live web UI.

    python -m ui.server                         # offline templates, sample match, http://localhost:8000
    python -m ui.server --backend foundry       # live Microsoft Foundry model
    python -m ui.server --data data/match_001 --speed 20 --port 8000

A background thread replays the synthetic match as if live, at the chosen speed, through the
insights engine and the agent team. The browser polls /api/state for the latest tracking frame,
score, stats and every viewer's personalised feed. Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agents.profiles import Profile
from agents.run import Orchestrator
from engine.replay import stream

ROOT = Path(__file__).parent
IDLE_SECONDS = float(os.getenv("MATCHROOM_IDLE_SECONDS", "30"))  # pause when no browser has polled for this long
EVIDENCE_KEYS = ("type", "team", "x", "y", "end_x", "end_y", "outcome", "player_name", "minute")


def make_profiles(meta: dict, focus_name: str | None) -> dict[str, Profile]:
    lineups = meta["lineups"]
    everyone = [p for lu in lineups.values() for p in lu]
    focus = next((p for p in everyone if focus_name and p["name"].lower() == focus_name.lower()), None)
    if not focus:  # default: the away side's striker
        focus = next(p for p in lineups[meta["away"]["id"]] if p["role"] == "ST")
    away = meta["away"]["id"]
    return {
        "casual": Profile("Casual fan", "casual"),
        "analyst": Profile("Analyst", "analyst"),
        "fan_es": Profile(f"{meta['away']['short']} fan, Spanish", "casual", "es", [away]),
        "player": Profile(f"Following {focus['name']}", "player_focus", "en", [], focus["id"]),
        "xg": Profile("xG tracker", "metric", "en", [], None, "xg"),
    }


class LiveMatch:
    def __init__(self, folder: Path, backend: str, speed: float, focus: str | None, inject_faults: int = 0):
        self.folder, self.backend, self.speed = folder, backend, speed
        meta = json.loads(open(folder / "events.jsonl", encoding="utf-8").readline())
        self.meta = meta
        self.profiles = make_profiles(meta, focus)
        self.orch = Orchestrator(self.profiles, backend, inject_faults)
        self.lock = threading.Lock()
        self.frame = None
        self.events: dict[str, dict] = {}
        self.run_id = f"{folder.name}-{int(time.time() * 1000)}"  # changes on restart or redeploy
        self.paused = False
        self.idle = False        # paused automatically because nobody is watching
        self.watchers = None     # callable returning seconds since the last browser poll
        self.finished = False
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        o = self.orch
        o.echo, o._print_lock, o.pending = None, threading.Lock(), []
        o.pool = ThreadPoolExecutor(max_workers=8) if o.llm.online else None
        match_t, wall = None, time.time()
        try:
            for t, kind, obj in stream(self.folder):
                while (self.paused or self._nobody_watching()) and not self.stopped:
                    time.sleep(0.1)
                    wall = time.time() - (t - match_t) / self.speed if match_t is not None else time.time()
                if self.stopped:
                    return
                if match_t is None:
                    match_t, wall = t, time.time()
                wait = (t - match_t) / self.speed - (time.time() - wall)
                if wait > 0:
                    time.sleep(wait)
                with self.lock:
                    if kind == "frame":
                        self.frame = obj
                        new = o.engine.ingest_frame(obj)
                    else:
                        self.events[obj["event_id"]] = {k: obj.get(k) for k in EVIDENCE_KEYS}
                        new = o.engine.ingest_event(obj)
                        if obj["type"] == "match_metadata":
                            o._setup()
                            continue
                    for c in new:
                        o.handle_cue(c)
                    if kind == "event" and o.ready:
                        stats = o.engine.stats()
                        for k, ps in o.pers.items():
                            for c in ps.periodic(o.engine.t, stats, o.engine.names):
                                o._emit(k, self.profiles[k], c)
                        if obj["type"] in ("half_time", "full_time"):
                            o.handle_recap("Half-time" if obj["type"] == "half_time" else "Full-time")
        finally:
            with self.lock:
                self.finished = True
            if o.pool:
                o.pool.shutdown(wait=False)

    def _nobody_watching(self) -> bool:
        """Cost guard for the public demo: no browser has polled recently, so stop replaying."""
        self.idle = bool(self.watchers and self.watchers() > IDLE_SECONDS)
        return self.idle

    def blocked(self) -> list[dict]:
        """Everything the Verifier stopped, across every viewer: what the model wrote, why, what aired instead."""
        with self.lock:
            o = self.orch
            out = []
            for k, p in self.profiles.items():
                for it in o.feeds[k] + o.recaps[k]:
                    if it.get("verified") is False:
                        out.append({"viewer": p.name, "clock": it.get("clock") or it.get("label"),
                                    "kind": it.get("kind", "recap"), "t": it.get("t", 0),
                                    "rejected": it.get("rejected"), "issues": it.get("issues", []),
                                    "aired": it.get("text"), "lang": it.get("lang")})
            return sorted(out, key=lambda x: -x["t"])

    @staticmethod
    def _ready(items, since):
        out = []
        for it in items[since:]:
            if it.get("verified") is None:  # still being written by the model: keep on-screen order
                break
            out.append(it)
        return out

    def state(self, cursors: dict[str, int], recap_cursors: dict[str, int]) -> dict:
        with self.lock:
            o, e = self.orch, self.orch.engine
            if not o.ready:
                return {"ready": False, "run_id": self.run_id}
            feeds, recaps = {}, {}
            for k in self.profiles:
                items = []
                for it in self._ready(o.feeds[k], cursors.get(k, 0)):
                    ev = [self.events[i] for i in it["evidence_event_ids"] if i in self.events]
                    items.append({kk: it.get(kk) for kk in (
                        "t", "clock", "kind", "priority", "team", "player_id", "player_name", "text", "lang",
                        "verified", "fallback", "issues", "rejected", "why", "headline", "data")} | {"evidence": ev})
                feeds[k] = items
                recaps[k] = [r for r in o.recaps[k][recap_cursors.get(k, 0):] if r.get("verified") is not None]
            s = e.stats()
            return {
                "ready": True, "run_id": self.run_id, "t": e.t, "clock": f"{int(e.t // 60)}'", "finished": self.finished,
                "paused": self.paused, "speed": self.speed, "score": s["score"], "momentum": s["momentum"],
                "regime": s["regime"], "teams": s["teams"], "frame": self.frame,
                "timeline": [{"m": x["minute"], "v": x["momentum"]} for x in e.timeline],
                "feeds": feeds, "recaps": recaps,
                "focus_player": self.profiles["player"].player_id,
                "model": {"backend": o.llm.backend, "label": o.llm.label, "calls": o.llm.calls,
                          "agent_calls": o.llm.agent_calls,
                          "faults_every": getattr(o.llm, "every", 0) if type(o.llm).__name__ == "FaultInjector" else 0,
                          "faults_injected": getattr(o.llm, "injected", 0) if type(o.llm).__name__ == "FaultInjector" else 0,
                          "verifier_checks": getattr(getattr(o, "verifier", None), "checked", 0),
                          "verifier_blocked": getattr(getattr(o, "verifier", None), "blocked", 0)},
            }


class App:
    def __init__(self, args):
        self.args = args
        self.folder = Path(args.data)
        self.last_seen = time.time()
        self.match = self._new(self.folder, args.speed)

    def _new(self, folder, speed):
        m = LiveMatch(folder, self.args.backend, speed, self.args.focus, self.args.inject_faults)
        m.watchers = lambda: time.time() - self.last_seen
        return m.start()

    def autoplay(self):
        """Public demo: after full time, wait 90 s then replay, so the URL is always showing a live match."""
        while True:
            time.sleep(5)
            m = self.match
            if m.finished and not getattr(m, "_done_at", None):
                m._done_at = time.time()
            watched = time.time() - self.last_seen < IDLE_SECONDS
            if getattr(m, "_done_at", None) and time.time() - m._done_at > 90 and m is self.match and watched:
                self.restart()

    def restart(self, new_match=False):
        now = time.time()
        if new_match and now - getattr(self, "_last_new", 0) < 60:
            new_match = False  # public demo guard: at most one freshly generated match per minute
        if new_match:
            self._last_new = now
        self.match.stopped = True
        if new_match:
            seed = random.randint(100, 9999)
            self.folder = Path("data") / f"live_{seed}"
            subprocess.run([sys.executable, "-m", "simulator.sim", "--seed", str(seed), "--out", str(self.folder)],
                           check=True, capture_output=True)
        self.match = self._new(self.folder, self.match.speed)

    def meta(self):
        m = self.match.meta
        return {"home": m["home"], "away": m["away"], "venue": m["venue"], "competition": m["competition"],
                "lineups": m["lineups"], "backend": self.args.backend,
                "profiles": {k: {"name": p.name, "mode": p.mode, "language": p.language, "clubs": p.clubs,
                                 "player_id": p.player_id, "metric": p.metric}
                             for k, p in self.match.profiles.items()}}


def handler_for(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                return self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/meta":
                return self._send(200, json.dumps(app.meta()).encode())
            if u.path == "/api/state":
                app.last_seen = time.time()
                cur = json.loads(q.get("c", "{}"))
                rc = json.loads(q.get("r", "{}"))
                return self._send(200, json.dumps(app.match.state(cur, rc), ensure_ascii=False).encode("utf-8"))
            if u.path == "/api/blocked":
                return self._send(200, json.dumps(app.match.blocked(), ensure_ascii=False).encode("utf-8"))
            if u.path == "/api/control":
                act = q.get("action")
                if act == "pause":
                    app.match.paused = True
                elif act == "play":
                    app.match.paused = False
                elif act == "speed":
                    app.match.speed = max(1.0, min(200.0, float(q.get("value", 20))))
                elif act == "restart":
                    app.restart()
                elif act == "new":
                    app.restart(new_match=True)
                return self._send(200, b'{"ok": true}')
            self._send(404, b'{"error": "not found"}')

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/match_001")
    ap.add_argument("--backend", choices=["offline", "foundry", "agents"], default=None)
    ap.add_argument("--speed", type=float, default=20)
    ap.add_argument("--focus", default="Elias Castell", help="player to follow in player-focus mode")
    ap.add_argument("--inject-faults", nargs="?", const=5, type=int, default=0, metavar="N",
                    help="demo: corrupt every Nth AI line on purpose (default 5) to show the Verifier catching it")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="0.0.0.0 inside a container")
    a = ap.parse_args()
    a.backend = a.backend or os.getenv("MATCHROOM_BACKEND", "offline")
    app = App(a)
    srv = ThreadingHTTPServer((a.host, a.port), handler_for(app))
    threading.Thread(target=app.autoplay, daemon=True).start()
    print(f"Matchroom Studio running at http://localhost:{a.port}  (backend: {a.backend}, {a.speed:g}x speed)")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
