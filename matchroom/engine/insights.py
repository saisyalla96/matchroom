"""
Matchroom real-time insights engine (pipeline stages b-d: interpret, explain, render).

Feed it events and tracking frames in time order. It keeps live match state and
returns *cues*: timed, machine-readable overlay items. Every cue carries an
`explain` block with the reason it matters and the event IDs that prove it, so
downstream AI agents can only narrate what the data supports.

Deterministic by design: no LLM here. Agents consume cues + state.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

WINDOW = 300  # rolling window, seconds (5 match minutes)
SPRINT_MS = 7.5  # 27 km/h sprint threshold
SPRINT_COOLDOWN = 90
SHOT_SPEED_KMH = 105
DISTANCE_MILESTONES_KM = (5, 10)
PASS_MILESTONES = (50, 100)
KEY_PASS_COOLDOWN = 300


def difficulty_rating(d: float) -> int:
    """Map 0..1 difficulty to a 1-10 broadcast rating."""
    return max(1, min(10, round(1 + 9 * d)))


def fmt_clock(t: float) -> str:
    return f"{int(t // 60)}'"


class MatchEngine:
    def __init__(self):
        self.meta = None
        self.teams: list[str] = []
        self.names: dict[str, str] = {}
        self.player_team: dict[str, str] = {}
        self.t = 0.0
        self.score = {}
        self.cues: list[dict] = []
        self._cid = 0
        self.poss_team = None
        self._last_t = 0.0
        self.poss_time = defaultdict(float)
        self.chain: list[dict] = []  # current possession sequence

        self.team = defaultdict(lambda: defaultdict(float))
        self.player = defaultdict(lambda: defaultdict(float))
        self.win = defaultdict(deque)  # team -> deque of (t, kind, value, event_id)

        self.dist = defaultdict(float)
        self.top_speed = defaultdict(float)
        self.match_top_speed = 0.0
        self._last_sprint = {}
        self._dist_ms = defaultdict(set)

        self.momentum = 0.0
        self._mom_leader = None
        self._last_swing = -1e9
        self._last_surge = defaultdict(lambda: -1e9)
        self.regime = "balanced"
        self._last_regime_t = -1e9
        self.timeline = []  # per-minute snapshots for charts
        self._last_key_pass = {}
        self._hot = defaultdict(int)

    # ------------------------------------------------------------------ cues
    def _cue(self, kind, t, headline, *, priority=2, duration=8, team=None, player_id=None,
             audience=("casual", "analyst"), data=None, why="", evidence=(), metrics=None):
        self._cid += 1
        cue = {
            "cue_id": f"CUE-{self._cid:05d}", "kind": kind,
            "t_in": round(t, 1), "t_out": round(t + duration, 1), "clock": fmt_clock(t),
            "priority": priority, "team": team, "player_id": player_id,
            "player_name": self.names.get(player_id) if player_id else None,
            "audience": list(audience), "headline": headline, "data": data or {},
            "explain": {"why": why, "evidence_event_ids": list(evidence), "metrics": metrics or {}},
        }
        self.cues.append(cue)
        return cue

    # ------------------------------------------------------------ windowing
    def _push(self, team, kind, value, eid):
        self.win[team].append((self.t, kind, value, eid))

    def _trim(self):
        for q in self.win.values():
            while q and q[0][0] < self.t - WINDOW:
                q.popleft()

    def _wsum(self, team, kind):
        return sum(v for (_, k, v, _) in self.win[team] if k == kind)

    def _wids(self, team, kind, limit=8):
        return [e for (_, k, _, e) in self.win[team] if k == kind][-limit:]

    def opp(self, team):
        return self.teams[1] if team == self.teams[0] else self.teams[0]

    # --------------------------------------------------------------- ingest
    def ingest_event(self, ev: dict) -> list[dict]:
        n0 = len(self.cues)
        et = ev["type"]
        if et == "match_metadata":
            self.meta = ev
            self.teams = [ev["home"]["id"], ev["away"]["id"]]
            self.score = {t: 0 for t in self.teams}
            for tid, lu in ev["lineups"].items():
                for p in lu:
                    self.names[p["id"]] = p["name"]
                    self.player_team[p["id"]] = tid
            return []

        # possession clock
        if self.poss_team:
            dt = max(0.0, ev["t"] - self._last_t)
            self.poss_time[self.poss_team] += dt
            self.win[self.poss_team].append((ev["t"], "poss", dt, None))
        self._last_t = ev["t"]
        self.t = ev["t"]
        self._trim()
        team, pid, eid = ev.get("team"), ev.get("player_id"), ev["event_id"]

        if et in ("pass", "carry", "shot", "kickoff") and team:
            if team != self.poss_team:
                self.chain = []
            self.poss_team = team
            self.chain.append(ev)

        handler = getattr(self, f"_on_{et}", None)
        if handler:
            handler(ev, team, pid, eid)
        if int(self.t) // 60 != int(self._last_snapshot_t()) // 60:
            self._snapshot()
        self._update_momentum(ev)
        self._update_regime(ev)
        return self.cues[n0:]

    def _last_snapshot_t(self):
        return self.timeline[-1]["t"] if self.timeline else -60

    # ----- per event type
    def _on_pass(self, ev, team, pid, eid):
        ok = ev["outcome"] == "complete"
        T, P = self.team[team], self.player[pid]
        T["passes"] += 1; P["passes"] += 1
        self._push(team, "pass_att", 1, eid)
        T["pass_dist"] += ev["distance_m"]; T["pass_diff"] += ev["difficulty"]
        if ok:
            T["passes_ok"] += 1; P["passes_ok"] += 1
            self._push(team, "pass_ok", 1, eid)
            prog = ev["progression_m"]
            # final-third entry (attacking third defined by progression into x>70 rel.)
            end_rel = ev["end_x"] if team == self.teams[0] else 105 - ev["end_x"]
            start_rel = ev["x"] if team == self.teams[0] else 105 - ev["x"]
            if end_rel > 70 >= start_rel:
                T["final_third_entries"] += 1
                self._push(team, "f3", 1, eid)
            if ev["difficulty"] >= 0.55 and prog >= 15:
                T["line_breaking"] += 1
            if (ev["difficulty"] >= 0.68 and prog >= 22 and ev["lane_clearance_m"] < 1.5
                    and self.t - self._last_key_pass.get(team, -1e9) > KEY_PASS_COOLDOWN):
                self._last_key_pass[team] = self.t
                r = difficulty_rating(ev["difficulty"])
                self._cue("key_pass", self.t,
                          f"Line-breaking pass by {ev['player_name']}: difficulty {r}/10",
                          team=team, player_id=pid, priority=2,
                          data={"distance_m": ev["distance_m"], "difficulty_rating": r,
                                "ball_speed_kmh": round(ev["ball_speed_ms"] * 3.6, 1),
                                "progression_m": prog, "recipient": ev["recipient_name"],
                                "lane_clearance_m": ev["lane_clearance_m"]},
                          why=(f"{ev['distance_m']} m pass moved play {prog:.0f} m forward; " + (
                               f"the nearest defender was just {ev['lane_clearance_m']} m from the passing lane."
                               if ev["lane_clearance_m"] >= 0.1 else "it went straight past a defender in the passing lane.")),
                          evidence=[eid], metrics={"difficulty": ev["difficulty"]})
            n = P["passes_ok"]
            if n in PASS_MILESTONES:
                acc = round(100 * P["passes_ok"] / P["passes"])
                self._cue("milestone", self.t, f"{ev['player_name']}: {int(n)} completed passes ({acc}% accuracy)",
                          team=team, player_id=pid, priority=3, audience=("analyst",),
                          data={"completed": int(n), "accuracy_pct": acc},
                          why="Volume of completed passes shows who is controlling the ball for their side.",
                          evidence=[eid])

    def _on_carry(self, ev, team, pid, eid):
        self.team[team]["carries"] += 1
        self.player[pid]["carry_m"] += ev["distance_m"]

    def _on_shot(self, ev, team, pid, eid):
        T, P = self.team[team], self.player[pid]
        T["shots"] += 1; P["shots"] += 1
        T["xg"] += ev["xg"]; P["xg"] += ev["xg"]
        if ev["outcome"] in ("goal", "saved"):
            T["on_target"] += 1
        T["max_shot_kmh"] = max(T["max_shot_kmh"], ev["shot_speed_kmh"])
        self._push(team, "xg", ev["xg"], eid)
        chain_ids = [e["event_id"] for e in self.chain[-8:]]
        n_pass = sum(1 for e in self.chain if e["type"] == "pass")
        if ev["outcome"] != "goal" and ev["xg"] >= 0.2:
            self._cue("big_chance", self.t,
                      f"Big chance: {ev['player_name']} ({ev['xg']:.2f} xG), {ev['outcome'].replace('_', ' ')}",
                      team=team, player_id=pid, priority=1,
                      data={"xg": ev["xg"], "distance_m": ev["distance_m"], "outcome": ev["outcome"],
                            "shot_speed_kmh": ev["shot_speed_kmh"]},
                      why=(f"A shot from {ev['distance_m']} m with {ev['defenders_close']} defender(s) close is "
                           f"converted about {round(ev['xg'] * 100)}% of the time; " +
                           (f"it came after a {n_pass}-pass move." if n_pass >= 2 else
                            "it came from a quick turnover or individual run.")),
                      evidence=chain_ids, metrics={"xg": ev["xg"], "chain_passes": n_pass})
        if ev["shot_speed_kmh"] >= SHOT_SPEED_KMH:
            self._cue("speed", self.t, f"Shot speed: {ev['shot_speed_kmh']} km/h ({ev['player_name']})",
                      team=team, player_id=pid, priority=3, duration=6,
                      data={"shot_speed_kmh": ev["shot_speed_kmh"]},
                      why="Shots above 105 km/h give a goalkeeper well under half a second to react from distance.",
                      evidence=[eid])

    def _on_goal(self, ev, team, pid, eid):
        self.score[team] += 1
        shot = next((e for e in reversed(self.chain) if e["type"] == "shot"), None)
        chain = self.chain[-10:]
        n_pass = sum(1 for e in chain if e["type"] == "pass")
        opp = self.opp(team)
        mom_against = (self.momentum < -0.25 and team == self.teams[0]) or (self.momentum > 0.25 and team == self.teams[1])
        xg = shot["xg"] if shot else ev.get("xg", 0)
        move = f"{n_pass}-pass move" if n_pass >= 2 else "quick break"
        why = (f"{'Against the run of play: a' if mom_against else 'A'} {move} finished from a "
               f"{xg:.2f} xG chance")
        why += " (a low-probability finish)." if xg < 0.12 else "."
        assist = next((e for e in reversed(chain) if e["type"] == "pass" and e["outcome"] == "complete"), None)
        self._cue("goal", self.t,
                  f"GOAL {team}: {ev['player_name']} | {self.score[self.teams[0]]}-{self.score[self.teams[1]]}",
                  team=team, player_id=pid, priority=1, duration=15,
                  data={"score": dict(self.score), "xg": xg, "assist": assist["player_name"] if assist else None,
                        "chain_passes": n_pass, "against_run_of_play": mom_against},
                  why=why, evidence=[e["event_id"] for e in chain] + [eid],
                  metrics={"xg": xg, "momentum": round(self.momentum, 2)})

    def _on_pressure(self, ev, team, pid, eid):
        self.team[team]["pressures"] += 1
        self._push(team, "press", 1, eid)
        opp = self.opp(team)
        n = self._wsum(team, "press")
        opp_poss = self._wsum(opp, "poss")
        if opp_poss < 60 or self.t < 600:
            return
        intensity = 60 * n / opp_poss  # pressures per minute of opponent possession
        base = 60 * self.team[team]["pressures"] / max(self.poss_time[opp], 1)
        hot = intensity > 1.35 * base and n >= 12
        self._hot[team] = self._hot[team] + 1 if hot else 0
        if self._hot[team] >= 4 and self.t - self._last_surge[team] > 480:
            self._last_surge[team] = self.t
            self._cue("pressure_surge", self.t, f"{team} have turned up the press",
                      team=team, priority=2, duration=12,
                      data={"pressures_last_5": int(n), "pressures_per_opp_min": round(intensity, 1),
                            "match_avg_per_opp_min": round(base, 1)},
                      why=(f"{team} are pressing {intensity:.1f} times per minute of {opp} possession over the last "
                           f"5 minutes, up from a match average of {base:.1f}. {opp} have less time on the ball."),
                      evidence=self._wids(team, "press"), metrics={"ratio": round(intensity / base, 2)})

    def _on_tackle(self, ev, team, pid, eid):
        if ev["outcome"] == "won":
            self.team[team]["tackles_won"] += 1
            self.player[pid]["tackles_won"] += 1

    def _on_possession_change(self, ev, team, pid, eid):
        self.team[team]["recoveries"] += 1
        self._push(self.opp(team), "turnover", 1, eid)
        self.chain = []
        self.poss_team = team

    # -------------------------------------------------- derived explainers
    def _team_pressure_score(self, team):
        return (self._wsum(team, "xg") * 10 + self._wsum(team, "f3") * 0.8
                + self._wsum(team, "press") * 0.08 + self._wsum(team, "pass_ok") * 0.02)

    def _update_momentum(self, ev):
        if len(self.teams) < 2 or self.t < 240:
            return
        a, b = (self._team_pressure_score(t) for t in self.teams)
        self.momentum = round((a - b) / (a + b + 1e-6), 3)
        leader = self.teams[0] if self.momentum > 0.3 else self.teams[1] if self.momentum < -0.3 else None
        if leader and leader != self._mom_leader and self.t - self._last_swing > 360:
            prev = self._mom_leader
            self._mom_leader, self._last_swing = leader, self.t
            ev_ids = self._wids(leader, "f3", 4) + self._wids(leader, "xg", 4)
            self._cue("momentum_swing", self.t, f"Momentum swings to {leader}",
                      team=leader, priority=2, duration=12,
                      data={"momentum": self.momentum, "previous_leader": prev,
                            "xg_last_5": round(self._wsum(leader, "xg"), 2),
                            "final_third_entries_last_5": int(self._wsum(leader, "f3"))},
                      why=(f"In the last 5 minutes {leader} have {int(self._wsum(leader, 'f3'))} final-third "
                           f"entries and {self._wsum(leader, 'xg'):.2f} xG vs "
                           f"{int(self._wsum(self.opp(leader), 'f3'))} and "
                           f"{self._wsum(self.opp(leader), 'xg'):.2f} for {self.opp(leader)}."),
                      evidence=ev_ids, metrics={"momentum_index": self.momentum})
        elif not leader and abs(self.momentum) < 0.15:
            self._mom_leader = None

    def _update_regime(self, ev):
        """Control vs chaos: turnover rate relative to completed passes in the window."""
        if len(self.teams) < 2 or self.t < 300:
            return
        to = sum(self._wsum(t, "turnover") for t in self.teams)
        ok = sum(self._wsum(t, "pass_ok") for t in self.teams)
        ratio = to / (ok + to + 1e-6)
        regime = "chaos" if ratio > 0.30 else "control" if ratio < 0.16 else "balanced"
        if regime != self.regime and self.t - self._last_regime_t > 420:
            self.regime, self._last_regime_t = regime, self.t
            if regime == "balanced":
                return
            ids = []
            for t in self.teams:
                ids += self._wids(t, "turnover", 4)
            dom = max(self.teams, key=lambda t: self._wsum(t, "pass_ok"))
            self._cue("game_state", self.t,
                      "The game has broken up: chaos phase" if regime == "chaos"
                      else f"Control phase: {dom} dictating with the ball",
                      team=None if regime == "chaos" else dom, priority=2, duration=12,
                      data={"regime": regime, "turnover_ratio": round(ratio, 2),
                            "turnovers_last_5": int(to), "completed_passes_last_5": int(ok)},
                      why=(f"{int(to)} turnovers vs {int(ok)} completed passes in 5 minutes "
                           f"({round(ratio * 100)}% of ball actions ending in a turnover)."),
                      evidence=ids, metrics={"turnover_ratio": round(ratio, 3)})

    # ------------------------------------------------------------ tracking
    def ingest_frame(self, fr: dict) -> list[dict]:
        n0 = len(self.cues)
        t = fr["t"]
        for p in fr["players"]:
            pid, v = p["id"], p["v"]
            before = self.dist[pid]
            self.dist[pid] += v
            self.top_speed[pid] = max(self.top_speed[pid], v)
            for km in DISTANCE_MILESTONES_KM:
                if before < km * 1000 <= self.dist[pid] and km not in self._dist_ms[pid]:
                    self._dist_ms[pid].add(km)
                    if km == 10:
                        self._cue("player_tag", t, f"{self.names.get(pid, pid)} passes 10 km covered",
                                  team=self.player_team.get(pid), player_id=pid, priority=3, duration=6,
                                  data={"distance_km": round(self.dist[pid] / 1000, 2)},
                                  why="Distance covered shows work rate, and 10 km is a typical full-match benchmark.")
            if v >= SPRINT_MS and t - self._last_sprint.get(pid, -1e9) > SPRINT_COOLDOWN:
                self._last_sprint[pid] = t
                record = v > self.match_top_speed + 0.05 and t > 600
                self.match_top_speed = max(self.match_top_speed, v)
                if record:
                    self._cue("player_tag", t,
                              f"{self.names.get(pid, pid)}: {v * 3.6:.1f} km/h, fastest of the match",
                              team=self.player_team.get(pid), player_id=pid, priority=2, duration=6,
                              data={"speed_kmh": round(v * 3.6, 1), "x": p["x"], "y": p["y"],
                                    "distance_km": round(self.dist[pid] / 1000, 2)},
                              why="Top-speed record for this match, triggered by the 27 km/h sprint threshold.")
        return self.cues[n0:]

    # ------------------------------------------------------------- outputs
    def _snapshot(self):
        tot = sum(self.poss_time.values()) or 1
        self.timeline.append({
            "t": self.t, "minute": int(self.t // 60), "momentum": self.momentum, "regime": self.regime,
            "score": dict(self.score),
            "possession_pct": {t: round(100 * self.poss_time[t] / tot, 1) for t in self.teams},
            "xg": {t: round(self.team[t]["xg"], 2) for t in self.teams},
        })

    def stats(self) -> dict:
        tot = sum(self.poss_time.values()) or 1
        out = {"t": self.t, "score": dict(self.score), "momentum": self.momentum, "regime": self.regime,
               "teams": {}, "players": {}}
        for t in self.teams:
            T = self.team[t]
            out["teams"][t] = {
                "possession_pct": round(100 * self.poss_time[t] / tot, 1),
                "passes": int(T["passes"]),
                "pass_accuracy_pct": round(100 * T["passes_ok"] / max(T["passes"], 1), 1),
                "avg_pass_distance_m": round(T["pass_dist"] / max(T["passes"], 1), 1),
                "avg_pass_difficulty": difficulty_rating(T["pass_diff"] / max(T["passes"], 1)),
                "line_breaking_passes": int(T["line_breaking"]),
                "final_third_entries": int(T["final_third_entries"]),
                "shots": int(T["shots"]), "on_target": int(T["on_target"]), "xg": round(T["xg"], 2),
                "max_shot_speed_kmh": T["max_shot_kmh"], "pressures": int(T["pressures"]),
                "tackles_won": int(T["tackles_won"]), "recoveries": int(T["recoveries"]),
            }
        for pid, name in self.names.items():
            P = self.player[pid]
            out["players"][pid] = {
                "name": name, "team": self.player_team[pid],
                "passes": int(P["passes"]),
                "pass_accuracy_pct": round(100 * P["passes_ok"] / P["passes"], 1) if P["passes"] else None,
                "shots": int(P["shots"]), "xg": round(P["xg"], 2), "tackles_won": int(P["tackles_won"]),
                "distance_km": round(self.dist[pid] / 1000, 2),
                "top_speed_kmh": round(self.top_speed[pid] * 3.6, 1),
            }
        return out
