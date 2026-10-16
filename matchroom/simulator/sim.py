"""
Matchroom synthetic match simulator.

Produces football-realistic, fully synthetic match data:
  - events.jsonl    : event stream (passes, carries, shots, tackles, pressure,
                      possession changes, goals, match metadata)
  - tracking.jsonl  : 1 Hz positions + speeds for ball and all 22 players
  - truth.json      : hidden ground-truth labels (tactical phases), useful for
                      evaluating whether the explainability layer detects them

All clubs, players and venues are fictional. No real Premier League data is used.

Usage:
    python -m simulator.sim --seed 7 --out data/match_001
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

PITCH_X, PITCH_Y = 105.0, 68.0
HALF = 45 * 60

# 4-3-3 base shape in team-relative coords (team attacks +x)
FORMATION = [
    ("GK", 5, 34), ("LB", 25, 10), ("CB", 22, 26), ("CB", 22, 42), ("RB", 25, 58),
    ("CM", 45, 20), ("DM", 40, 34), ("CM", 45, 48),
    ("LW", 70, 12), ("ST", 76, 34), ("RW", 70, 56),
]

FIRST = ["Leo", "Kai", "Mateo", "Arlo", "Noah", "Idris", "Tomas", "Rafe", "Jonah", "Elias",
         "Oskar", "Dario", "Felix", "Nico", "Sami", "Bram", "Caio", "Ravi", "Luka", "Theo",
         "Milo", "Ezra", "Hugo", "Aren"]
LAST = ["Halden", "Okafor", "Vance", "Moreno", "Kessler", "Abara", "Lindqvist", "Duarte",
        "Ashby", "Novak", "Petrov", "Rinaldi", "Castell", "Mbeki", "Sorensen", "Quill",
        "Varga", "Hollis", "Iwata", "Brandt", "Achebe", "Ferro", "Laine", "Strand"]

CLUBS = [
    {"id": "RVA", "name": "Riverside Athletic", "short": "Riverside", "colour": "#1E5AA8"},
    {"id": "NGC", "name": "Northgate City", "short": "Northgate", "colour": "#C8102E"},
]


@dataclass
class Player:
    pid: str
    name: str
    number: int
    role: str
    team: int  # 0 home, 1 away
    base: tuple
    passing: float
    shooting: float
    pace: float  # top speed m/s
    x: float = 0.0
    y: float = 0.0
    speed: float = 0.0


@dataclass
class TeamStyle:
    press: float   # 0..1 pressing intensity
    tempo: float   # 0..1 directness/risk


@dataclass
class Sim:
    seed: int
    rng: random.Random = field(init=False)
    players: list = field(default_factory=list)
    events: list = field(default_factory=list)
    tracking: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    t: float = 0.0
    score: list = field(default_factory=lambda: [0, 0])
    eid: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.match_id = f"SYN-{self.seed:04d}"
        names = [(f, l) for f in FIRST for l in LAST]
        self.rng.shuffle(names)
        for team in (0, 1):
            for i, (role, bx, by) in enumerate(FORMATION):
                f, l = names.pop()
                r = self.rng
                p = Player(
                    pid=f"{CLUBS[team]['id']}-{i + 1:02d}", name=f"{f} {l}",
                    number=[1, 3, 4, 5, 2, 8, 6, 10, 11, 9, 7][i], role=role, team=team,
                    base=(bx, by),
                    passing=r.uniform(0.80, 0.95) if role in ("CM", "DM") else r.uniform(0.70, 0.90),
                    shooting=r.uniform(0.9, 1.25) if role in ("ST", "LW", "RW") else r.uniform(0.6, 0.95),
                    pace=r.uniform(8.6, 9.8) if role in ("LW", "RW", "ST", "LB", "RB") else r.uniform(7.6, 9.0),
                )
                self.players.append(p)
        self.styles = [TeamStyle(0.5, 0.5), TeamStyle(0.5, 0.5)]
        self._schedule_phases()
        self._reset_positions()

    # ---------- geometry helpers ----------
    @staticmethod
    def to_abs(team, rx, ry):
        return (rx, ry) if team == 0 else (PITCH_X - rx, PITCH_Y - ry)

    @staticmethod
    def to_rel(team, ax, ay):
        return (ax, ay) if team == 0 else (PITCH_X - ax, PITCH_Y - ay)

    def team_players(self, team):
        return [p for p in self.players if p.team == team]

    def _reset_positions(self):
        for p in self.players:
            rx, ry = p.base
            rx = min(rx * 0.55, 50)  # own half for kickoff
            p.x, p.y = self.to_abs(p.team, rx, ry)
            p.speed = 0.0

    # ---------- tactical phases (hidden ground truth) ----------
    def _schedule_phases(self):
        r = self.rng
        for team in (0, 1):
            t = 0
            while t < 2 * HALF:
                dur = r.uniform(12, 25) * 60
                kind = r.choice(["high_press", "low_block", "direct", "patient", "balanced"])
                self.phases.append({"team": team, "start": round(t), "end": round(min(t + dur, 2 * HALF)), "phase": kind})
                t += dur

    def _apply_phase(self):
        for team in (0, 1):
            ph = next(p for p in self.phases if p["team"] == team and p["start"] <= self.t < p["end"]) \
                if self.t < 2 * HALF else {"phase": "balanced"}
            self.styles[team] = {
                "high_press": TeamStyle(0.9, 0.55), "low_block": TeamStyle(0.15, 0.5),
                "direct": TeamStyle(0.5, 0.9), "patient": TeamStyle(0.45, 0.15),
                "balanced": TeamStyle(0.5, 0.5),
            }[ph["phase"]]

    # ---------- tracking ----------
    def _advance(self, dt, ball):
        """Move players toward shape targets for dt seconds; emit 1 Hz frames."""
        steps = max(1, int(round(dt)))
        bx0, by0 = self._ball_prev
        for s in range(steps):
            frac = (s + 1) / steps
            bx = bx0 + (ball[0] - bx0) * frac
            by = by0 + (ball[1] - by0) * frac
            for team in (0, 1):
                tps = self.team_players(team)
                brx, bry = self.to_rel(team, bx, by)
                in_poss = team == self.poss
                press = self.styles[team].press
                chasers = sorted(tps, key=lambda p: (p.x - bx) ** 2 + (p.y - by) ** 2)
                n_chase = 1 if in_poss else (1 + int(press > 0.7))
                for p in tps:
                    if p in chasers[:n_chase] and p.role != "GK":
                        tx, ty = bx, by
                    else:
                        rx = p.base[0] + (brx - 52.5) * 0.45 + (6 if in_poss else -4)
                        if not in_poss:
                            rx += (press - 0.5) * 18
                        if p.role == "GK":
                            rx = min(p.base[0] + max(0, brx - 52.5) * 0.15, 18)
                        ry = p.base[1] + (bry - 34) * 0.3
                        tx, ty = self.to_abs(team, max(2, min(103, rx)), max(2, min(66, ry)))
                    dx, dy = tx - p.x, ty - p.y
                    d = math.hypot(dx, dy)
                    vmax = p.pace * self.rng.uniform(0.92, 1.0)
                    chasing = (tx, ty) == (bx, by)
                    v = min(d * (0.8 if chasing else 0.22), vmax)
                    if d > 0:
                        p.x += dx / d * v
                        p.y += dy / d * v
                    p.speed = v
            self.t += dt / steps
            self.tracking.append({
                "t": round(self.t, 1),
                "ball": {"x": round(bx, 1), "y": round(by, 1)},
                "players": [{"id": p.pid, "x": round(p.x, 1), "y": round(p.y, 1), "v": round(p.speed, 2)}
                            for p in self.players],
            })
        self._ball_prev = ball

    # ---------- events ----------
    def emit(self, etype, team=None, player=None, x=None, y=None, **attrs):
        self.eid += 1
        ev = {
            "event_id": f"{self.match_id}-{self.eid:05d}", "match_id": self.match_id,
            "t": round(self.t, 1), "minute": int(self.t // 60) + 1,
            "period": 1 if self.t < HALF else 2, "type": etype,
            "team": CLUBS[team]["id"] if team is not None else None,
            "player_id": player.pid if player else None,
            "player_name": player.name if player else None,
            "x": round(x, 1) if x is not None else None,
            "y": round(y, 1) if y is not None else None,
            "score": list(self.score),
        }
        ev.update(attrs)
        self.events.append(ev)
        return ev

    def _pressure_on(self, x, y, team):
        opps = self.team_players(1 - team)
        dists = sorted(((math.hypot(o.x - x, o.y - y), o) for o in opps), key=lambda d: d[0])
        n_close = sum(1 for d, _ in dists if d < 5)
        return n_close, dists[0][0], dists[0][1]

    @staticmethod
    def _seg_dist(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy or 1e-6
        t = max(0, min(1, ((px - ax) * vx + (py - ay) * vy) / L2))
        return math.hypot(px - (ax + t * vx), py - (ay + t * vy))

    def _xg(self, team, x, y, pressure):
        rx, ry = self.to_rel(team, x, y)
        gx, gy = PITCH_X - rx, ry - 34
        dist = math.hypot(gx, gy)
        a1 = math.atan2(gy + 3.66, gx)
        a2 = math.atan2(gy - 3.66, gx)
        angle = abs(a1 - a2)
        logit = -1.0 - 0.11 * dist + 1.6 * angle - 0.3 * pressure
        return 1 / (1 + math.exp(-logit)), dist

    def kickoff(self, team):
        self._reset_positions()
        self.poss = team
        st = next(p for p in self.team_players(team) if p.role == "ST")
        st.x, st.y = 52.5, 34
        self.holder = st
        self.ball = (52.5, 34)
        self._ball_prev = self.ball
        self.emit("kickoff", team, st, 52.5, 34)

    def run(self):
        self.emit("match_metadata", None, None, competition="Synthetic Showcase League",
                  venue="Harbourside Stadium (fictional)",
                  home=CLUBS[0], away=CLUBS[1],
                  lineups={CLUBS[t]["id"]: [{"id": p.pid, "name": p.name, "number": p.number, "role": p.role}
                                            for p in self.team_players(t)] for t in (0, 1)})
        self.kickoff(0)
        half_done = False
        end = 2 * HALF + self.rng.uniform(60, 240)
        while self.t < end:
            if not half_done and self.t >= HALF + self.rng.uniform(30, 120):
                self.emit("half_time", None, None)
                self.t = HALF
                self.kickoff(1)
                half_done = True
                continue
            self._apply_phase()
            self.step()
        self.emit("full_time", None, None)

    def step(self):
        r = self.rng
        team, h = self.poss, self.holder
        bx, by = self.ball
        n_close, nearest_d, nearest = self._pressure_on(bx, by, team)
        dstyle = self.styles[1 - team]
        rx, _ = self.to_rel(team, bx, by)

        # pressure event from defending team
        if nearest_d < 4 and r.random() < 0.08 + 0.45 * dstyle.press:
            self.emit("pressure", 1 - team, nearest, nearest.x, nearest.y,
                      on_player=h.pid, distance_m=round(nearest_d, 1))

        # tackle attempt
        if nearest_d < 2.5 and r.random() < 0.03 + 0.07 * dstyle.press:
            won = r.random() < 0.55
            self.emit("tackle", 1 - team, nearest, bx, by, outcome="won" if won else "lost", on_player=h.pid)
            self._advance(r.uniform(1, 2), self.ball)
            if won:
                self._change_poss(nearest, "tackle")
                return

        # shot decision
        xg, dist = self._xg(team, bx, by, n_close)
        shoot_p = 0.0 if rx < 72 else min(0.45, xg * 0.9 + (0.015 if dist < 28 else 0.0)) * h.shooting * 0.55
        if r.random() < shoot_p:
            self._shot(h, xg, dist, n_close)
            return

        if r.random() < 0.2:
            self._carry(h)
        else:
            self._pass(h, n_close)

    def _carry(self, h):
        r = self.rng
        team = self.poss
        rx, ry = self.to_rel(team, *self.ball)
        d = r.uniform(3, 14)
        nrx = min(100, rx + d * r.uniform(0.4, 1.0))
        nry = max(3, min(65, ry + r.uniform(-6, 6)))
        nx, ny = self.to_abs(team, nrx, nry)
        dur = d / r.uniform(4.5, 7.5)
        self.emit("carry", team, h, *self.ball, end_x=round(nx, 1), end_y=round(ny, 1),
                  distance_m=round(math.hypot(nx - self.ball[0], ny - self.ball[1]), 1))
        self._advance(dur, (nx, ny))
        h.x, h.y = nx, ny
        self.ball = (nx, ny)

    def _pass(self, h, n_close):
        r = self.rng
        team = self.poss
        tempo = self.styles[team].tempo
        bx, by = self.ball
        hrx, _ = self.to_rel(team, bx, by)
        mates = [p for p in self.team_players(team) if p is not h]
        weights = []
        for m in mates:
            d = math.hypot(m.x - bx, m.y - by)
            mrx, _ = self.to_rel(team, m.x, m.y)
            prog = mrx - hrx
            w = math.exp(-((d - 16) ** 2) / (2 * (10 + 25 * tempo) ** 2))
            w *= math.exp(prog / (40 - 25 * tempo))
            if m.role == "GK":
                w *= 0.15
            weights.append(max(w, 1e-4))
        rec = r.choices(mates, weights)[0]
        ex, ey = rec.x + r.uniform(-1.5, 1.5), rec.y + r.uniform(-1.5, 1.5)
        dist = math.hypot(ex - bx, ey - by)
        erx, _ = self.to_rel(team, ex, ey)
        prog = erx - hrx
        opps = self.team_players(1 - team)
        lane = min(self._seg_dist(o.x, o.y, bx, by, ex, ey) for o in opps)
        rec_press = sum(1 for o in opps if math.hypot(o.x - ex, o.y - ey) < 4)
        difficulty = (min(dist, 50) / 50) * 0.35 + min(n_close, 3) * 0.08 + max(0, prog) / 45 * 0.2 \
            + (0.25 if lane < 2 else 0.1 if lane < 4 else 0) + rec_press * 0.06
        difficulty = max(0.02, min(1.0, difficulty))
        p_success = max(0.2, min(0.98, h.passing * (1 - 0.4 * difficulty) + 0.17))
        ok = r.random() < p_success
        speed = min(32, 7 + dist * 0.38 + r.uniform(-1.5, 2.5))
        dur = dist / speed + r.uniform(0.6, 1.6)
        ev = self.emit("pass", team, h, bx, by, end_x=round(ex, 1), end_y=round(ey, 1),
                       recipient_id=rec.pid, recipient_name=rec.name,
                       outcome="complete" if ok else "incomplete",
                       distance_m=round(dist, 1), ball_speed_ms=round(speed, 1),
                       progression_m=round(prog, 1), lane_clearance_m=round(lane, 1),
                       difficulty=round(difficulty, 3))
        if ok:
            self._advance(dur, (ex, ey))
            self.holder = rec
            rec.x, rec.y = ex, ey
            self.ball = (ex, ey)
        else:
            # nearest opponent to the lane wins it
            win = min(opps, key=lambda o: self._seg_dist(o.x, o.y, bx, by, ex, ey))
            pt = (win.x, win.y)
            self._advance(dur, pt)
            self.ball = pt
            self._change_poss(win, "interception", ref=ev["event_id"])

    def _shot(self, h, xg, dist, n_close):
        r = self.rng
        team = self.poss
        bx, by = self.ball
        speed = min(37, r.uniform(15, 21) + dist * 0.3 + r.uniform(0, 5))
        roll = r.random()
        if roll < xg:
            outcome = "goal"
        else:
            outcome = r.choices(["saved", "off_target", "blocked"], [0.4, 0.35, 0.25])[0]
        gx, _ = self.to_abs(team, PITCH_X, 34)
        self.emit("shot", team, h, bx, by, end_x=gx, end_y=round(34 + r.uniform(-4, 4), 1),
                  outcome=outcome, xg=round(xg, 3), distance_m=round(dist, 1),
                  shot_speed_ms=round(speed, 1), shot_speed_kmh=round(speed * 3.6, 1),
                  defenders_close=n_close)
        self._advance(r.uniform(1.5, 3), (gx, 34))
        if outcome == "goal":
            self.score[team] += 1
            self.emit("goal", team, h, bx, by, xg=round(xg, 3))
            self._advance(r.uniform(40, 70), (52.5, 34))
            self.kickoff(1 - team)
        else:
            gk = next(p for p in self.team_players(1 - team) if p.role == "GK")
            if outcome == "saved":
                self.emit("save", 1 - team, gk, gk.x, gk.y, shot_xg=round(xg, 3))
            self._advance(r.uniform(4, 10), (gk.x, gk.y))
            self.ball = (gk.x, gk.y)
            self._change_poss(gk, "goal_kick" if outcome == "off_target" else outcome)

    def _change_poss(self, new_holder, cause, ref=None):
        self.poss = new_holder.team
        self.holder = new_holder
        self.ball = (new_holder.x, new_holder.y)
        self.emit("possession_change", new_holder.team, new_holder, *self.ball, cause=cause, ref_event=ref)


def write(sim: Sim, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "events.jsonl", "w") as f:
        for e in sim.events:
            f.write(json.dumps(e) + "\n")
    with open(out / "tracking.jsonl", "w") as f:
        for fr in sim.tracking:
            f.write(json.dumps(fr) + "\n")
    with open(out / "truth.json", "w") as f:
        json.dump({"match_id": sim.match_id, "seed": sim.seed, "final_score": sim.score,
                   "tactical_phases": [dict(p, team=CLUBS[p["team"]]["id"]) for p in sim.phases]}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/match_001")
    a = ap.parse_args()
    sim = Sim(a.seed)
    sim.run()
    write(sim, Path(a.out))
    print(f"{sim.match_id}: {CLUBS[0]['short']} {sim.score[0]}-{sim.score[1]} {CLUBS[1]['short']} | "
          f"{len(sim.events)} events, {len(sim.tracking)} tracking frames -> {a.out}")


if __name__ == "__main__":
    main()
