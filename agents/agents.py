"""
The Matchroom agent team (pipeline stage e: personalise, plus narrative and recap).

  Personaliser  decides WHAT each viewer sees (filtering, priority, periodic cards)
  Narrator      decides HOW it is said, in the viewer's mode
  Verifier      checks every number and player name against the grounded facts
  RecapWriter   half-time / full-time recaps per audience
  Localiser     translates while preserving names and numbers

The engine computes; agents communicate. With the Foundry backend the Narrator,
RecapWriter and Localiser use the model, and the Verifier blocks any output that
adds a number or name not present in the facts (falling back to a safe template).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import uuid

from agents.llm import LLM
from agents.profiles import Profile

NUM_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")

METRIC_KINDS = {
    "xg": {"goal", "big_chance"},
    "possession": {"goal", "game_state", "momentum_swing"},
    "pass_accuracy": {"goal", "key_pass", "milestone"},
    "pressures": {"goal", "pressure_surge", "game_state"},
    "distance": {"goal", "player_tag"},
}
CASUAL_KINDS = {"goal", "big_chance", "momentum_swing", "game_state", "pressure_surge", "speed", "player_tag"}


# ------------------------------------------------------------------ helpers
def numbers_in(text: str) -> set[float]:
    out = set()
    for m in NUM_RE.findall(text):
        try:
            out.add(float(m.replace(",", ".")))
        except ValueError:
            pass
    return out


UNIT_RE = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)?)\s*(km/h|kmh|km|metres|metre|meters|meter|metros|metro|m(?![\w])|%)?",
                     re.IGNORECASE)
UNIT_OF = {"km/h": "kmh", "kmh": "kmh", "km": "km", "%": "pct"}


def _unit(u: str | None) -> str | None:
    if not u:
        return None
    u = u.lower()
    return UNIT_OF.get(u, "m")


def _walk(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield key, float(obj)


def unit_numbers(facts) -> dict[str, set[float]]:
    """Numbers in the facts grouped by the unit their key implies."""
    out = {"kmh": set(), "km": set(), "m": set(), "pct": set()}
    for k, n in _walk(facts):
        k = k.lower()
        if "kmh" in k:
            u, vals = "kmh", [n]
        elif k.endswith("_km"):
            u, vals = "km", [n]
        elif k.endswith("_m") or k == "distance":
            u, vals = "m", [n]
        elif "pct" in k or "xg" in k or "ratio" in k:
            u, vals = "pct", [n, n * 100]
        else:
            continue
        for v in vals:
            out[u].update({round(v), round(v, 1), round(v, 2)})
    # numbers quoted inside text facts ("why", "headline") keep their written unit
    text = json.dumps(facts, ensure_ascii=False)
    for num, unit in UNIT_RE.findall(text):
        u = _unit(unit)
        if u:
            v = float(num.replace(",", "."))
            out[u].update({round(v), round(v, 1), round(v, 2)})
    return out


def allowed_numbers(facts) -> set[float]:
    """Every number in the facts, plus common presentational transforms of it."""
    base = numbers_in(json.dumps(facts, ensure_ascii=False))
    out = set()
    for n in base:
        for v in (n, n * 100, n / 1000):  # 0.23 -> 23%, metres -> km
            out.update({round(v), round(v, 1), round(v, 2)})
    return out


# --------------------------------------------------------------- Verifier
class Verifier:
    """Deterministic guardrail: no new numbers, no misattributed players."""

    def __init__(self, roster: dict[str, str]):
        self.full = set(roster.values())
        last = {}
        for n in self.full:
            s = n.split()[-1]
            last[s] = last.get(s, 0) + 1
        self.unique_last = {s for s, c in last.items() if c == 1}
        self.blocked = 0
        self.checked = 0

    def check(self, text: str, facts) -> list[str]:
        self.checked += 1
        issues = []
        ok_nums = allowed_numbers(facts)
        by_unit = unit_numbers(facts)
        for num, unit in UNIT_RE.findall(text):
            n = float(num.replace(",", "."))
            u = _unit(unit)
            pool = by_unit[u] if u else ok_nums
            if not any(abs(n - a) < 1e-6 for a in pool):
                issues.append(f"number {n:g}{' ' + unit if unit else ''} not in facts")
        facts_s = json.dumps(facts, ensure_ascii=False)
        for name in self.full:
            if name in text and name not in facts_s:
                issues.append(f"player '{name}' not in facts")
        for s in self.unique_last:
            if re.search(rf"\b{re.escape(s)}\b", text) and s not in facts_s:
                issues.append(f"player '{s}' not in facts")
        if issues:
            self.blocked += 1
        return issues


# ------------------------------------------------------------ Personaliser
class Personaliser:
    def __init__(self, profile: Profile):
        self.p = profile
        self._last_minor = -1e9
        self._last_card = 0.0
        self.focus_name = ""

    def wants(self, cue: dict) -> bool:
        p, k = self.p, cue["kind"]
        fav = cue.get("team") in p.clubs
        if p.mode == "analyst":
            return True
        if p.mode == "player_focus":
            d = json.dumps(cue["data"])
            return k == "goal" or cue.get("player_id") == p.player_id or bool(self.focus_name and self.focus_name in d)
        if p.mode == "metric":
            return k in METRIC_KINDS[p.metric]
        # casual
        if k not in CASUAL_KINDS and not (fav and k in ("key_pass", "milestone")):
            return False
        if k == "player_tag" and "fastest" not in cue["headline"]:
            return False
        if cue["priority"] == 1:
            return True
        if cue["t_in"] - self._last_minor < 180 and not fav:
            return False
        self._last_minor = cue["t_in"]
        return True

    def periodic(self, t: float, stats: dict, names: dict) -> list[dict]:
        """Player cards / metric tickers every few minutes (synthetic, grounded cues)."""
        p = self.p
        every = 900 if p.mode == "player_focus" else 600 if p.mode == "metric" else None
        if not every or t - self._last_card < every:
            return []
        self._last_card = t
        clock = f"{int(t // 60)}'"
        if p.mode == "player_focus":
            s = stats["players"][p.player_id]
            data = {k: s[k] for k in ("passes", "pass_accuracy_pct", "shots", "xg", "tackles_won",
                                      "distance_km", "top_speed_kmh")}
            return [_card("player_card", t, clock, f"{s['name']} at {clock}", data, s["team"], p.player_id,
                          s["name"], "Running player report for player-focus mode.")]
        teams = stats["teams"]
        key = {"xg": "xg", "possession": "possession_pct", "pass_accuracy": "pass_accuracy_pct",
               "pressures": "pressures"}.get(p.metric)
        if p.metric == "distance":
            top = sorted(stats["players"].values(), key=lambda x: -x["distance_km"])[:3]
            data = {x["name"]: x["distance_km"] for x in top}
            head = f"Distance leaders at {clock}"
        else:
            data = {tid: v[key] for tid, v in teams.items()}
            label = {"xg": "xG", "possession": "Possession %", "pass_accuracy": "Pass accuracy %",
                     "pressures": "Pressures"}[p.metric]
            head = f"{label} at {clock}"
        return [_card("metric_ticker", t, clock, head, data, None, None, None,
                      f"Live {p.metric} tracker for metric mode.")]


def _card(kind, t, clock, headline, data, team, pid, pname, why):
    return {"cue_id": f"CARD-{uuid.uuid4().hex[:8]}", "kind": kind, "t_in": round(t, 1),
            "t_out": round(t + 10, 1), "clock": clock, "priority": 3, "team": team, "player_id": pid,
            "player_name": pname, "audience": ["all"], "headline": headline, "data": data,
            "explain": {"why": why, "evidence_event_ids": [], "metrics": {}}}


# ----------------------------------------------------------------- Narrator
NARRATOR_SYSTEM = """You are a live football broadcast writer for a studio graphics + commentary feed.
Write for the viewer described in `viewer`:
- casual: warm, simple, no jargon, max 22 words.
- analyst: precise, tactical, include the key metric(s), max 40 words.
- player_focus: centre the sentence on the focus player, max 30 words.
- metric: lead with the metric the viewer tracks, max 30 words.
If `viewer.clubs` contains the cue's team, sound pleased; if the opponent, sound concerned (stay fair).
HARD RULES: use ONLY facts in `facts`. Do not invent numbers, players, minutes or events.
Use team short names from `teams`. Return JSON: {"text": "<one or two sentences>"}"""


class Narrator:
    def __init__(self, llm: LLM, teams: dict):
        self.llm, self.teams = llm, teams

    def facts(self, cue):
        return {k: cue[k] for k in ("kind", "clock", "team", "player_name", "headline", "data")} | {
            "why": cue["explain"]["why"]}

    def write(self, cue: dict, p: Profile) -> str:
        if self.llm.online:
            viewer = {"mode": p.mode, "clubs": p.clubs, "metric": p.metric}
            return self.llm.json(NARRATOR_SYSTEM, {"viewer": viewer, "teams": self.teams,
                                                   "facts": self.facts(cue)})["text"].strip()
        return self.template(cue, p)

    def template(self, cue: dict, p: Profile) -> str:
        k, d = cue["kind"], cue["data"]
        team = self.teams.get(cue.get("team"), cue.get("team") or "")
        who = cue.get("player_name") or ""
        why = cue["explain"]["why"]
        fav = cue.get("team") in p.clubs
        if p.mode == "analyst" or k in ("player_card", "metric_ticker"):
            if k == "player_card":
                return (f"{who}, {cue['clock']}: {d['passes']} passes ({d['pass_accuracy_pct']}%), "
                        f"{d['shots']} shots, {d['xg']} xG, {d['tackles_won']} tackles won, "
                        f"{d['distance_km']} km, top speed {d['top_speed_kmh']} km/h.")
            if k == "metric_ticker":
                return f"{cue['headline']}: " + ", ".join(f"{self.teams.get(a, a)} {b}" for a, b in d.items()) + "."
            return f"{cue['headline']}. {why}"
        if p.mode == "player_focus":
            if k == "key_pass" and d.get("recipient") and cue.get("player_id") != p.player_id:
                return (f"{d['recipient']} receives a line-breaking pass from {who} "
                        f"({d['distance_m']} m, difficulty {d['difficulty_rating']}/10). {why}")
            return f"{cue['headline']}. {why}"
        lines = {
            "goal": f"GOAL! {who} scores for {team}" + (", against the run of play!" if d.get("against_run_of_play") else "!"),
            "big_chance": f"So close! {who} goes close for {team}.",
            "key_pass": f"What a pass from {who}!",
            "momentum_swing": f"{team} are on top right now.",
            "game_state": ("It's getting scrappy: end-to-end stuff." if d.get("regime") == "chaos"
                           else f"{team} have taken control of the ball."),
            "pressure_surge": f"{team} are hunting the ball high up the pitch.",
            "speed": f"Rocket! {who}'s shot flew at {d.get('shot_speed_kmh')} km/h.",
            "player_tag": cue["headline"] + "!",
            "milestone": cue["headline"] + ".",
        }
        text = lines.get(k, cue["headline"])
        if p.clubs and cue.get("team"):
            if k == "goal":
                text = ("Get in! " if fav else "Oh no. ") + text
            elif k in ("momentum_swing", "pressure_surge") and not fav:
                text = "Careful now: " + text
        return text


# -------------------------------------------------------------- RecapWriter
RECAP_SYSTEM = """You write {label} recaps for a football studio. Audience mode: {mode}.
casual: 60-90 words, story first, simple language. analyst: 90-130 words, tactical, cite metrics.
If the viewer has favourite clubs, frame the story from their side but stay accurate.
Use ONLY the facts provided; never invent numbers, players or events.
Return JSON: {{"title": "...", "text": "..."}}"""


class RecapWriter:
    def __init__(self, llm: LLM, teams: dict):
        self.llm, self.teams = llm, teams

    def facts(self, label, stats, cues):
        keep = [{"clock": c["clock"], "kind": c["kind"], "headline": c["headline"], "why": c["explain"]["why"]}
                for c in cues if c["kind"] in ("goal", "big_chance", "momentum_swing", "pressure_surge", "game_state")]
        goals = [k for k in keep if k["kind"] == "goal"]
        others = [k for k in keep if k["kind"] != "goal"][-12:]
        moments = sorted(goals + others, key=lambda k: int(k["clock"].rstrip("'")))
        return {"label": label, "score": stats["score"], "teams": self.teams, "team_stats": stats["teams"],
                "key_moments": moments}

    def write(self, label, stats, cues, p: Profile) -> dict:
        f = self.facts(label, stats, cues)
        if self.llm.online:
            r = self.llm.json(RECAP_SYSTEM.format(label=label, mode=p.mode),
                              {"viewer": {"mode": p.mode, "clubs": p.clubs}, "facts": f}, max_tokens=500)
            return {"title": r.get("title", label), "text": r["text"], "facts": f}
        return {"title": label, "text": self.template(f, p), "facts": f}

    def template(self, f, p):
        a, b = list(f["teams"])
        sa, sb = f["score"][a], f["score"][b]
        ta, tb = f["team_stats"][a], f["team_stats"][b]
        goals = [m for m in f["key_moments"] if m["kind"] == "goal"]
        swings = [m for m in f["key_moments"] if m["kind"] in ("momentum_swing", "pressure_surge")]
        lead = f"{f['label']}: {self.teams[a]} {sa}-{sb} {self.teams[b]}."
        g = " ".join(f"{m['clock']} {m['headline'].split(': ', 1)[-1].split(' | ')[0]} scored." for m in goals)
        if p.mode == "analyst":
            body = (f" Possession {ta['possession_pct']}% to {tb['possession_pct']}%. xG {ta['xg']} to {tb['xg']} "
                    f"from {ta['shots']} and {tb['shots']} shots. Pass accuracy {ta['pass_accuracy_pct']}% vs "
                    f"{tb['pass_accuracy_pct']}%. Pressures {ta['pressures']} vs {tb['pressures']}.")
            turn = f" Key shift: {swings[-1]['headline']} ({swings[-1]['clock']}). {swings[-1]['why']}" if swings else ""
            return lead + (" " + g if g else "") + body + turn
        better = a if ta["xg"] > tb["xg"] else b
        return lead + (" " + g if g else " No goals yet.") + (
            f" {self.teams[better]} created the better chances." if abs(ta["xg"] - tb["xg"]) > 0.3 else
            " Chances were evenly shared.")


# ---------------------------------------------------------------- Localiser
LOCALISE_SYSTEM = """Translate the football broadcast text into language `lang` (BCP-47) in a natural,
broadcast-ready register. Keep player names, team names and every number exactly as given.
Return JSON: {"text": "..."}"""


class Localiser:
    """Azure AI Translator if configured, else the Foundry model, else pass-through."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.key = os.getenv("AZURE_TRANSLATOR_KEY")
        self.region = os.getenv("AZURE_TRANSLATOR_REGION", "global")
        self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")

    def translate(self, text: str, lang: str) -> tuple[str, str]:
        if lang.split("-")[0] == "en":
            return text, "en"
        if self.key:
            req = urllib.request.Request(
                f"{self.endpoint}/translate?api-version=3.0&from=en&to={lang}",
                data=json.dumps([{"Text": text}]).encode(),
                headers={"Ocp-Apim-Subscription-Key": self.key, "Ocp-Apim-Subscription-Region": self.region,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.load(r)[0]["translations"][0]["text"], lang
        if self.llm.online:
            return self.llm.json(LOCALISE_SYSTEM, {"lang": lang, "text": text})["text"], lang
        return text, "en"  # offline: no translation available
