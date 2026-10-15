"""Validate explainability cues against simulator ground truth over 20 synthetic matches.

    python -m scripts.evaluate
"""
import json, subprocess, sys
from engine.replay import stream, evaluate
from engine.insights import MatchEngine
from pathlib import Path
P=H=PH=REC=0
for s in range(1, 21):
    d=Path(f"data/eval/m{s}")
    subprocess.run([sys.executable,"-m","simulator.sim","--seed",str(s),"--out",str(d)],check=True,capture_output=True)
    e=MatchEngine()
    for t,k,o in stream(d):
        e.ingest_event(o) if k=="event" else e.ingest_frame(o)
    r=evaluate(e.cues,d/"truth.json"); P+=r["pressure_surge_cues"]; H+=r["inside_true_high_press"]
    ph=[p for p in json.load(open(d/"truth.json"))["tactical_phases"] if p["phase"]=="high_press" and p["end"]-p["start"]>600]
    PH+=len(ph); REC+=sum(1 for p in ph if any(c["kind"]=="pressure_surge" and c["team"]==p["team"] and p["start"]<=c["t_in"]<=p["end"]+120 for c in e.cues))
print(f"20 matches: precision {H}/{P} = {H/P:.0%}, recall {REC}/{PH} = {REC/PH:.0%}")
