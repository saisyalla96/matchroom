"""Smoke + integrity tests.  Run:  python -m pytest -q   (or python -m unittest)"""
import json
import tempfile
import unittest
from pathlib import Path

from engine.insights import MatchEngine, difficulty_rating
from engine.replay import stream
from simulator.sim import Sim, write


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        sim = Sim(3)
        sim.run()
        write(sim, cls.dir)
        cls.sim = sim
        cls.eng = MatchEngine()
        for _, kind, obj in stream(cls.dir):
            cls.eng.ingest_event(obj) if kind == "event" else cls.eng.ingest_frame(obj)
        cls.event_ids = {e["event_id"] for e in sim.events}

    def test_realistic_ranges(self):
        s = self.eng.stats()
        for t in s["teams"].values():
            self.assertTrue(65 <= t["pass_accuracy_pct"] <= 92)
            self.assertTrue(0 <= t["shots"] <= 30)
        outfield = [p["distance_km"] for p in s["players"].values()]
        self.assertTrue(8 <= sorted(outfield)[len(outfield) // 2] <= 13)

    def test_score_matches_simulator(self):
        self.assertEqual(list(self.eng.stats()["score"].values()), self.sim.score)

    def test_every_cue_is_grounded(self):
        """Explainability contract: evidence must reference real events."""
        for c in self.eng.cues:
            self.assertTrue(c["explain"]["why"], c)
            for eid in c["explain"]["evidence_event_ids"]:
                self.assertIn(eid, self.event_ids)
            self.assertLess(c["t_in"], c["t_out"])

    def test_cues_are_time_ordered_and_json(self):
        ts = [c["t_in"] for c in self.eng.cues]
        self.assertEqual(ts, sorted(ts))
        json.dumps(self.eng.cues)

    def test_difficulty_rating_bounds(self):
        self.assertEqual(difficulty_rating(0), 1)
        self.assertEqual(difficulty_rating(1), 10)


if __name__ == "__main__":
    unittest.main()
