"""Agent-layer tests: personalisation differs by viewer, and the Verifier blocks hallucinations."""
import tempfile
import unittest
from pathlib import Path

from agents.agents import Verifier
from agents.profiles import Profile
from agents.run import Orchestrator
from simulator.sim import Sim, write


class FakeHallucinatingLLM:
    """Pretends to be the cloud model but invents a number on every call."""
    online, calls, latency_ms = True, 0, 0.0

    def json(self, system, payload, max_tokens=300):
        self.calls += 1
        return {"text": "An incredible 97 metre solo run!", "title": "x"}


def run_match(profiles, llm=None):
    d = Path(tempfile.mkdtemp())
    sim = Sim(5)
    sim.run()
    write(sim, d)
    orch = Orchestrator(profiles, "offline")
    if llm:
        orch.llm = llm
    orch.run(d)
    return orch, sim


class AgentTest(unittest.TestCase):
    def test_verifier_rules(self):
        v = Verifier({"A-01": "Leo Halden", "B-01": "Kai Okafor"})
        facts = {"player_name": "Leo Halden", "data": {"xg": 0.23, "distance_m": 18.4}}
        self.assertEqual(v.check("Leo Halden's chance was worth 23% from 18.4 m", facts), [])
        self.assertTrue(v.check("Leo Halden scored from 30 m", facts))           # invented number
        self.assertTrue(v.check("Kai Okafor's chance from 18.4 m", facts))       # wrong player
        speed = {"player_name": "Leo Halden", "data": {"shot_speed_kmh": 97.4}}
        self.assertEqual(v.check("Leo Halden hit it at 97 km/h", speed), [])
        self.assertTrue(v.check("Leo Halden's 97 metre run", speed))            # right number, wrong unit

    def test_modes_produce_different_feeds(self):
        orch, sim = run_match({
            "casual": Profile(mode="casual"), "analyst": Profile(mode="analyst"),
            "metric": Profile(mode="metric", metric="xg"),
        })
        f = orch.feeds
        self.assertLess(len(f["casual"]), len(f["analyst"]))
        self.assertTrue(any(i["kind"] == "metric_ticker" for i in f["metric"]))
        for feed in f.values():
            self.assertTrue(all(i["verified"] for i in feed))
        self.assertEqual(len(orch.recaps["casual"]), 2)  # half-time + full-time

    def test_hallucinations_are_blocked_and_replaced(self):
        orch, _ = run_match({"casual": Profile(mode="casual")}, FakeHallucinatingLLM())
        feed = orch.feeds["casual"]
        self.assertTrue(feed)
        self.assertTrue(all(i["fallback"] for i in feed))
        self.assertTrue(all("97" not in i["text"] for i in feed))
        self.assertGreater(orch.verifier.blocked, 0)


if __name__ == "__main__":
    unittest.main()
