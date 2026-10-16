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
    online, calls, latency_ms, agent_calls, label = True, 0, 0.0, 0, 'fake'

    def json(self, system, payload, max_tokens=300, agent=None):
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


class FakeSlowLLM:
    """Online fake: 0.2 s per call, echoes the headline (grounded), counts concurrency."""
    online, calls, latency_ms, backend, agent_calls, label = True, 0, 0.0, "fake", 0, "fake"

    def __init__(self):
        import threading
        self.lock, self.active, self.peak = threading.Lock(), 0, 0

    def json(self, system, payload, max_tokens=300, agent=None):
        import time
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.2)
        with self.lock:
            self.active -= 1
            self.calls += 1
        f = payload.get("facts", {})
        return {"text": f.get("headline") or "Recap.", "title": "t"}


class ImprovementsTest(unittest.TestCase):
    def test_wrong_script_is_blocked(self):
        from agents.agents import foreign_scripts
        v = Verifier({"A-01": "Leo Halden"})
        facts = {"player_name": "Leo Halden"}
        self.assertTrue(v.check("Leo Halden da un pase शानदार", facts, "es"))
        self.assertEqual(v.check("¡Qué pase de Leo Halden!", facts, "es"), [])
        self.assertEqual(foreign_scripts("लियो Halden का शानदार पास", "hi"), set())

    def test_perspective(self):
        from agents.agents import Narrator
        fan = Profile(mode="casual", clubs=["NGC"])
        self.assertEqual(Narrator.perspective({"team": "NGC"}, fan), "good_for_fan")
        self.assertEqual(Narrator.perspective({"team": "RVA"}, fan), "bad_for_fan")
        self.assertEqual(Narrator.perspective({"team": "RVA"}, Profile(mode="casual")), "neutral")

    def test_parallel_calls_keep_order(self):
        import time
        llm = FakeSlowLLM()
        t0 = time.time()
        orch, _ = run_match({"fan": Profile(mode="casual", clubs=["NGC"])}, llm)
        elapsed = time.time() - t0
        feed = orch.feeds["fan"]
        self.assertGreater(llm.peak, 1)                       # calls really overlapped
        self.assertLess(elapsed, llm.calls * 0.2 * 0.6 + 3)   # much faster than sequential
        self.assertEqual([i["t"] for i in feed], sorted(i["t"] for i in feed))
        self.assertTrue(all(i["verified"] for i in feed))
        self.assertTrue(all(r.get("text") for r in orch.recaps["fan"]))

    def test_favourite_fan_feed_is_paced(self):
        orch, _ = run_match({"fan": Profile(mode="casual", clubs=["NGC"]), "plain": Profile(mode="casual")})
        self.assertLess(len(orch.feeds["fan"]), len(orch.feeds["plain"]) + 12)
