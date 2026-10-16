"""Foundry Agent Service path, tested against an in-memory fake project (no Azure needed)."""
import json
import types
import unittest

try:
    import azure.ai.projects  # noqa: F401
    import httpx
    import openai
    HAVE_SDK = True
except ImportError:
    HAVE_SDK = False


class FakeAgents:
    def __init__(self):
        self.store = {}  # name -> list of versions

    def list_versions(self, name, limit=None, order=None):
        if name not in self.store:
            raise KeyError("not found")
        return iter(reversed(self.store[name]))

    def create_version(self, agent_name, definition, description=None, metadata=None):
        vs = self.store.setdefault(agent_name, [])
        v = types.SimpleNamespace(version=str(len(vs) + 1), metadata=metadata, definition=definition)
        vs.append(v)
        return v


class FakeProject:
    def __init__(self, fail_agent=None):
        self.agents = FakeAgents()
        self.calls = []
        self.fail_agent = fail_agent

    def get_openai_client(self, agent_name=None):
        proj = self

        class Responses:
            def create(self, input):
                proj.calls.append(agent_name)
                if agent_name == proj.fail_agent:
                    req = httpx.Request("POST", "https://x")
                    raise openai.NotFoundError("agent missing", response=httpx.Response(404, request=req), body=None)
                assert "json" in input.lower(), "Foundry rejects json_object output unless the input mentions json"
                payload = json.loads(input[input.index("{"):])
                text = (payload.get("facts", {}).get("headline") or payload.get("text") or "ok")
                return types.SimpleNamespace(output_text=json.dumps({"text": text, "title": "t"}),
                                             usage=types.SimpleNamespace(input_tokens=120, output_tokens=30))
        return types.SimpleNamespace(responses=Responses())


def agents_llm(project, with_direct=False):
    from agents.llm import LLM

    llm = LLM("offline")
    llm.backend, llm.project, llm.deployment = "agents", project, "gpt-5.4-mini"
    from agents.foundry import ensure_agents
    llm.agent_versions = ensure_agents(project, llm.deployment, verbose=False)
    llm.agents_ok = True
    if with_direct:
        class Chat:
            def create(self, **k):
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"text": "direct"}'))],
                    usage=None)
        llm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Chat()))
    return llm


@unittest.skipUnless(HAVE_SDK, "azure-ai-projects not installed")
class FoundryAgentsTest(unittest.TestCase):
    def test_agents_are_versioned_only_when_changed(self):
        from agents.foundry import ensure_agents
        proj = FakeProject()
        v1 = ensure_agents(proj, "gpt-5.4-mini", verbose=False)
        self.assertEqual(set(v1), {"matchroom-narrator", "matchroom-recap-writer", "matchroom-localiser"})
        self.assertEqual(ensure_agents(proj, "gpt-5.4-mini", verbose=False), v1)       # no change: same version
        v3 = ensure_agents(proj, "gpt-5.4-mini", effort="low", verbose=False)           # changed: new version
        self.assertTrue(all(int(v3[k]) == int(v1[k]) + 1 for k in v1))
        d = proj.agents.store["matchroom-narrator"][0].definition
        self.assertIn("JSON", d.instructions)

    def test_full_match_through_agents(self):
        from agents.profiles import Profile
        from tests.test_agents import run_match
        proj = FakeProject()
        orch, _ = run_match({"fan": Profile(mode="casual", language="es", clubs=["NGC"])}, agents_llm(proj))
        self.assertTrue(orch.feeds["fan"])
        self.assertTrue(all(i["verified"] for i in orch.feeds["fan"]))
        self.assertIn("matchroom-narrator", proj.calls)
        self.assertIn("matchroom-recap-writer", proj.calls)
        self.assertEqual(orch.llm.agent_calls, orch.llm.calls)

    def test_rejected_reasoning_effort_reversions_agents(self):
        proj = FakeProject()
        llm = agents_llm(proj)
        real = proj.get_openai_client
        state = {"n": 0}

        def picky(agent_name=None):
            c = real(agent_name)
            create = c.responses.create

            def maybe_fail(input):
                latest = proj.agents.store[agent_name][-1].definition
                eff = getattr(getattr(latest, "reasoning", None), "effort", None)
                if eff not in (None, "none"):
                    state["n"] += 1
                    req = httpx.Request("POST", "https://x")
                    raise openai.BadRequestError("Unsupported value: 'minimal' is not supported. Supported values are: 'none', 'low' (param reasoning.effort)",
                                                 response=httpx.Response(400, request=req), body=None)
                return create(input)
            c.responses.create = maybe_fail
            return c
        proj.get_openai_client = picky
        from agents.foundry import ensure_agents
        llm.agent_versions = ensure_agents(proj, llm.deployment, effort="minimal", verbose=False)
        out = llm.json("", {"facts": {"headline": "hi"}}, agent="matchroom-narrator")
        self.assertEqual(out["text"], "hi")
        self.assertEqual(state["n"], 1)
        self.assertEqual(proj.agents.store["matchroom-narrator"][-1].definition.reasoning.effort, "none")

    def test_missing_agent_falls_back_to_direct_model(self):
        llm = agents_llm(FakeProject(fail_agent="matchroom-narrator"), with_direct=True)
        out = llm.json("system", {"facts": {"headline": "x"}}, agent="matchroom-narrator")
        self.assertEqual(out["text"], "direct")
        self.assertFalse(llm.agents_ok)

    def test_spans_carry_genai_and_verifier_attributes(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        import agents.telemetry as T
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        old = T.tracer
        T.tracer = provider.get_tracer("matchroom-test")
        try:
            from agents.profiles import Profile
            from tests.test_agents import run_match
            run_match({"fan": Profile(mode="casual")}, agents_llm(FakeProject()))
        finally:
            T.tracer = old
        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.name == "invoke_agent matchroom-narrator"]
        utter = [s for s in spans if s.name == "matchroom.utterance"]
        self.assertTrue(agent_spans and utter)
        a = agent_spans[0].attributes
        self.assertEqual(a["gen_ai.operation.name"], "invoke_agent")
        self.assertEqual(a["gen_ai.usage.input_tokens"], 120)
        self.assertIn("matchroom.verified", utter[0].attributes)
        utter_ids = {u.context.span_id for u in utter}
        self.assertTrue(all(s.parent is not None and s.parent.span_id in utter_ids for s in agent_spans))


if __name__ == "__main__":
    unittest.main()
