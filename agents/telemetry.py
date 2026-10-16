"""
OpenTelemetry tracing to Azure Application Insights (and so to Foundry's Traces view).

Enabled when APPLICATIONINSIGHTS_CONNECTION_STRING is set (the Azure deployment sets it), or when the
Foundry project has an Application Insights resource connected. Otherwise every span is a no-op, so
the code runs the same with or without Azure.

Spans follow the OpenTelemetry GenAI conventions (gen_ai.*) so Foundry can show agent runs, plus
matchroom.* attributes that make the Verifier's decisions visible in each trace.
"""
from __future__ import annotations

import contextlib
import os

_enabled = False

try:
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("matchroom")
except Exception:  # opentelemetry not installed: offline/dev mode
    _trace = None
    tracer = None


def setup(project=None) -> bool:
    """Configure export to Application Insights once. Returns True when tracing is on."""
    global _enabled, tracer
    if _enabled:
        return True
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn and project is not None:
        try:  # the App Insights resource connected to the Foundry project
            conn = project.telemetry.get_application_insights_connection_string()
        except Exception:
            conn = None
    if not conn:
        return False
    try:
        os.environ.setdefault("OTEL_SERVICE_NAME", "matchroom")
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=conn)
        try:  # Foundry SDK spans for agent calls (prompts and outputs recorded unless disabled)
            from azure.ai.projects.telemetry import AIProjectInstrumentor

            AIProjectInstrumentor().instrument(
                enable_content_recording=os.getenv("MATCHROOM_TRACE_CONTENT", "true").lower() == "true")
        except Exception:
            pass
        from opentelemetry import trace

        tracer = trace.get_tracer("matchroom")
        _enabled = True
        print("[tracing] sending traces to Application Insights")
    except Exception as ex:
        print(f"[tracing] disabled: {ex.__class__.__name__}: {ex}")
    return _enabled


@contextlib.contextmanager
def span(name: str, **attrs):
    """Start a span if tracing is available; attributes with None values are skipped."""
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))
        yield s


def set_attrs(s, **attrs):
    if s is None:
        return
    for k, v in attrs.items():
        if v is not None:
            s.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))
