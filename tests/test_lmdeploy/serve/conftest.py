# Copyright (c) OpenMMLab. All rights reserved.
"""Real SDK fixture shared by serving trace regressions."""
import pytest


@pytest.fixture
def trace_sdk(monkeypatch):
    pytest.importorskip('opentelemetry.sdk')
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lmdeploy.serve.observability import otel_trace, request_trace

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, '_tracer', provider.get_tracer('review'))
    monkeypatch.setattr(request_trace, 'ENABLED', True)
    monkeypatch.setattr(otel_trace, 'initialize', lambda: None)
    yield exporter
    provider.shutdown()
