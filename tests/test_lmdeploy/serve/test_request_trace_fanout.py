# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise real fan-out, ASGI observation and SDK spans with a fake engine."""
import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')
pytest.importorskip('opentelemetry.sdk')


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('include_usage', [False, True])
@pytest.mark.parametrize('count', [2, 3])
def test_concurrent_choices_keep_output_usage_and_adapter_identity(monkeypatch, stream, include_usage, count):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from httpx import ASGITransport, AsyncClient
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lmdeploy.serve.observability import otel_trace, request_trace
    from lmdeploy.serve.openai.chat_completions.fanout import fanout_chat_completions
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, '_tracer', provider.get_tracer('fanout-test'))
    monkeypatch.setattr(request_trace, 'ENABLED', True)
    monkeypatch.setattr(otel_trace, 'initialize', lambda: None)
    if stream:
        def reject_reparse(*args):
            raise AssertionError('The final fan-out frame must carry output metadata')
        monkeypatch.setattr(request_trace, '_decode_json', reject_reparse)

    class Engine:
        model_name = 'base-model'
        backend = 'pytorch'

        @request_trace.trace_preprocess
        async def preprocess(self, *, adapter_name):
            return SimpleNamespace(input_token_len=10, adapter_name=adapter_name)

        @request_trace.trace_generate
        async def generate(self, prepared):
            yield SimpleNamespace(finish_reason=None)
            await asyncio.sleep(0)
            yield SimpleNamespace(finish_reason='stop')

    engine = Engine()

    async def child(request, raw_request):
        index = request.seed - 100
        # Prepare choice 0 last: insertion order must not decide attribution.
        if index == 0:
            await asyncio.sleep(.01)
        prepared = await engine.preprocess(adapter_name=request.model,
                                           trace_choice_index=raw_request._trace_choice_index)
        prepared.trace.prompt_tokens = 10
        prepared.trace.cached_tokens = 4 if index == 0 else 0
        prepared.trace.generated_tokens = prepared.trace.output_tokens = index + 2
        prepared.trace.terminal = True
        usage = dict(prompt_tokens=10, completion_tokens=index + 2,
                     prompt_tokens_details=dict(cached_tokens=prepared.trace.cached_tokens))

        async def body():
            async for _ in engine.generate(prepared):
                key = 'reasoning_content' if index == 0 else 'content'
                payload = dict(choices=[dict(index=0, delta={key: 'private-output'})])
                yield f'data: {json.dumps(payload)}\n\n'
            yield f'data: {json.dumps(dict(choices=[], usage=usage))}\n\n'
            yield 'data: [DONE]\n\n'

        if stream:
            return StreamingResponse(body(), media_type='text/event-stream')
        async for _ in engine.generate(prepared):
            pass
        return dict(model=request.model, choices=[dict(index=0, message=dict(content='private-output'))], usage=usage)

    app = FastAPI()

    @app.post('/v1/chat/completions')
    async def endpoint(raw_request: Request):
        payload = await raw_request.json()
        request = ChatCompletionRequest(**payload)
        return await fanout_chat_completions(child, request, raw_request, payload)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=request_trace.RequestTraceMiddleware(app)),
                               base_url='http://test', trust_env=False) as client:
            return await client.post('/v1/chat/completions', json=dict(
                model='finance-lora', messages=[dict(role='user', content='private-input')],
                n=count, seed=100, stream=stream, stream_options=dict(include_usage=include_usage)))

    try:
        response = asyncio.run(run())
        assert response.status_code == 200
        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name == 'llm_request')
        prompts = {s.attributes['lmdeploy.choice_index']: s for s in spans if s.name == 'inference'}
        assert set(prompts) == set(range(count))
        assert root.attributes['gen_ai.usage.input_tokens'] == 10
        assert root.attributes['gen_ai.usage.cached_tokens'] == 4
        assert root.attributes['gen_ai.usage.cache_read.input_tokens'] == 4
        assert root.attributes['gen_ai.usage.output_tokens'] == sum(range(2, count + 2))
        assert root.attributes['lmdeploy.token_counts_complete']
        for span in [root, *prompts.values()]:
            assert span.attributes['gen_ai.request.model'] == 'finance-lora'
            assert span.attributes['gen_ai.response.model'] == 'finance-lora'
        for index, span in prompts.items():
            assert span.attributes['lmdeploy.prompt_index'] == 0
            assert span.attributes['lmdeploy.base_model'] == 'base-model'
            assert span.attributes['lmdeploy.finish_reason'] == 'stop'
            assert 'gen_ai.response.finish_reasons' not in span.attributes
            assert span.parent.span_id == root.context.span_id
            if stream:
                visible = 'lmdeploy.reasoning_first_seconds' if index == 0 else 'lmdeploy.content_first_seconds'
                absent = 'lmdeploy.content_first_seconds' if index == 0 else 'lmdeploy.reasoning_first_seconds'
                assert visible in span.attributes
                assert absent not in span.attributes
        if stream:
            assert 'lmdeploy.reasoning_first_seconds' in root.attributes
            assert 'lmdeploy.content_first_seconds' in root.attributes
        else:
            assert response.json()['usage']['prompt_tokens'] == root.attributes['gen_ai.usage.input_tokens']
            assert response.json()['usage']['prompt_tokens_details']['cached_tokens'] == 4
        if stream and include_usage:
            usage = [json.loads(line[6:])['usage'] for line in response.text.splitlines()
                     if line.startswith('data: {') and json.loads(line[6:]).get('usage')][-1]
            assert usage['prompt_tokens'] == 10
            assert usage['completion_tokens'] == root.attributes['gen_ai.usage.output_tokens']
        assert 'private' not in str([dict(s.attributes) for s in spans])
        assert request_trace.request_outcome_context.get() is None
    finally:
        provider.shutdown()


@pytest.mark.parametrize('cached', [0, 6])
def test_protocol_usage_conventions_match_normalized_trace_counts(monkeypatch, cached):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lmdeploy.messages import EngineOutput, ResponseType
    from lmdeploy.serve.anthropic.protocol import MessageUsage
    from lmdeploy.serve.core.async_engine import GenOut
    from lmdeploy.serve.observability import otel_trace, request_trace
    from lmdeploy.serve.openai.protocol import UsageInfo

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, '_tracer', provider.get_tracer('usage-conventions'))
    monkeypatch.setattr(request_trace, 'ENABLED', True)
    trace = request_trace.RequestTrace()
    prepared = SimpleNamespace(trace=trace, consumed=False)

    @request_trace.trace_generate
    async def generate(engine, request):
        # Backend has one extra stop token; API adapters consume the filtered count.
        request.trace.observe(EngineOutput(ResponseType.FINISH, [1, 2, 3, 4, 5]))
        yield GenOut('', 20, 4, 'stop', cached_tokens=cached)

    async def run():
        return [r async for r in generate(None, prepared)]

    try:
        result = asyncio.run(run())[-1]
        openai = UsageInfo.build(result.input_token_len, result.generate_token_len, result.cached_tokens)
        anthropic = MessageUsage.build(result.input_token_len, result.generate_token_len, result.cached_tokens)
        span = exporter.get_finished_spans()[0]
        assert span.attributes['gen_ai.usage.input_tokens'] == openai.prompt_tokens
        assert span.attributes['gen_ai.usage.input_tokens'] == (
            anthropic.input_tokens + anthropic.cache_read_input_tokens)
        assert span.attributes['gen_ai.usage.output_tokens'] == openai.completion_tokens == anthropic.output_tokens
        assert span.attributes['gen_ai.usage.cached_tokens'] == openai.prompt_tokens_details.cached_tokens
        assert span.attributes['gen_ai.usage.cached_tokens'] == anthropic.cache_read_input_tokens
        assert span.attributes['gen_ai.usage.cache_read.input_tokens'] == anthropic.cache_read_input_tokens
        assert span.attributes['lmdeploy.finish_reason'] == result.finish_reason
        assert all(isinstance(value, (str, bool, int, float)) for value in span.attributes.values())
    finally:
        provider.shutdown()
