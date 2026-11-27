# Copyright (c) OpenMMLab. All rights reserved.
"""Review regressions through native output coalescing and the real ASGI/SDK
path."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

pytest.importorskip('torch')
pytest.importorskip('opentelemetry.sdk')


def test_native_coalescing_preserves_first_token_and_request_reset(monkeypatch, trace_sdk):
    from lmdeploy.messages import EngineOutput, ResponseType
    from lmdeploy.pytorch.engine import engine_loop
    from lmdeploy.pytorch.engine.request import Response
    from lmdeploy.pytorch.messages import MessageStatus
    from lmdeploy.serve.observability.request_trace import RequestTrace

    monkeypatch.setattr(engine_loop, 'TRACE_ENABLED', True)
    loop = engine_loop.EngineLoop.__new__(engine_loop.EngineLoop)
    loop.config = NS(num_speculative_tokens=None)
    loop.seq_strategy = NS(update_running=lambda **kw: None)
    loop.scheduler = NS(cache_routed_experts=lambda *a: None)
    response = Response(ResponseType.SUCCESS, 1, asyncio.Event())
    msg = NS(status=MessageStatus.RUNNING, num_token_ids=1, generated_ids=[1],
             session_id=1, resp=response, resp_cache=False, sampling_param=NS(num_logprobs=0),
             cached_tokens=0, engine_events=[], routed_experts=None, return_ce_loss=False, return_logits=False)
    batch = NS(logits=None, all_routed_experts=None, ce_loss=None, logprobs=None,
               stop_pos=None, new_token_timestamp=10.0)
    first = loop._make_infer_outputs(batch, [msg], None, None)[1]
    msg.generated_ids = [1, 2, 3]
    batch.new_token_timestamp = 12.0
    latest = loop._make_infer_outputs(batch, [msg], None, None)[1]
    delivered = []
    loop._log_resps = lambda *a: None
    loop._send_resp = delivered.append
    loop._send_resps([first, latest])
    assert delivered == [latest]
    trace = RequestTrace()
    trace.observe(EngineOutput(ResponseType.SUCCESS, latest.token_ids, req_metrics=latest.req_metrics))
    assert (trace.first_token, trace.last_token) == (10.0, 12.0)
    trace.finish('success')
    # A fresh response is used for each request, including reused engine sessions.
    msg.resp = Response(ResponseType.SUCCESS, 1, asyncio.Event())
    batch.new_token_timestamp = 20.0
    assert loop._make_infer_outputs(batch, [msg], None, None)[1].req_metrics.first_token_timestamp == 20.0


def test_coalesced_batch_without_explicit_first_time_stays_unknown(trace_sdk):
    from lmdeploy.messages import EngineOutput, RequestMetrics, ResponseType
    from lmdeploy.serve.observability.request_trace import RequestTrace

    trace = RequestTrace()
    trace.observe(EngineOutput(ResponseType.SUCCESS, [1, 2, 3], req_metrics=RequestMetrics(token_timestamp=12)))
    assert trace.first_token is None
    assert trace.last_token == 12
    trace.finish('success')


@pytest.mark.parametrize('stream', [False, True])
def test_unsampled_request_skips_prompt_and_output_observation(monkeypatch, trace_sdk, stream):
    from lmdeploy.serve.observability import request_trace as trace

    factory = Mock(side_effect=AssertionError('unsampled prompt state allocated'))
    parser = Mock(side_effect=AssertionError('unsampled SSE parsed'))
    monkeypatch.setattr(trace, 'RequestTrace', factory)
    monkeypatch.setattr(trace, '_decode_json', parser)

    @trace.trace_preprocess
    async def preprocess():
        return NS(trace=None, consumed=False)

    @trace.trace_generate
    async def generate(engine, request):
        for _ in range(128):
            yield 'data: {"choices":[{"index":0,"text":"private"}]}\n\n'

    async def app(scope, receive, send):
        request = await preprocess()
        assert request.trace is None
        headers = [(b'content-type', b'text/event-stream')] if stream else []
        await send(dict(type='http.response.start', status=200, headers=headers))
        async for frame in generate(None, request):
            assert trace.trace_frame(frame) is frame
            await send(dict(type='http.response.body', body=frame.encode(), more_body=True))
        await send(dict(type='http.response.body', body=b'', more_body=False))

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return dict(type='http.request', body=b'')

    asyncio.run(trace.RequestTraceMiddleware(app)(dict(type='http', method='POST', path='/v1/completions',
        headers=[(b'traceparent', b'00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-00'),
                 (b'x-request-id', b'req_unsampled_123456')]), receive, send))
    assert (b'x-request-id', b'req_unsampled_123456') in sent[0]['headers']
    assert trace_sdk.get_finished_spans() == ()
    assert trace.request_outcome_context.get() is None
    factory.assert_not_called()
    parser.assert_not_called()


def test_mount_path_preserves_parent_and_request_id(trace_sdk):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lmdeploy.serve.observability.request_trace import RequestTraceMiddleware

    child = FastAPI()

    @child.post('/v1/completions')
    async def complete():
        return {'ok': True}

    app = FastAPI()
    app.mount('/llm', RequestTraceMiddleware(child))
    with TestClient(app) as client:
        response = client.post('/llm/v1/completions', headers={
            'traceparent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
            'x-request-id': 'req_mounted_123456'})
    assert response.status_code == 200
    assert response.headers['x-request-id'] == 'req_mounted_123456'
    span, = trace_sdk.get_finished_spans()
    assert span.context.trace_id == int('a' * 32, 16)
    assert span.parent.span_id == int('b' * 16, 16)
    assert span.attributes['http.route'] == '/v1/completions'


def test_failed_asgi_send_does_not_record_visible_output(monkeypatch, trace_sdk):
    from lmdeploy.serve.observability import request_trace as trace

    observed = Mock()
    monkeypatch.setattr(trace.OutputObservation, 'observe_metadata', observed)

    async def app(scope, receive, send):
        await send(dict(type='http.response.start', status=200,
                        headers=[(b'content-type', b'text/event-stream')]))
        data = dict(choices=[dict(index=0, text='private')])
        frame = trace.trace_frame(f'data: {json.dumps(data)}\n\n', data)
        await send(dict(type='http.response.body', body=frame.encode(), more_body=True))

    async def receive():
        return dict(type='http.request', body=b'')

    async def send(message):
        if message['type'] == 'http.response.body':
            raise OSError('disconnected')

    with pytest.raises(OSError):
        asyncio.run(trace.RequestTraceMiddleware(app)(dict(type='http', method='POST',
            path='/v1/completions', headers=[]), receive, send))
    observed.assert_not_called()


@pytest.mark.parametrize('role,expected', [('Hybrid', True), ('Prefill', False), ('Decode', False)])
def test_explicit_engine_opt_in_obeys_full_engine_scope(monkeypatch, role, expected):
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.pytorch.disagg.config import EngineRole
    from lmdeploy.serve.core import async_engine

    class StopBeforeModelLoad(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopBeforeModelLoad

    monkeypatch.setattr(async_engine, 'get_chat_template', stop)
    monkeypatch.setattr(async_engine.otel_trace, 'initialize', lambda: None)
    engine = async_engine.AsyncEngine.__new__(async_engine.AsyncEngine)
    with pytest.raises(StopBeforeModelLoad):
        engine.__init__('unused', backend='pytorch', enable_request_trace=True,
                        backend_config=PytorchEngineConfig(role=EngineRole[role]))
    assert engine.request_trace_enabled is expected


@pytest.mark.parametrize('role,expected', [('Hybrid', True), ('Prefill', False), ('Decode', False)])
def test_api_server_only_installs_observer_for_full_engine(monkeypatch, trace_sdk, role, expected):
    from fastapi import APIRouter
    from fastapi.testclient import TestClient

    from lmdeploy import trace
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.pytorch.disagg.config import EngineRole
    from lmdeploy.serve.openai import api_server

    pipeline = Mock(return_value=NS(is_sleeping=False))
    launch = Mock()
    monkeypatch.setattr(trace, 'ENABLED', True)
    monkeypatch.setattr(api_server, 'resolve_default_gen_config', lambda *a: None)
    monkeypatch.setattr(api_server, 'handle_torchrun', lambda: None)
    monkeypatch.setattr(api_server, 'get_task', lambda *a, **kw: (None, pipeline))
    monkeypatch.setattr(api_server, 'set_parsers', lambda *a: None)
    monkeypatch.setattr(api_server, 'create_lifespan_handler', lambda *a, **kw: None)
    monkeypatch.setattr(api_server, 'mount_metrics', lambda *a: None)
    monkeypatch.setattr(api_server, 'APIServer', lambda config, draining: NS(
        started=True, run=lambda: launch(app=config.app)))
    router = APIRouter()

    @router.post('/v1/completions')
    async def complete():
        return {'ok': True}

    monkeypatch.setattr(api_server, 'create_openai_router', lambda *a: router)
    monkeypatch.setattr(api_server, 'create_anthropic_router', lambda *a: APIRouter())
    monkeypatch.setattr(api_server, 'create_responses_router', lambda *a: APIRouter())
    api_server.serve('unused', backend='pytorch',
                     backend_config=PytorchEngineConfig(role=EngineRole[role]))
    assert pipeline.call_args.kwargs['enable_request_trace'] is expected
    app = launch.call_args.kwargs['app']
    assert isinstance(app, api_server._TracingFastAPI) is expected
    with TestClient(app) as client:
        assert client.post('/v1/completions').status_code == 200
    assert bool(trace_sdk.get_finished_spans()) is expected
