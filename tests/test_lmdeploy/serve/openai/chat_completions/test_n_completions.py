# Copyright (c) OpenMMLab. All rights reserved.
"""Regression tests for multiple chat completion choices."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from lmdeploy.serve.openai.chat_completions.fanout import _batch_stream_payloads
from lmdeploy.serve.openai.protocol import ChatCompletionRequest


class _PreprocessingEngine:
    """Provide the preprocessing stage expected by the serving endpoint."""

    async def preprocess(self, prompt, session, **kwargs):
        self.gen_configs.append(kwargs.get('gen_config'))
        return SimpleNamespace(prompt=prompt, session=session)


def _request(**kwargs):
    return ChatCompletionRequest(
        model='fake-model',
        messages=[{
            'role': 'user',
            'content': 'hi'
        }],
        **kwargs,
    )


def _sse_payloads(text):
    payloads = []
    for line in text.splitlines():
        if line.startswith('data: '):
            data = line.removeprefix('data: ')
            if data != '[DONE]':
                payloads.append(json.loads(data))
    return payloads


async def _collect_stream(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return ''.join(chunks)


def _stream_payload(index, content, **metadata):
    return {
        'id': 'chatcmpl-test',
        'object': 'chat.completion.chunk',
        'created': 1,
        'model': 'fake-model',
        'choices': [{
            'index': index,
            'delta': {
                'content': content
            }
        }],
        **metadata,
    }


def test_ready_stream_chunks_are_batched_by_choice():
    payloads = [
        _stream_payload(0, '0-a'),
        _stream_payload(0, '0-b'),
        _stream_payload(1, '1-a'),
    ]

    batches = _batch_stream_payloads(payloads)

    assert [[choice['index'] for choice in batch['choices']]
            for batch in batches] == [[0, 1], [0]]
    assert [choice['delta']['content']
            for batch in batches
            for choice in batch['choices'] if choice['index'] == 0
            ] == ['0-a', '0-b']


def test_handler_n3_nonstream_collates_single_choice_path(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    response = asyncio.run(endpoint(_request(n=3, seed=42), fake_raw_request))

    assert response['object'] == 'chat.completion'
    assert [choice['index'] for choice in response['choices']] == [0, 1, 2]
    assert {choice['message']['content']
            for choice in response['choices']
            } == {'choice-1', 'choice-2', 'choice-3'}
    assert response['usage']['prompt_tokens'] == 4
    assert response['usage']['completion_tokens'] == 6
    assert [config.random_seed
            for config in context.async_engine.gen_configs] == [42, 43, 44]
    assert context.async_engine.call_count == 3
    assert len(context.session_manager.removed) == 3
    assert context.session_manager.sessions == {}


def test_handler_n3_stream_interleaves_indices_and_aggregates_usage(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    request = _request(
        n=3,
        stream=True,
        stream_options={'include_usage': True},
    )
    response = asyncio.run(endpoint(request, fake_raw_request))
    text = asyncio.run(_collect_stream(response))
    payloads = _sse_payloads(text)

    indices = {
        choice['index']
        for payload in payloads
        for choice in payload.get('choices', [])
    }
    assert indices == {0, 1, 2}
    usage_chunks = [
        payload for payload in payloads if payload.get('usage') is not None
    ]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]['choices'] == []
    assert usage_chunks[0]['usage']['prompt_tokens'] == 4
    assert usage_chunks[0]['usage']['completion_tokens'] == 6
    assert text.rstrip().endswith('data: [DONE]')
    assert len(context.session_manager.removed) == 3
    assert context.session_manager.sessions == {}


@pytest.mark.parametrize('n', [None, 1])
def test_handler_single_choice_keeps_fast_path(n, chat_endpoint,
                                               fake_raw_request):
    endpoint, context = chat_endpoint
    response = asyncio.run(endpoint(_request(n=n), fake_raw_request))
    assert len(response['choices']) == 1
    assert context.async_engine.call_count == 1


def test_handler_unseeded_choices_leave_seed_resolution_to_engine(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    asyncio.run(endpoint(_request(n=3), fake_raw_request))
    assert [config.random_seed for config in context.async_engine.gen_configs
            ] == [None, None, None]


def test_handler_negative_seed_is_mapped_to_engine_seed_domain(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    asyncio.run(endpoint(_request(n=2, seed=-1), fake_raw_request))
    assert [config.random_seed
            for config in context.async_engine.gen_configs] == [(1 << 64) - 1,
                                                                0]


@pytest.mark.parametrize('n, expected', [
    (0, 'positive int'),
    (129, 'maximum supported'),
])
def test_validation_rejects_invalid_n(n, expected, chat_endpoint,
                                      fake_raw_request):
    endpoint, context = chat_endpoint
    response = asyncio.run(endpoint(_request(n=n), fake_raw_request))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert expected in response.body.decode()
    assert context.async_engine.call_count == 0


def test_handler_rejects_explicit_session_id_for_multiple_choices(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    response = asyncio.run(
        endpoint(_request(n=2, session_id=777), fake_raw_request))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert 'explicit session_id' in response.body.decode()
    assert context.async_engine.call_count == 0
    assert context.session_manager.sessions == {}


def test_handler_rejects_cache_migration_for_multiple_choices(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    fake_raw_request._payload = {'with_cache': True}
    response = asyncio.run(endpoint(_request(n=2), fake_raw_request))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert 'cache migration' in response.body.decode()
    assert context.async_engine.call_count == 0


def test_distserve_proxy_rejects_multiple_choices(monkeypatch):
    from lmdeploy.pytorch.disagg.config import ServingStrategy
    from lmdeploy.serve.proxy import proxy

    async def model_exists(model):
        return None

    monkeypatch.setattr(proxy.node_manager, 'check_request_model',
                        model_exists)
    monkeypatch.setattr(proxy.node_manager, 'serving_strategy',
                        ServingStrategy.DistServe)
    response = asyncio.run(proxy.chat_completions_v1(_request(n=2)))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert 'DistServe' in response.body.decode()


def test_prompt_cache_usage_is_counted_once(chat_endpoint, fake_raw_request):

    class CachedEngine(_PreprocessingEngine):
        model_name = 'fake-model'
        backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)

        def __init__(self, original_engine):
            self.session_mgr = original_engine.session_mgr
            self.tokenizer = original_engine.tokenizer
            self.call_count = 0
            self.gen_configs = []

        def generate(self, preprocessed, **kwargs):
            self.call_count += 1
            index = self.call_count

            async def generate():
                yield SimpleNamespace(
                    response=f'choice-{index}',
                    token_ids=[index],
                    input_token_len=4,
                    generate_token_len=1,
                    finish_reason='stop',
                    logprobs=None,
                    cached_tokens=index,
                    routed_experts=None,
                    cache_block_ids=None,
                )

            return generate()

    endpoint, context = chat_endpoint
    context.async_engine = CachedEngine(context.async_engine)
    response = asyncio.run(endpoint(_request(n=2), fake_raw_request))
    assert response['usage']['prompt_tokens'] == 4
    assert response['usage']['completion_tokens'] == 2
    assert response['usage']['prompt_tokens_details']['cached_tokens'] == 1


def test_streaming_multiple_choices_preserves_each_inner_parser(
        chat_endpoint, fake_raw_request):

    class MultiChunkEngine(_PreprocessingEngine):
        model_name = 'fake-model'
        backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)

        def __init__(self, original_engine):
            self.session_mgr = original_engine.session_mgr
            self.tokenizer = original_engine.tokenizer
            self.call_count = 0
            self.gen_configs = []

        def generate(self, preprocessed, **kwargs):
            self.call_count += 1
            index = self.call_count

            async def generate():
                for suffix in ('a', 'b', 'c'):
                    yield SimpleNamespace(
                        response=f'{index}-{suffix}',
                        token_ids=[index],
                        input_token_len=3,
                        generate_token_len=1,
                        finish_reason=None,
                        logprobs=None,
                        cached_tokens=0,
                        routed_experts=None,
                        cache_block_ids=None,
                    )
                yield SimpleNamespace(
                    response='',
                    token_ids=[],
                    input_token_len=3,
                    generate_token_len=3,
                    finish_reason='stop',
                    logprobs=None,
                    cached_tokens=0,
                    routed_experts=None,
                    cache_block_ids=None,
                )

            return generate()

    endpoint, context = chat_endpoint
    context.async_engine = MultiChunkEngine(context.async_engine)
    response = asyncio.run(
        endpoint(_request(n=2, stream=True), fake_raw_request))
    payloads = _sse_payloads(asyncio.run(_collect_stream(response)))
    content = {0: '', 1: ''}
    for payload in payloads:
        for choice in payload.get('choices', []):
            content[choice['index']] += choice['delta'].get('content') or ''
    assert content == {0: '1-a1-b1-c', 1: '2-a2-b2-c'}


def test_fanout_error_cancels_siblings_and_cleans_sessions(
        chat_endpoint, fake_raw_request):

    class FailingEngine(_PreprocessingEngine):
        model_name = 'fake-model'
        backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)

        def __init__(self, original_engine):
            self.session_mgr = original_engine.session_mgr
            self.tokenizer = original_engine.tokenizer
            self.call_count = 0
            self.gen_configs = []
            self.sibling_started = asyncio.Event()
            self.sibling_closed = False

        def generate(self, preprocessed, **kwargs):
            self.call_count += 1
            index = self.call_count

            async def fail():
                await self.sibling_started.wait()
                raise RuntimeError('choice failed')
                yield  # noqa: unreachable

            async def wait_forever():
                self.sibling_started.set()
                try:
                    await asyncio.Event().wait()
                    yield  # noqa: unreachable
                finally:
                    self.sibling_closed = True

            return fail() if index == 1 else wait_forever()

    endpoint, context = chat_endpoint
    engine = FailingEngine(context.async_engine)
    context.async_engine = engine

    with pytest.raises(RuntimeError, match='choice failed'):
        asyncio.run(endpoint(_request(n=2), fake_raw_request))
    assert engine.sibling_closed
    assert context.session_manager.sessions == {}


def test_early_stream_close_cleans_all_choice_sessions(chat_endpoint,
                                                       fake_raw_request):
    endpoint, context = chat_endpoint
    response = asyncio.run(
        endpoint(_request(n=2, stream=True), fake_raw_request))

    async def consume_one_chunk():
        iterator = response.body_iterator
        await anext(iterator)
        await iterator.aclose()

    asyncio.run(consume_one_chunk())
    assert context.session_manager.sessions == {}


def test_asgi_disconnect_before_stream_start_cleans_all_choice_sessions(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint

    async def disconnect_before_stream_start():
        response = await endpoint(
            _request(n=2, stream=True), fake_raw_request)

        async def receive():
            return {'type': 'http.disconnect'}

        async def send(message):
            if message['type'] == 'http.response.start':
                await asyncio.Event().wait()

        scope = {
            'type': 'http',
            'asgi': {
                'version': '3.0',
                'spec_version': '2.3'
            },
        }
        await response(scope, receive, send)

    asyncio.run(asyncio.wait_for(disconnect_before_stream_start(), 1))
    assert context.session_manager.sessions == {}


def test_streaming_cancelled_choice_fails_without_hanging(
        chat_endpoint, fake_raw_request):

    class CancelledChoiceEngine(_PreprocessingEngine):
        model_name = 'fake-model'
        backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)

        def __init__(self, original_engine):
            self.session_mgr = original_engine.session_mgr
            self.tokenizer = original_engine.tokenizer
            self.call_count = 0
            self.gen_configs = []
            self.sibling_started = asyncio.Event()
            self.sibling_closed = False

        def generate(self, preprocessed, **kwargs):
            self.call_count += 1
            index = self.call_count

            async def cancel():
                await self.sibling_started.wait()
                raise asyncio.CancelledError
                yield  # noqa: unreachable

            async def wait_forever():
                self.sibling_started.set()
                try:
                    await asyncio.Event().wait()
                    yield  # noqa: unreachable
                finally:
                    self.sibling_closed = True

            return cancel() if index == 1 else wait_forever()

    endpoint, context = chat_endpoint
    engine = CancelledChoiceEngine(context.async_engine)
    context.async_engine = engine

    async def collect():
        response = await endpoint(
            _request(n=2, stream=True), fake_raw_request)
        await asyncio.wait_for(_collect_stream(response), 1)

    with pytest.raises(RuntimeError, match='choice 0 was cancelled'):
        asyncio.run(collect())
    assert engine.sibling_closed
    assert context.session_manager.sessions == {}


def test_streaming_generation_failure_cleans_all_sessions(
        chat_endpoint, fake_raw_request):

    class FailingEngine(_PreprocessingEngine):
        model_name = 'fake-model'
        backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)

        def __init__(self, original_engine):
            self.session_mgr = original_engine.session_mgr
            self.tokenizer = original_engine.tokenizer
            self.call_count = 0
            self.gen_configs = []
            self.sibling_started = asyncio.Event()
            self.sibling_closed = False

        def generate(self, preprocessed, **kwargs):
            self.call_count += 1
            index = self.call_count

            async def wait_forever():
                self.sibling_started.set()
                try:
                    await asyncio.Event().wait()
                    yield  # noqa: unreachable
                finally:
                    self.sibling_closed = True

            async def fail():
                await self.sibling_started.wait()
                raise RuntimeError('choice generation failed')
                yield  # noqa: unreachable

            return wait_forever() if index == 1 else fail()

    endpoint, context = chat_endpoint
    engine = FailingEngine(context.async_engine)
    context.async_engine = engine

    async def collect():
        response = await endpoint(
            _request(n=2, stream=True), fake_raw_request)
        await asyncio.wait_for(_collect_stream(response), 1)

    with pytest.raises(RuntimeError, match='choice generation failed'):
        asyncio.run(collect())
    assert engine.sibling_closed
    assert context.session_manager.sessions == {}


def test_stream_usage_is_omitted_when_a_choice_has_no_usage(
        chat_endpoint, fake_raw_request):

    class IncompleteUsageEngine(_PreprocessingEngine):
        model_name = 'fake-model'
        backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)

        def __init__(self, original_engine):
            self.session_mgr = original_engine.session_mgr
            self.tokenizer = original_engine.tokenizer
            self.call_count = 0
            self.gen_configs = []

        def generate(self, preprocessed, **kwargs):
            self.call_count += 1
            index = self.call_count

            async def generate():
                yield SimpleNamespace(
                    response=f'choice-{index}',
                    token_ids=[index],
                    input_token_len=4,
                    generate_token_len=1,
                    finish_reason='stop' if index == 1 else None,
                    logprobs=None,
                    cached_tokens=0,
                    routed_experts=None,
                    cache_block_ids=None,
                )

            return generate()

    endpoint, context = chat_endpoint
    context.async_engine = IncompleteUsageEngine(context.async_engine)
    request = _request(
        n=2,
        stream=True,
        stream_options={'include_usage': True},
    )
    response = asyncio.run(endpoint(request, fake_raw_request))
    payloads = _sse_payloads(asyncio.run(_collect_stream(response)))

    assert not [
        payload for payload in payloads if payload.get('usage') is not None
    ]
    assert context.session_manager.sessions == {}
