# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for ``n > 1`` server-side fan-out in the chat completions handler.

The fan-out is engine-agnostic handler-layer logic: a single ``n > 1`` request
becomes N independent ``engine.generate()`` calls with distinct random seeds,
collated into N choices. These tests cover the pure aggregation helper
``_fanout_generate_collect`` using fake async generators that mimic the engine's
``GenOut`` yields. Both pytorch and turbomind engines are covered because the
fan-out lives entirely in the handler and treats the engine as a black box.

Note: the repo uses ``asyncio.run`` (no ``pytest-asyncio`` dependency), so async
test bodies are driven through ``asyncio.run``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.endpoints.chat_completions.serving import (
    _fanout_generate_collect,
)


def _fake_gen(outputs):
    """Build an async generator yielding ``GenOut``-like objects."""

    async def _gen():
        for o in outputs:
            yield o

    return _gen()


def _genout(text, completion_tokens, *, prompt_tokens=5, finish_reason='stop'):
    return SimpleNamespace(
        response=text,
        input_token_len=prompt_tokens,
        generate_token_len=completion_tokens,
        finish_reason=finish_reason,
        token_ids=[],
        logprobs=None,
        cached_tokens=0,
        routed_experts=None,
        cache_block_ids=None,
    )


def test_fanout_assigns_distinct_indices_and_aggregates_completion_tokens():
    """Three generators producing 2/3/1 completion tokens are collated into
    three choices with distinct indices; prompt_tokens counted once,
    completion_tokens summed."""
    gens = [
        (0, _fake_gen([_genout('a', 2)])),
        (1, _fake_gen([_genout('b', 3)])),
        (2, _fake_gen([_genout('c', 1)])),
    ]
    choices, usage = asyncio.run(_fanout_generate_collect(gens, prompt_tokens=5))
    assert len(choices) == 3
    assert {c.index for c in choices} == {0, 1, 2}
    assert usage['prompt_tokens'] == 5  # counted once
    assert usage['completion_tokens'] == 6  # 2 + 3 + 1


def test_fanout_prompt_tokens_from_generator_overrides_when_unspecified():
    """When prompt_tokens is passed explicitly it is used as the single
    prompt-token count (never summed across choices)."""
    gens = [(0, _fake_gen([_genout('a', 2, prompt_tokens=99)]))]
    choices, usage = asyncio.run(_fanout_generate_collect(gens, prompt_tokens=7))
    assert usage['prompt_tokens'] == 7
    assert usage['completion_tokens'] == 2


def test_fanout_propagates_error_to_whole_request():
    """If any inner generator raises, the whole fan-out request fails."""

    async def _boom():
        raise RuntimeError('choice 1 failed')
        yield  # noqa: unreachable, makes it an async generator

    with pytest.raises(RuntimeError, match='choice 1 failed'):
        asyncio.run(_fanout_generate_collect([(0, _boom())], prompt_tokens=1))


# ---------------------------------------------------------------------------
# Handler-level integration: exercise the n>1 branch end-to-end with a fake
# engine. Validates wiring (N sessions, N parsers, distinct seeds, aggregated
# usage, N choices) for both streaming and non-streaming.
# ---------------------------------------------------------------------------

from lmdeploy.serve.openai.endpoints.chat_completions.protocol import (  # noqa: E402
    ChatCompletionRequest,
)


def _sse_payloads(text):
    import json
    payloads = []
    for line in text.splitlines():
        if line.startswith('data: '):
            data = line.removeprefix('data: ')
            if data == '[DONE]':
                continue
            payloads.append(json.loads(data))
    return payloads


def test_handler_n3_nonstream_returns_three_choices_with_aggregated_usage(
        chat_endpoint, fake_raw_request):
    """n=3 non-streaming: 3 distinct choices, prompt counted once,
    completion_tokens summed; engine called 3 times with distinct seeds."""
    endpoint, context = chat_endpoint
    request = ChatCompletionRequest(model='fake-model',
                                    messages=[{'role': 'user',
                                               'content': 'hi'}],
                                    n=3,
                                    seed=42,
                                    stream=False)
    response = asyncio.run(endpoint(request, fake_raw_request))

    assert response['object'] == 'chat.completion'
    assert len(response['choices']) == 3
    assert {c['index'] for c in response['choices']} == {0, 1, 2}
    # Each choice got distinct text from a distinct generate() call.
    assert {c['message']['content']
            for c in response['choices']} == {'choice-1', 'choice-2',
                                              'choice-3'}
    # prompt_tokens counted once (4), completion_tokens = 1 + 2 + 3 = 6.
    assert response['usage']['prompt_tokens'] == 4
    assert response['usage']['completion_tokens'] == 6
    # Engine was invoked 3 times with derived seeds 42, 43, 44.
    assert context.async_engine.call_count == 3
    seeds = [gc.random_seed for gc in context.async_engine.gen_configs]
    assert seeds == [42, 43, 44]


def test_handler_n3_stream_interleaves_three_indices_and_aggregates_usage(
        chat_endpoint, fake_raw_request):
    """n=3 streaming: deltas carry indices 0/1/2, final usage chunk sums
    completion tokens across choices."""
    endpoint, context = chat_endpoint
    request = ChatCompletionRequest(model='fake-model',
                                    messages=[{'role': 'user',
                                               'content': 'hi'}],
                                    n=3,
                                    stream=True,
                                    stream_options={'include_usage': True})
    response = asyncio.run(endpoint(request, fake_raw_request))

    # StreamingResponse.body is an async iterable; collect it.
    body_iterator = response.body_iterator

    async def _collect():
        chunks = []
        async for chunk in body_iterator:
            chunks.append(chunk.decode()
                          if isinstance(chunk, bytes) else chunk)
        return ''.join(chunks)

    text = asyncio.run(_collect())
    payloads = _sse_payloads(text)

    choice_indices = set()
    for p in payloads:
        for c in p.get('choices', []):
            choice_indices.add(c['index'])
    assert choice_indices == {0, 1, 2}

    # The final chunk carries aggregated usage (prompt once, completion sum).
    usage_chunks = [p for p in payloads if p.get('usage') is not None]
    assert usage_chunks, 'expected a final usage chunk'
    final_usage = usage_chunks[-1]['usage']
    assert final_usage['prompt_tokens'] == 4
    assert final_usage['completion_tokens'] == 6  # 1 + 2 + 3
    assert text.rstrip().endswith('data: [DONE]')


def test_handler_n1_keeps_single_generator_fast_path(chat_endpoint,
                                                     fake_raw_request):
    """n=1 (default) must not fan out: exactly one engine.generate() call."""
    endpoint, context = chat_endpoint
    request = ChatCompletionRequest(model='fake-model',
                                    messages=[{'role': 'user',
                                               'content': 'hi'}],
                                    stream=False)
    response = asyncio.run(endpoint(request, fake_raw_request))
    assert len(response['choices']) == 1
    assert context.async_engine.call_count == 1


def test_handler_n3_unseeded_leaves_random_seed_none(chat_endpoint,
                                                     fake_raw_request):
    """When request.seed is unset, each sub gen_config keeps random_seed=None
    so the engine randomizes each choice independently."""
    endpoint, context = chat_endpoint
    request = ChatCompletionRequest(model='fake-model',
                                    messages=[{'role': 'user',
                                               'content': 'hi'}],
                                    n=3,
                                    stream=False)
    asyncio.run(endpoint(request, fake_raw_request))
    seeds = [gc.random_seed for gc in context.async_engine.gen_configs]
    assert seeds == [None, None, None]


def test_validation_rejects_oversized_n():
    """Fan-out resource cap: n above _MAX_FANOUT_N is rejected."""
    from lmdeploy.serve.openai.endpoints.chat_completions.validation import \
        _MAX_FANOUT_N, check_request
    from types import SimpleNamespace

    request = ChatCompletionRequest(model='fake-model',
                                    messages=[{'role': 'user',
                                               'content': 'hi'}],
                                    n=_MAX_FANOUT_N + 1)
    ctx = SimpleNamespace(
        engine_config=SimpleNamespace(logprobs_mode=None, adapters=[]),
        session_manager=SimpleNamespace(has=lambda sid: False),
        response_parser_cls=None,
    )
    msg = check_request(request, ctx)
    assert 'exceeds the maximum' in msg


def test_validation_rejects_negative_seed():
    from lmdeploy.serve.openai.endpoints.chat_completions.validation import \
        check_request
    from types import SimpleNamespace

    request = ChatCompletionRequest(model='fake-model',
                                    messages=[{'role': 'user',
                                               'content': 'hi'}],
                                    seed=-7)
    ctx = SimpleNamespace(
        engine_config=SimpleNamespace(logprobs_mode=None, adapters=[]),
        session_manager=SimpleNamespace(has=lambda sid: False),
        response_parser_cls=None,
    )
    msg = check_request(request, ctx)
    assert 'non-negative' in msg
