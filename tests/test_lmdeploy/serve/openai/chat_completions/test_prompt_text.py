# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('options, expected', [
    ({}, False),
    ({'return_prompt_text': True}, True),
    ({'extra_body': {'return_prompt_text': True}}, True),
    ({'return_prompt_text': False, 'extra_body': {'return_prompt_text': True}}, False),
])
def test_chat_echoes_rendered_prompt_only_when_requested(chat_endpoint, fake_raw_request, stream, n, options, expected):
    endpoint, context = chat_endpoint
    rendered = 'Reasoning Effort: High\n[user]Hello[assistant]'
    original_preprocess = context.async_engine.preprocess

    async def preprocess(prompt, session, **kwargs):
        assert kwargs['reasoning_effort'] == 'high'
        await original_preprocess(prompt, session, **kwargs)
        return SimpleNamespace(inputs={'prompt': rendered, 'input_ids': [1, 2]})

    context.async_engine.preprocess = preprocess
    request = ChatCompletionRequest(
        model='fake-model', messages=[{'role': 'user', 'content': 'Hello'}],
        reasoning_effort='high', stream=stream, n=n, **options)

    async def run():
        response = await endpoint(request, fake_raw_request)
        if not stream:
            return [response]
        events = [event async for event in response.body_iterator]
        return [json.loads(line[6:]) for event in events for line in event.splitlines()
                if line.startswith('data: ') and line != 'data: [DONE]']

    responses = asyncio.run(run())
    prompts = [response['prompt_text'] for response in responses if 'prompt_text' in response]
    assert bool(prompts) is expected
    assert all(prompt == rendered for prompt in prompts)
    if stream:
        seen = set()
        for response in responses:
            for choice in response['choices']:
                if choice['index'] not in seen:
                    assert ('prompt_text' in response) is expected
                    seen.add(choice['index'])
        assert seen == set(range(n))


@pytest.mark.parametrize('return_token_ids', [False, True])
@pytest.mark.parametrize('first_index', [0, 1])
@pytest.mark.parametrize('rendered', ['[user]Hello[assistant]', None])
def test_staggered_choices_keep_prompt_on_each_first_chunk(
        chat_endpoint, fake_raw_request, first_index, rendered, return_token_ids):
    endpoint, context = chat_endpoint
    original_preprocess = context.async_engine.preprocess

    async def preprocess(prompt, session, **kwargs):
        await original_preprocess(prompt, session, **kwargs)
        return SimpleNamespace(inputs={'prompt': rendered, 'input_ids': [10, 11, 12, 13]},
                               index=kwargs['gen_config'].random_seed - 42)

    async def run():
        release = [asyncio.Event(), asyncio.Event()]
        queued = [asyncio.Event(), asyncio.Event()]
        finish = asyncio.Event()

        def generate(preprocessed, **kwargs):
            index = preprocessed.index

            def chunk(content, count, finish_reason=None):
                return SimpleNamespace(
                    response=content, token_ids=[count], input_token_len=4, generate_token_len=count,
                    finish_reason=finish_reason, logprobs=None, cached_tokens=0, routed_experts=None,
                    cache_block_ids=None)

            async def results():
                if index == first_index:
                    yield chunk(f'{index}-first', 1)
                await release[index].wait()
                count = 2 if index == first_index else 1
                yield chunk(f'{index}-second' if index == first_index else f'{index}-first', count)
                # The child stream has queued this delta before asking the
                # engine for another one. Hold completion until both arrive.
                queued[index].set()
                await finish.wait()
                yield chunk(f'{index}-last', count + 1, 'stop')

            return results()

        context.async_engine.preprocess = preprocess
        context.async_engine.generate = generate
        response = await endpoint(ChatCompletionRequest(
            model='fake-model', messages=[{'role': 'user', 'content': 'Hello'}], seed=42,
            n=2, stream=True, return_prompt_text=True, return_token_ids=return_token_ids,
            stream_options={'include_usage': True}), fake_raw_request)
        try:
            first = json.loads((await anext(response.body_iterator)).removeprefix('data: '))
            assert [choice['index'] for choice in first['choices']] == [first_index]
            assert first['prompt_text'] == rendered
            assert ('prompt_token_ids' in first) is return_token_ids
            if return_token_ids:
                assert first['prompt_token_ids'] == [10, 11, 12, 13]

            # Queue the leading choice's second delta first, then the other
            # choice's first delta, while the consumer is suspended at yield.
            # Events make this ordering deterministic without timing sleeps.
            for index in (first_index, 1 - first_index):
                release[index].set()
                await queued[index].wait()
            merged = json.loads((await anext(response.body_iterator)).removeprefix('data: '))
            assert [choice['index'] for choice in merged['choices']] == [0, 1]
            assert [choice['delta']['content'] for choice in merged['choices']] == [
                f'{index}-second' if index == first_index else f'{index}-first' for index in range(2)]
            assert merged['prompt_text'] == rendered
            assert ('prompt_token_ids' in merged) is return_token_ids
            if return_token_ids:
                assert merged['prompt_token_ids'] == [10, 11, 12, 13]

            finish.set()
            remaining = [event async for event in response.body_iterator]
            assert remaining[-1] == 'data: [DONE]\n\n'
            payloads = [json.loads(event.removeprefix('data: ')) for event in remaining[:-1]]
            assert all('prompt_text' not in payload and 'prompt_token_ids' not in payload for payload in payloads)
            assert payloads[-1]['choices'] == []
            assert payloads[-1]['usage']['completion_tokens'] == 5
            assert {choice['index'] for payload in payloads for choice in payload['choices']
                    if choice.get('finish_reason') == 'stop'} == {0, 1}
        finally:
            await response.close()

    async def run_with_timeout():
        await asyncio.wait_for(run(), timeout=5)

    asyncio.run(run_with_timeout())
    assert len(context.session_manager.removed) == 2
    assert context.session_manager.sessions == {}
