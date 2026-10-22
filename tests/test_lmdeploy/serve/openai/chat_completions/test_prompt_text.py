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
