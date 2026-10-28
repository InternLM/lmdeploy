# Copyright (c) OpenMMLab. All rights reserved.
"""Raw JSON extra_body controls reach preprocessing with top-level
precedence."""
import asyncio

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('top_level', [False, True, None])
@pytest.mark.parametrize('field,nested,override', [
    ('enable_thinking', False, True),
    ('media_io_kwargs', {'video': {'num_frames': 3}}, {'video': {'num_frames': 4}}),
])
def test_nested_controls_reach_preprocessor(chat_endpoint, fake_raw_request, stream,
                                           top_level, field, nested, override):
    endpoint, context = chat_endpoint
    original = context.async_engine.preprocess
    seen = []

    async def preprocess(prompt, session, **kwargs):
        seen.append(kwargs)
        return await original(prompt, session, **kwargs)

    context.async_engine.preprocess = preprocess
    options = {'extra_body': {field: nested}}
    if top_level is not False:
        options[field] = override if top_level else None
    request = ChatCompletionRequest(model='fake-model', messages=[{'role': 'user', 'content': 'Hi'}],
                                    stream=stream, **options)

    async def run():
        response = await endpoint(request, fake_raw_request)
        if stream:
            return [event async for event in response.body_iterator]
        return response

    asyncio.run(run())
    expected = nested if top_level is False else override if top_level else None
    actual = ((seen[0]['chat_template_kwargs'] or {}).get('enable_thinking')
              if field == 'enable_thinking' else seen[0][field])
    assert actual == expected
