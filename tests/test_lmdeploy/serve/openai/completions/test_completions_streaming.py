# Copyright (c) OpenMMLab. All rights reserved.
import json
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.core.async_engine import GenOut
from lmdeploy.serve.openai.endpoints.completions import register


@pytest.mark.parametrize('prompt', ['first', ['first'], ['first', 'second']])
@pytest.mark.parametrize('include_usage', [False, True])
def test_streaming_choices_match_non_streaming(prompt, include_usage):
    class AsyncEngine:
        model_name = 'test-model'

        async def preprocess(self, prompt, session, **kwargs):
            return SimpleNamespace(inputs={'prompt': prompt})

        async def generate(self, request, **kwargs):
            yield GenOut(response=request.inputs['prompt'],
                         input_token_len=1,
                         generate_token_len=1)
            yield GenOut(response=' answer',
                         input_token_len=1,
                         generate_token_len=2)
            yield GenOut(response='',
                         input_token_len=1,
                         generate_token_len=2,
                         finish_reason='stop')

    context = SimpleNamespace(
        async_engine=AsyncEngine(),
        engine_config=SimpleNamespace(logprobs_mode=None),
        session_manager=SimpleNamespace(has=lambda _: False, remove=lambda _: None),
        default_gen_config={},
        create_session=lambda *args: object(),
    )
    router = APIRouter()
    register(router, context)
    app = FastAPI()
    app.include_router(router)
    payload = dict(model='test-model', prompt=prompt, n=1, max_completion_tokens=2)

    with TestClient(app) as client:
        response = client.post('/v1/completions', json=payload)
        assert response.status_code == 200
        expected = response.json()
        response = client.post('/v1/completions', json={
            **payload,
            'stream': True,
            'stream_options': {'include_usage': include_usage},
        })
        assert response.status_code == 200

    events = [line.removeprefix('data: ') for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[-1] == '[DONE]'
    chunks = [json.loads(event) for event in events[:-1]]
    text_by_index = {}
    finish_by_index = {}
    for chunk in chunks:
        for choice in chunk['choices']:
            index = choice['index']
            text_by_index[index] = text_by_index.get(index, '') + choice['text']
            if choice.get('finish_reason') is not None:
                finish_by_index[index] = choice['finish_reason']

    assert text_by_index == {choice['index']: choice['text'] for choice in expected['choices']}
    assert finish_by_index == {choice['index']: choice['finish_reason'] for choice in expected['choices']}
    if include_usage:
        assert chunks[-1]['choices'] == []
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            assert chunks[-1]['usage'][field] == expected['usage'][field]
    else:
        assert all('usage' not in chunk for chunk in chunks)
