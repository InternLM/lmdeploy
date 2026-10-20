# Copyright (c) OpenMMLab. All rights reserved.
"""HTTP regressions for tool images; generation is simulated, media decoding is
real."""
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import normalize_chat_request
from lmdeploy.serve.processors.multimodal import MultimodalProcessor
from lmdeploy.vl.model.base import VisionModel


def _image(color):
    buffer = BytesIO()
    Image.new('RGB', (28, 28), color).save(buffer, format='PNG')
    data = base64.b64encode(buffer.getvalue()).decode()
    return dict(type='image_url', image_url=dict(url=f'data:image/png;base64,{data}'))


@pytest.mark.parametrize('stream', [False, True])
def test_http_tool_images_reach_shared_preprocessing_with_correct_ownership(chat_endpoint, stream, hf_media_template):
    _, context = chat_endpoint
    seen = {}
    template = hf_media_template()

    class NormalizedParser(context.response_parser_cls):

        def __init__(self, request):
            super().__init__(normalize_chat_request(request))

    context.response_parser_cls = NormalizedParser

    def get_input_prompt(messages, **kwargs):
        seen['rendered'] = messages
        return VisionModel.get_input_prompt(messages, **kwargs)

    async def preprocess_images(messages, input_prompt, **kwargs):
        assert messages is seen['rendered']
        seen['media'] = VisionModel.collect_multimodal_items(messages)
        return messages

    encoder = SimpleNamespace(_uses_new_preprocess=True, model=SimpleNamespace(get_input_prompt=get_input_prompt),
                              preprocess=preprocess_images)
    processor = MultimodalProcessor(None, template, vl_encoder=encoder, backend='pytorch')
    original_preprocess = context.async_engine.preprocess

    async def preprocess(prompt, session, **kwargs):
        seen['received'] = prompt
        await processor.get_prompt_input(prompt, do_preprocess=kwargs['do_preprocess'])
        return await original_preprocess(prompt, session, **kwargs)

    context.async_engine.preprocess = preprocess
    messages = [dict(role='user', content='Read the tool screenshots.'), dict(role='assistant', content=None,
                tool_calls=[dict(id=identifier, type='function', function=dict(name='screenshot', arguments='{}'))
                            for identifier in ('first', 'second')])]
    messages += [dict(role='tool', tool_call_id=identifier, content=[
        dict(type='text', text='before'), _image(color), dict(type='text', text='between'), _image('white'),
        dict(type='text', text='after'),
    ]) for identifier, color in [('second', 'blue'), ('first', 'red')]]
    app = FastAPI()
    router = APIRouter()
    register(router, context)
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=dict(model='fake-model', messages=messages, stream=stream))

    assert response.status_code == 200
    if stream:
        assert response.headers['content-type'].startswith('text/event-stream')
        assert response.text.rstrip().endswith('data: [DONE]')
        assert 'choice-1' in response.text
    else:
        assert response.json()['choices'][0]['message']['content'] == 'choice-1'
        assert response.json()['choices'][0]['finish_reason'] == 'stop'
    # Request validation/forwarding keeps IDs, content, and arrival order;
    # GLM preparation gives rendering and media collection the same call order.
    assert seen['received'] == normalize_chat_request(
        ChatCompletionRequest(model='fake-model', messages=messages)).messages
    assert [m['tool_call_id'] for m in seen['rendered'][2:]] == ['first', 'second']
    for message in seen['rendered'][2:]:
        assert [p['type'] for p in message['content']] == ['text', 'image', 'text', 'image', 'text']
        assert [p['text'] for p in message['content'] if p['type'] == 'text'] == ['before', 'between', 'after']
    assert [image.getpixel((0, 0)) for _, image, _ in seen['media']] == [
        (255, 0, 0), (255, 255, 255), (0, 0, 255), (255, 255, 255),
    ]
    assert len(context.session_manager.removed) == 1
    assert context.session_manager.sessions == {}
