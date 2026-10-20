# Copyright (c) OpenMMLab. All rights reserved.
"""Responses image conversion and HTTP tests with real media decoding."""

from __future__ import annotations

import asyncio
import base64
import copy
from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.responses import ResponseInputImage
from PIL import Image

from lmdeploy.serve.openai.responses import ResponsesRequest, create_responses_router
from lmdeploy.serve.openai.responses.request import messages_from_input
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


def _image(color='red', image_format='PNG'):
    buffer = BytesIO()
    Image.new('RGB', (32, 32), color).save(buffer, format=image_format)
    mime = {'PNG': 'png', 'JPEG': 'jpeg', 'WEBP': 'webp'}[image_format]
    data = base64.b64encode(buffer.getvalue()).decode()
    return dict(type='input_image', image_url=f'data:image/{mime};base64,{data}')


def _request(parts, **kwargs):
    return dict(model='fake-model', input=[dict(role='user', content=parts)], **kwargs)


@pytest.mark.parametrize('detail', [None, 'auto', 'low', 'high', 'original'])
def test_image_parameters_survive_request_validation(detail):
    image = dict(type='input_image', image_url='https://example.com/image.png',
                 min_dynamic_patch=1, max_dynamic_patch=4)
    if detail is not None:
        image['detail'] = detail
    request = ResponsesRequest(**_request([image]))
    before = copy.deepcopy(request.model_dump())

    messages = messages_from_input(request)

    expected = dict(url=image['image_url'], min_dynamic_patch=1, max_dynamic_patch=4)
    if detail is not None:
        expected['detail'] = detail
    assert messages == [dict(role='user', content=[dict(type='image_url', image_url=expected)])]
    assert request.model_dump() == before


def test_sdk_image_object_is_supported():
    image = ResponseInputImage(type='input_image', image_url='http://example.com/image.png', detail='auto')
    request = ResponsesRequest(**_request([image]))
    assert messages_from_input(request)[0]['content'] == [
        dict(type='image_url', image_url=dict(url=image.image_url, detail='auto')),
    ]


def test_multiturn_text_and_image_order():
    red, blue = _image(), _image('blue')
    request = ResponsesRequest(
        model='fake-model',
        instructions='Keep image order.',
        input=[
            dict(role='user', content=[dict(type='input_text', text='before'), red,
                                       dict(type='input_text', text='between'), blue,
                                       dict(type='input_text', text='after')]),
            dict(role='assistant', content=[dict(type='output_text', text='Understood.')]),
            dict(role='user', content=[
                dict(type='input_text', text='Repeat'), dict(type='input_text', text=' colors.'),
            ]),
        ])

    messages = messages_from_input(request)

    assert messages == [
        dict(role='system', content='Keep image order.'),
        dict(role='user', content=[
            dict(type='text', text='before'), dict(type='image_url', image_url=dict(url=red['image_url'])),
            dict(type='text', text='between'), dict(type='image_url', image_url=dict(url=blue['image_url'])),
            dict(type='text', text='after'),
        ]),
        dict(role='assistant', content='Understood.'),
        dict(role='user', content='Repeat colors.'),
    ]


@pytest.mark.parametrize('image_format', ['PNG', 'JPEG', 'WEBP'])
def test_images_use_existing_media_decoder(image_format):
    messages = messages_from_input(ResponsesRequest(**_request([_image(image_format=image_format)])))
    parsed = asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))
    image = parsed[0]['content'][0]
    assert image['type'] == 'image'
    assert image['data'].size == (32, 32)
    red, green, blue = image['data'].getpixel((0, 0))
    assert red > 240 and green < 10 and blue < 10


@pytest.mark.parametrize('role', ['assistant', 'system', 'developer'])
def test_images_outside_user_messages_remain_unsupported(role):
    request = ResponsesRequest(model='fake-model', input=[dict(role=role, content=[_image()])])
    with pytest.raises(ValueError, match=r'input_image.*input\[0\].content\[0\]'):
        messages_from_input(request)


def test_tool_result_images_remain_unsupported():
    request = ResponsesRequest(model='fake-model', input=[
        dict(type='function_call_output', call_id='call_1', output=[_image()]),
    ])
    with pytest.raises(ValueError, match=r'input_image.*input\[0\].output\[0\]'):
        messages_from_input(request)


@pytest.mark.parametrize('part, message', [
    (dict(type='input_image'), 'image_url'),
    (dict(type='input_image', image_url=''), 'image_url'),
    (dict(type='input_image', image_url='  '), 'image_url'),
    (dict(type='input_image', image_url=123), 'image_url'),
    (dict(type='input_image', image_url=dict(url='https://example.com/image.png')), 'image_url'),
    (dict(type='input_image', file_id='file_1'), 'file_id'),
    (dict(type='input_image', image_url='https://example.com/image.png', file_id='file_1'), 'file_id'),
    (dict(type='input_image', image_url='file:///tmp/image.png'), 'HTTP'),
    (dict(type='input_image', image_url='/tmp/image.png'), 'HTTP'),
    (dict(type='input_image', image_url='https:///image.png'), 'HTTP'),
    (dict(type='input_image', image_url='http://[invalid'), 'image_url'),
    (dict(type='input_image', image_url='data:text/plain;base64,YQ=='), 'HTTP'),
    (dict(type='input_image', image_url='https://example.com/image.png', detail='invalid'), 'detail'),
    (dict(type='input_file', file_id='file_1'), 'input_file'),
    (dict(type='unknown'), 'unknown'),
])
@pytest.mark.parametrize('stream', [False, True])
def test_invalid_images_return_http_400_before_preprocessing(responses_endpoint, part, message, stream):
    _, context = responses_endpoint
    app = FastAPI()
    app.include_router(create_responses_router(context))
    with TestClient(app) as client:
        response = client.post('/v1/responses', json=_request([part], stream=stream))

    assert response.status_code == 400
    error = response.json()['error']
    assert error['type'] == 'invalid_request_error'
    assert error['param'] == 'input'
    assert message in error['message']
    assert 'input[0].content[0]' in error['message']
    assert context.async_engine.prompt is None
    assert context.sessions == []


@pytest.mark.parametrize('stream', [False, True])
def test_http_images_reach_preprocessor_in_order(responses_endpoint, stream):
    _, context = responses_endpoint
    original_preprocess = context.async_engine.preprocess
    decoded = []

    async def preprocess(prompt, session, **kwargs):
        decoded.extend(await MultimodalProcessor.async_parse_multimodal_item(prompt))
        return await original_preprocess(prompt, session, **kwargs)

    context.async_engine.preprocess = preprocess
    app = FastAPI()
    app.include_router(create_responses_router(context))
    parts = [dict(type='input_text', text='Colors in order?'), _image(), _image('blue')]
    with TestClient(app) as client:
        response = client.post('/v1/responses', json=_request(parts, stream=stream))

    assert response.status_code == 200
    if stream:
        assert response.headers['content-type'].startswith('text/event-stream')
        assert 'event: response.completed' in response.text
        assert 'event: response.failed' not in response.text
    else:
        assert response.json()['status'] == 'completed'
        assert response.json()['output_text'] == 'ok'
    content = decoded[0]['content']
    assert [part['type'] for part in content] == ['text', 'image', 'image']
    assert content[0]['text'] == 'Colors in order?'
    assert [part['data'].getpixel((0, 0)) for part in content[1:]] == [(255, 0, 0), (0, 0, 255)]
    assert context.async_engine.preprocess_kwargs['do_preprocess'] is True
    assert context.async_engine.session_mgr.removed == context.sessions
