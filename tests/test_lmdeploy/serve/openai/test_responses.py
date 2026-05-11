from __future__ import annotations

import json
from types import SimpleNamespace

from lmdeploy.serve.openai.responses import (
    ResponsesRequest,
    _make_response,
    _messages_from_input,
    _stream_response,
    _to_generation_config,
    _validate_text_v1_request,
)


def _sse_payloads(events: list[str]):
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith('data: '):
                payloads.append(json.loads(line.removeprefix('data: ')))
    return payloads


def test_responses_string_input_maps_to_user_message():
    request = ResponsesRequest(model='fake-model', input='Hi there')

    assert _messages_from_input(request) == [{'role': 'user', 'content': 'Hi there'}]


def test_responses_maps_instructions_and_typed_message_input():
    request = ResponsesRequest(
        model='fake-model',
        instructions='You are concise.',
        input=[{
            'type': 'message',
            'role': 'user',
            'content': [{
                'type': 'input_text',
                'text': 'Say hello.',
            }],
        }],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'system',
            'content': 'You are concise.',
        },
        {
            'role': 'user',
            'content': 'Say hello.',
        },
    ]


def test_responses_generation_config_mapping():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        stop=['!'],
        text={
            'format': {
                'type': 'json_schema',
                'name': 'answer',
                'schema': {
                    'type': 'object',
                },
                'strict': True,
            }
        },
    )

    gen_config = _to_generation_config(request)

    assert gen_config.max_new_tokens == 32
    assert gen_config.temperature == 0.2
    assert gen_config.top_p == 0.9
    assert gen_config.top_k == 20
    assert gen_config.stop_words == ['!']
    assert gen_config.response_format == {
        'type': 'json_schema',
        'json_schema': {
            'name': 'answer',
            'schema': {
                'type': 'object',
            },
            'strict': True,
        },
    }


def test_responses_non_stream_response_shape():
    request = ResponsesRequest(model='fake-model', input='Hi there')

    response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='Hello world!',
        input_tokens=8,
        output_tokens=2,
        finish_reason='stop',
    ).model_dump(exclude_none=True)

    assert response['object'] == 'response'
    assert response['status'] == 'completed'
    assert response['output_text'] == 'Hello world!'
    assert response['output'][0]['type'] == 'message'
    assert response['output'][0]['content'][0] == {
        'type': 'output_text',
        'text': 'Hello world!',
        'annotations': [],
    }
    assert response['usage'] == {
        'input_tokens': 8,
        'output_tokens': 2,
        'total_tokens': 10,
    }


def test_responses_rejects_agentic_fields_for_text_v1():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        tools=[{
            'type': 'function',
            'name': 'search',
        }],
    )

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'tools'


def test_responses_rejects_unsupported_input_items():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'message',
            'role': 'user',
            'content': [{
                'type': 'input_image',
                'image_url': 'https://example.com/cat.png',
            }],
        }],
    )

    try:
        _messages_from_input(request)
    except ValueError as err:
        assert 'input_image' in str(err)
    else:
        raise AssertionError('input_image should be rejected by Text V1')


def test_responses_streaming_sse_shape():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='Hello ',
            input_token_len=8,
            generate_token_len=1,
            finish_reason=None,
        )
        yield SimpleNamespace(
            response='world!',
            input_token_len=8,
            generate_token_len=2,
            finish_reason='stop',
        )

    async def _collect_events():
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    import asyncio

    events = asyncio.run(_collect_events())
    body = ''.join(events)
    payloads = _sse_payloads(events)

    assert 'event: response.created' in body
    assert 'event: response.in_progress' in body
    assert 'event: response.output_item.added' in body
    assert 'event: response.content_part.added' in body
    assert 'event: response.output_text.delta' in body
    assert 'event: response.completed' in body
    assert any(payload.get('delta') == 'Hello ' for payload in payloads)
    added_item = next(payload['item'] for payload in payloads if payload['type'] == 'response.output_item.added')
    done_item = next(payload['item'] for payload in payloads if payload['type'] == 'response.output_item.done')
    assert done_item['id'] == added_item['id']
    assert payloads[-1]['type'] == 'response.completed'
    assert payloads[-1]['response']['output_text'] == 'Hello world!'
