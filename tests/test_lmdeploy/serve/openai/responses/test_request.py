# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import json

from openai.types.responses import ResponseFunctionToolCall

from lmdeploy.serve.openai.responses import ResponsesRequest
from lmdeploy.serve.openai.responses.protocol import ResponseInputOutputItem
from lmdeploy.serve.openai.responses.request import (
    _messages_from_input,
    _openai_tools_from_responses,
    _to_generation_config,
    _tool_choice_from_responses,
    _validate_text_v1_request,
)


def test_responses_request_uses_structured_input_item_alias():
    item: ResponseInputOutputItem = ResponseFunctionToolCall(
        id='fc_1',
        call_id='call_1',
        name='search',
        arguments='{"query":"lmdeploy"}',
        type='function_call',
    )
    request = ResponsesRequest(model='fake-model', input=[item])

    assert _messages_from_input(request) == [{
        'role': 'assistant',
        'content': None,
        'tool_calls': [{
            'id': 'call_1',
            'type': 'function',
            'function': {
                'name': 'search',
                'arguments': '{"query":"lmdeploy"}',
            },
        }],
    }]


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


def test_responses_merges_multiple_system_messages():
    request = ResponsesRequest(
        model='fake-model',
        instructions='You are concise.',
        input=[
            {
                'type': 'message',
                'role': 'developer',
                'content': 'Follow the repo instructions.',
            },
            {
                'type': 'message',
                'role': 'system',
                'content': 'Use plain text.',
            },
            {
                'type': 'message',
                'role': 'user',
                'content': 'Say hello.',
            },
        ],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'system',
            'content': 'You are concise.\n\nFollow the repo instructions.\n\nUse plain text.',
        },
        {
            'role': 'user',
            'content': 'Say hello.',
        },
    ]


def test_responses_moves_developer_messages_before_conversation():
    request = ResponsesRequest(
        model='fake-model',
        input=[
            {
                'type': 'message',
                'role': 'user',
                'content': 'Say hello.',
            },
            {
                'type': 'message',
                'role': 'developer',
                'content': 'Follow the repo instructions.',
            },
        ],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'system',
            'content': 'Follow the repo instructions.',
        },
        {
            'role': 'user',
            'content': 'Say hello.',
        },
    ]


def test_responses_maps_function_call_history_to_chat_messages():
    request = ResponsesRequest(
        model='fake-model',
        input=[
            {
                'type': 'message',
                'role': 'user',
                'content': 'Search for lmdeploy.',
            },
            {
                'type': 'function_call',
                'call_id': 'call_123',
                'name': 'search',
                'arguments': '{"query":"lmdeploy"}',
            },
            {
                'type': 'function_call_output',
                'call_id': 'call_123',
                'output': '{"result":"ok"}',
            },
        ],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'user',
            'content': 'Search for lmdeploy.',
        },
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{
                'id': 'call_123',
                'type': 'function',
                'function': {
                    'name': 'search',
                    'arguments': '{"query":"lmdeploy"}',
                },
            }],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_123',
            'content': '{"result":"ok"}',
        },
    ]


def test_responses_rejects_non_string_function_call_arguments():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'function_call',
            'call_id': 'call_123',
            'name': 'search',
            'arguments': {
                'query': 'lmdeploy',
            },
        }],
    )

    try:
        _messages_from_input(request)
    except ValueError as err:
        assert 'Unsupported `arguments` in function_call item' in str(err)
    else:
        raise AssertionError(
            'non-string function_call arguments should be rejected')


def test_responses_maps_function_call_output_text_parts():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'function_call_output',
            'call_id': 'call_123',
            'output': [{
                'type': 'input_text',
                'text': '{"result":"ok"}',
            }],
        }],
    )

    assert _messages_from_input(request) == [{
        'role': 'tool',
        'tool_call_id': 'call_123',
        'content': '{"result":"ok"}',
    }]


def test_responses_maps_function_tools_to_openai_tools():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        tools=[{
            'type': 'function',
            'name': 'search',
            'description': 'demo',
            'parameters': {
                'type': 'object',
            },
        }, {
            'type': 'web_search',
        }],
    )

    tools = _openai_tools_from_responses(request)

    assert tools is not None
    assert len(tools) == 1
    assert tools[0].function.name == 'search'
    assert tools[0].function.description == 'demo'
    assert tools[0].function.parameters == {'type': 'object'}


def test_responses_tool_choice_without_function_tools_validation():
    assert _tool_choice_from_responses('auto', None) == 'none'
    assert _tool_choice_from_responses('none', None) == 'none'

    for tool_choice in ('required', {'type': 'function', 'name': 'search'}):
        try:
            _tool_choice_from_responses(tool_choice, None)
        except ValueError as err:
            assert 'tools' in str(err)
        else:
            raise AssertionError(
                f'{tool_choice!r} should require function tools')


def test_responses_named_tool_choice_must_match_function_tools():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        tools=[{
            'type': 'function',
            'name': 'search',
        }],
        tool_choice={
            'type': 'function',
            'name': 'missing',
        },
    )
    tools = _openai_tools_from_responses(request)

    try:
        _tool_choice_from_responses(request.tool_choice, tools)
    except ValueError as err:
        assert 'not found' in str(err)
    else:
        raise AssertionError(
            'named tool_choice should match one of the function tools')


def test_responses_explicit_unsupported_tool_choice_is_rejected():
    try:
        _tool_choice_from_responses({'type': 'web_search_preview'}, None)
    except ValueError as err:
        assert 'Unsupported tool_choice type' in str(err)
    else:
        raise AssertionError(
            'explicit unsupported tool_choice should be rejected')


def test_responses_generation_config_mapping():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        stop=['!'],
        repetition_penalty=1.1,
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
    assert gen_config.repetition_penalty == 1.1
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


def test_responses_rejects_unsupported_agentic_fields_for_text_v1():
    request = ResponsesRequest(model='fake-model',
                               input='Hi',
                               previous_response_id='resp_123')

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(
        response.body)['error']['param'] == 'previous_response_id'


def test_responses_rejects_unsupported_conversation_for_text_v1():
    request = ResponsesRequest(model='fake-model',
                               input='Hi',
                               conversation={'id': 'conv_123'})

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'conversation'


def test_responses_rejects_unsupported_prompt():
    request = ResponsesRequest(model='fake-model',
                               input='Hi',
                               prompt={'id': 'pmpt_123'})

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'prompt'


def test_responses_rejects_missing_input_for_text_v1():
    request = ResponsesRequest(model='fake-model')

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'input'


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


def test_responses_rejects_reasoning_input_items():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'reasoning',
            'summary': [],
        }],
    )

    try:
        _messages_from_input(request)
    except ValueError as err:
        assert 'reasoning' in str(err)
    else:
        raise AssertionError(
            'reasoning input items should be rejected by Text V1')
