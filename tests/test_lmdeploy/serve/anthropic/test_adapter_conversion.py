from __future__ import annotations

from lmdeploy.serve.anthropic.adapter import to_openai_messages
from lmdeploy.serve.anthropic.protocol import MessagesRequest


def _make_request(*, messages, system=None):
    return MessagesRequest.model_validate(
        {
            'model': 'fake-model',
            'max_tokens': 32,
            'messages': messages,
            'system': system,
        })


def test_to_openai_messages_keeps_plain_text_messages():
    request = _make_request(messages=[{'role': 'user', 'content': 'hello'}])
    assert to_openai_messages(request) == [{'role': 'user', 'content': 'hello'}]


def test_to_openai_messages_converts_system_text_blocks():
    request = _make_request(
        system=[
            {
                'type': 'text',
                'text': 'You are helpful.',
            },
            {
                'type': 'text',
                'text': 'Answer briefly.',
            },
        ],
        messages=[{
            'role': 'user',
            'content': 'hi',
        }],
    )
    assert to_openai_messages(request)[0] == {
        'role': 'system',
        'content': 'You are helpful.Answer briefly.',
    }


def test_to_openai_messages_skips_anthropic_billing_header_system_block():
    request = _make_request(
        system=[
            {
                'type': 'text',
                'text': 'x-anthropic-billing-header: request-hash',
            },
            {
                'type': 'text',
                'text': 'You are helpful.',
            },
        ],
        messages=[{
            'role': 'user',
            'content': 'hi',
        }],
    )
    assert to_openai_messages(request)[0] == {
        'role': 'system',
        'content': 'You are helpful.',
    }


def test_to_openai_messages_converts_image_blocks_to_openai_image_urls():
    request = _make_request(
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'describe these',
                },
                {
                    'type': 'image',
                    'source': {
                        'type': 'url',
                        'url': 'https://example.com/cat.png',
                    },
                },
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': 'abc123',
                    },
                },
            ],
        }])
    assert to_openai_messages(request) == [{
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': 'describe these',
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'https://example.com/cat.png',
                },
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,abc123',
                },
            },
        ],
    }]


def test_to_openai_messages_maps_tool_use_to_tool_calls():
    request = _make_request(
        messages=[{
            'role': 'assistant',
            'content': [{
                'type': 'tool_use',
                'id': 'toolu_123',
                'name': 'search',
                'input': {
                    'query': 'lmdeploy'
                },
            }],
        }])
    messages = to_openai_messages(request)
    assert messages == [{
        'role': 'assistant',
        'tool_calls': [{
            'id': 'toolu_123',
            'type': 'function',
            'function': {
                'name': 'search',
                'arguments': '{"query": "lmdeploy"}',
            },
        }],
    }]


def test_to_openai_messages_uses_vllm_style_fallback_tool_call_id():
    request = _make_request(
        messages=[{
            'role': 'assistant',
            'content': [{
                'type': 'tool_use',
                'name': 'search',
                'input': {
                    'query': 'lmdeploy'
                },
            }],
        }])
    tool_call_id = to_openai_messages(request)[0]['tool_calls'][0]['id']
    assert tool_call_id.startswith('call_')


def test_to_openai_messages_maps_user_tool_result_to_tool_role_message():
    request = _make_request(
        messages=[
            {
                'role': 'assistant',
                'content': [{
                    'type': 'tool_use',
                    'id': 'toolu_123',
                    'name': 'search',
                    'input': {
                        'query': 'lmdeploy'
                    },
                }],
            },
            {
                'role': 'user',
                'content': [{
                    'type': 'tool_result',
                    'tool_use_id': 'toolu_123',
                    'content': [{
                        'type': 'text',
                        'text': 'LMDeploy serves LLMs.',
                    }],
                }],
            },
        ])
    messages = to_openai_messages(request)
    assert messages[1] == {
        'role': 'tool',
        'tool_call_id': 'toolu_123',
        'content': 'LMDeploy serves LLMs.',
    }


def test_to_openai_messages_maps_user_tool_result_images_to_user_message():
    request = _make_request(
        messages=[{
            'role': 'user',
            'content': [{
                'type': 'tool_result',
                'tool_use_id': 'toolu_123',
                'content': [
                    {
                        'type': 'text',
                        'text': 'see image',
                    },
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/jpeg',
                            'data': 'xyz789',
                        },
                    },
                ],
            }],
        }])
    assert to_openai_messages(request) == [
        {
            'role': 'tool',
            'tool_call_id': 'toolu_123',
            'content': 'see image',
        },
        {
            'role': 'user',
            'content': [{
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/jpeg;base64,xyz789',
                },
            }],
        },
    ]


def test_to_openai_messages_puts_assistant_tool_result_into_content():
    request = _make_request(
        messages=[{
            'role': 'assistant',
            'content': [{
                'type': 'tool_result',
                'tool_use_id': 'toolu_123',
                'content': 'search done',
            }],
        }])
    messages = to_openai_messages(request)
    assert messages == [{
        'role': 'assistant',
        'content': 'Tool result: search done',
    }]


def test_to_openai_messages_maps_thinking_block_to_reasoning_content():
    request = _make_request(
        messages=[{
            'role': 'assistant',
            'content': [{
                'type': 'thinking',
                'thinking': 'internal chain',
            }],
        }])
    messages = to_openai_messages(request)
    assert messages == [{
        'role': 'assistant',
        'reasoning_content': 'internal chain',
    }]


def test_to_openai_messages_ignores_redacted_thinking_block():
    request = _make_request(
        messages=[{
            'role': 'assistant',
            'content': [{
                'type': 'redacted_thinking',
                'data': 'opaque',
            }],
        }])
    assert to_openai_messages(request) == []


def test_to_openai_messages_skips_empty_user_tool_result_block_message():
    request = _make_request(
        messages=[{
            'role': 'user',
            'content': [{
                'type': 'tool_result',
                'tool_use_id': 'toolu_123',
                'content': None,
            }],
        }])
    messages = to_openai_messages(request)
    assert messages == [{
        'role': 'tool',
        'tool_call_id': 'toolu_123',
        'content': '',
    }]
