# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for tool-name validation at the chat-completions boundary."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


def _request(*, stream: bool) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model='fake-model',
        messages=[{
            'role': 'user',
            'content': 'Modify the previous image.',
        }],
        stream=stream,
        tools=[{
            'type': 'function',
            'function': {
                'name': 'search',
                'parameters': {
                    'type': 'object',
                    'properties': {},
                },
            },
        }],
        tool_choice='auto',
    )


def _generate_unknown_tool(preprocessed, **kwargs):
    async def generate():
        yield SimpleNamespace(
            response=(
                '<tool_call>img_gen'
                '<arg_key>prompt</arg_key><arg_value>edit image</arg_value>'
                '</tool_call>'
            ),
            token_ids=[1],
            input_token_len=4,
            generate_token_len=1,
            finish_reason='stop',
            logprobs=None,
            cached_tokens=0,
            routed_experts=None,
            cache_block_ids=None,
        )

    return generate()


async def _collect_stream(response) -> str:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return ''.join(chunks)


def _terminal_stream_choice(text: str) -> dict:
    for line in text.splitlines():
        if not line.startswith('data: ') or line == 'data: [DONE]':
            continue
        payload = json.loads(line.removeprefix('data: '))
        for choice in payload['choices']:
            if choice.get('finish_reason') is not None:
                return choice
    raise AssertionError('No terminal stream choice found.')


@pytest.mark.parametrize('stream', [False, True])
def test_handler_reports_filtered_unknown_tool_as_parse_error(
        stream, chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls

    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = ToolParserManager.get('glm47')
        context.response_parser_cls = parser_cls
        context.async_engine.generate = _generate_unknown_tool

        response = asyncio.run(endpoint(_request(stream=stream), fake_raw_request))

        if stream:
            choice = _terminal_stream_choice(asyncio.run(_collect_stream(response)))
            assert choice['finish_reason'] == 'parse_error'
            assert choice['delta'].get('tool_calls') is None
        else:
            choice = response['choices'][0]
            assert choice['finish_reason'] == 'parse_error'
            assert choice['message']['tool_calls'] is None
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls
