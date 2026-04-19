# Copyright (c) OpenMMLab. All rights reserved.
"""Streaming helpers for Anthropic-compatible responses."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from .adapter import map_finish_reason


def _format_sse(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


async def stream_messages_response(result_generator,
                                   *,
                                   request_id: str,
                                   model: str) -> AsyncGenerator[str, None]:
    """Convert LMDeploy generation stream to Anthropic SSE events."""

    yield _format_sse(
        'message_start',
        {
            'type': 'message_start',
            'message': {
                'id': request_id,
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': model,
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'output_tokens': 0,
                },
            },
        },
    )
    yield _format_sse(
        'content_block_start',
        {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {
                'type': 'text',
                'text': '',
            },
        },
    )

    final_res = None
    input_tokens = 0
    async for res in result_generator:
        final_res = res
        input_tokens = res.input_token_len
        text = res.response or ''
        if text:
            yield _format_sse(
                'content_block_delta',
                {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {
                        'type': 'text_delta',
                        'text': text,
                    },
                },
            )

    yield _format_sse('content_block_stop', {'type': 'content_block_stop', 'index': 0})

    output_tokens = 0 if final_res is None else final_res.generate_token_len
    stop_reason = map_finish_reason(None if final_res is None else final_res.finish_reason)
    yield _format_sse(
        'message_delta',
        {
            'type': 'message_delta',
            'delta': {
                'stop_reason': stop_reason,
                'stop_sequence': None,
            },
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            },
        },
    )
    yield _format_sse('message_stop', {'type': 'message_stop'})
