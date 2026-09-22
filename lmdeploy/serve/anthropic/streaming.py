# Copyright (c) OpenMMLab. All rights reserved.
"""Streaming helpers for Anthropic-compatible responses."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.protocol import ChatCompletionRequest

from .adapter import map_finish_reason
from .errors import anthropic_error_from_request
from .protocol import (
    LOCAL_THINKING_SIGNATURE,
    AnthropicStreamEvent,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorEvent,
    InputJsonDelta,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageStartEvent,
    MessageStartMessage,
    MessageStopEvent,
    MessageUsage,
    SignatureDelta,
    StreamTextBlock,
    StreamThinkingBlock,
    StreamToolUseBlock,
    TextDelta,
    ThinkingDelta,
)

_OPTIONAL_EXTENSION_FIELDS = ('output_ids', 'output_token_logprobs', 'routed_experts')


def _format_sse(data: AnthropicStreamEvent) -> str:
    payload = data.model_dump(mode='json')
    for key in _OPTIONAL_EXTENSION_FIELDS:
        if payload.get(key) is None:
            payload.pop(key, None)
    return f'event: {data.type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


async def _results_or_error(result_generator):
    try:
        async for result in result_generator:
            yield result
    except RequestError as error:
        yield error


@dataclass
class _StreamBlockState:
    next_block_index: int = 0
    current_block: dict[str, Any] | None = None
    tool_blocks: dict[int, dict[str, Any]] = field(default_factory=dict)


def _close_current_block(state: _StreamBlockState) -> list[str]:
    block = state.current_block
    if block is None:
        return []

    block_index = block['block_index']
    events: list[str] = []
    if block['kind'] == 'thinking':
        events.append(
            _format_sse(
                ContentBlockDeltaEvent(
                    index=block_index,
                    delta=SignatureDelta(signature=LOCAL_THINKING_SIGNATURE),
                )))
    events.append(_format_sse(ContentBlockStopEvent(index=block_index)))
    state.current_block = None
    return events


def _start_text_or_thinking(state: _StreamBlockState, kind: str) -> list[str]:
    events: list[str] = []
    if state.current_block is not None and state.current_block['kind'] == kind:
        return events
    events.extend(_close_current_block(state))
    block_index = state.next_block_index
    state.next_block_index += 1
    if kind == 'text':
        content_block = StreamTextBlock(text='')
    else:
        content_block = StreamThinkingBlock(thinking='')
    state.current_block = dict(kind=kind, block_index=block_index)
    events.append(
        _format_sse(
            ContentBlockStartEvent(index=block_index, content_block=content_block),
        ))
    return events


def _start_tool_block(state: _StreamBlockState, tool_delta) -> list[str]:
    events: list[str] = []
    tool_index = tool_delta.index
    block = state.tool_blocks.get(tool_index)
    current_block = state.current_block
    same_block = (
        current_block is not None
        and current_block['kind'] == 'tool_use'
        and current_block['tool_index'] == tool_index
        and block is not None
        and current_block['block_index'] == block['block_index'])
    if not same_block:
        events.extend(_close_current_block(state))
    if block is None:
        function_delta = tool_delta.function
        tool_name = '' if function_delta is None else function_delta.name or ''
        block_index = state.next_block_index
        state.next_block_index += 1
        block = dict(
            tool_index=tool_index,
            block_index=block_index,
            tool_use_id=tool_delta.id,
            name=tool_name,
        )
        state.tool_blocks[tool_index] = block
        events.append(
            _format_sse(
                ContentBlockStartEvent(
                    index=block_index,
                    content_block=StreamToolUseBlock(
                        id=tool_delta.id,
                        name=tool_name,
                        input={},
                    ),
                )
            ))
    state.current_block = dict(
        kind='tool_use',
        block_index=block['block_index'],
        tool_index=tool_index,
    )
    return events


def _emit_text_delta(state: _StreamBlockState,
                     text: str,
                     thinking: bool,
                     output_ids: list[int] | None = None,
                     output_token_logprobs: list[tuple[float, int]] | None = None) -> str:
    block_index = state.current_block['block_index']
    delta = ThinkingDelta(thinking=text) if thinking else TextDelta(text=text)
    return _format_sse(
        ContentBlockDeltaEvent(
            index=block_index,
            delta=delta,
            output_ids=output_ids,
            output_token_logprobs=output_token_logprobs,
        ))


def _emit_metadata_delta(state: _StreamBlockState,
                         output_ids: list[int] | None,
                         output_token_logprobs: list[tuple[float, int]] | None = None) -> str:
    current_block = state.current_block
    kind = current_block['kind']
    if kind == 'tool_use':
        return _format_sse(
            ContentBlockDeltaEvent(
                index=current_block['block_index'],
                delta=InputJsonDelta(partial_json=''),
                output_ids=output_ids,
                output_token_logprobs=output_token_logprobs,
            ))
    return _emit_text_delta(
        state,
        '',
        thinking=kind == 'thinking',
        output_ids=output_ids,
        output_token_logprobs=output_token_logprobs,
    )


async def stream_messages_response(parsed_stream,
                                   *,
                                   request_id: str,
                                   request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Convert LMDeploy generation stream to Anthropic SSE events."""
    return_token_ids = bool(request.return_token_ids)
    return_routed_experts = bool(request.return_routed_experts)
    return_logprob = bool(request.return_logprob)

    # Anthropic's message_start event carries usage while its content is
    # still empty. Buffer one parsed chunk to populate usage, then stream
    # that same chunk through the normal content-block path below.
    result_iter = _results_or_error(parsed_stream).__aiter__()
    try:
        first_chunk = await anext(result_iter)
    except StopAsyncIteration:
        first_chunk = None

    if isinstance(first_chunk, RequestError):
        yield _format_sse(
            ErrorEvent(error=anthropic_error_from_request(first_chunk)))
        return

    start_usage = MessageUsage(
        input_tokens=0 if first_chunk is None else first_chunk.input_token_len,
        output_tokens=0 if first_chunk is None else first_chunk.generate_token_len,
    )
    yield _format_sse(
        MessageStartEvent(
            message=MessageStartMessage(
                id=request_id,
                model=request.model,
                usage=start_usage,
            ),
        )
    )
    final_chunk = None
    block_state = _StreamBlockState()

    async def _results():
        if first_chunk is not None:
            yield first_chunk
        async for chunk in result_iter:
            yield chunk

    async for chunk in _results():
        if isinstance(chunk, RequestError):
            for event in _close_current_block(block_state):
                yield event
            yield _format_sse(
                ErrorEvent(error=anthropic_error_from_request(chunk)))
            return
        final_chunk = chunk
        delta_message = chunk.delta_message
        delta_token_ids = chunk.token_ids
        delta_logprobs = None
        if return_logprob and chunk.logprobs and delta_token_ids:
            delta_logprobs = [
                (tok_logprobs[tok], tok)
                for tok, tok_logprobs in zip(delta_token_ids, chunk.logprobs)
            ]

        if delta_message is None:
            continue

        delta_output_ids = delta_token_ids if return_token_ids and chunk.is_last_delta else None
        delta_output_logprobs = delta_logprobs if chunk.is_last_delta else None
        metadata_emitted = delta_output_ids is None and delta_output_logprobs is None
        has_content = bool(delta_message.content)
        has_tools = bool(delta_message.tool_calls)

        if delta_message.reasoning_content:
            for event in _start_text_or_thinking(block_state, 'thinking'):
                yield event
            reasoning_output_ids = None if has_content or has_tools else delta_output_ids
            reasoning_output_logprobs = None if has_content or has_tools else delta_output_logprobs
            yield _emit_text_delta(
                block_state,
                delta_message.reasoning_content,
                thinking=True,
                output_ids=reasoning_output_ids,
                output_token_logprobs=reasoning_output_logprobs,
            )
            metadata_emitted = metadata_emitted or (
                reasoning_output_ids is not None or reasoning_output_logprobs is not None)

        if delta_message.content:
            for event in _start_text_or_thinking(block_state, 'text'):
                yield event
            content_output_ids = None if has_tools else delta_output_ids
            content_output_logprobs = None if has_tools else delta_output_logprobs
            yield _emit_text_delta(
                block_state,
                delta_message.content,
                thinking=False,
                output_ids=content_output_ids,
                output_token_logprobs=content_output_logprobs,
            )
            metadata_emitted = metadata_emitted or (
                content_output_ids is not None or content_output_logprobs is not None)

        if delta_message.tool_calls:
            for tool_index, tool_delta in enumerate(delta_message.tool_calls):
                for event in _start_tool_block(block_state, tool_delta):
                    yield event
                function_delta = tool_delta.function
                if function_delta is None:
                    continue
                partial_json = function_delta.arguments or ''
                if partial_json:
                    output_ids = None
                    output_token_logprobs = None
                    if tool_index == len(delta_message.tool_calls) - 1:
                        output_ids = delta_output_ids
                        output_token_logprobs = delta_output_logprobs
                    yield _format_sse(
                        ContentBlockDeltaEvent(
                            index=block_state.current_block['block_index'],
                            delta=InputJsonDelta(partial_json=partial_json),
                            output_ids=output_ids,
                            output_token_logprobs=output_token_logprobs,
                        ))
                    metadata_emitted = metadata_emitted or (
                        output_ids is not None or output_token_logprobs is not None)

        if not metadata_emitted:
            if block_state.current_block is None:
                for event in _start_text_or_thinking(block_state, 'text'):
                    yield event
            yield _emit_metadata_delta(block_state, delta_output_ids, delta_output_logprobs)

    for event in _close_current_block(block_state):
        yield event

    output_tokens = 0 if final_chunk is None else final_chunk.generate_token_len
    stop_reason = map_finish_reason(None if final_chunk is None else final_chunk.finish_reason)
    routed_experts = final_chunk.routed_experts if return_routed_experts and final_chunk is not None else None
    yield _format_sse(
        MessageDeltaEvent(
            delta=MessageDelta(stop_reason=stop_reason, stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=output_tokens),
            routed_experts=routed_experts,
        ))
    yield _format_sse(MessageStopEvent())
