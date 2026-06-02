# Copyright (c) OpenMMLab. All rights reserved.
"""Responses final response construction helpers."""

from __future__ import annotations

from typing import Any, Literal

import shortuuid

from lmdeploy.serve.openai.responses.protocol import (
    ResponseIncompleteDetails,
    ResponseOutputFunctionCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
)


def _response_metadata_kwargs(request: ResponsesRequest) -> dict[str, Any]:
    return dict(
        instructions=request.instructions,
        metadata=request.metadata,
        max_output_tokens=request.max_output_tokens,
        max_tool_calls=request.max_tool_calls,
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        prompt=request.prompt,
        prompt_cache_key=request.prompt_cache_key,
        prompt_cache_retention=request.prompt_cache_retention,
        reasoning=request.reasoning,
        safety_identifier=request.safety_identifier,
        service_tier=request.service_tier,
        background=bool(request.background),
        conversation=request.conversation,
        store=False,
        temperature=request.temperature,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_logprobs=request.top_logprobs,
        top_p=request.top_p,
        truncation=request.truncation,
        user=request.user,
    )


def _filter_parallel_tool_calls(request: ResponsesRequest, tool_calls: list[Any] | None) -> list[Any] | None:
    if request.parallel_tool_calls is not False or not tool_calls:
        return tool_calls
    return tool_calls[:1]


def _response_status_from_finish_reason(
        finish_reason: str | None) -> Literal['completed', 'incomplete', 'failed', 'cancelled']:
    if finish_reason == 'length':
        return 'incomplete'
    if finish_reason == 'abort':
        return 'cancelled'
    if finish_reason == 'error':
        return 'failed'
    return 'completed'


def _response_error_from_finish_reason(finish_reason: str | None) -> dict[str, Any] | None:
    if finish_reason == 'error':
        return dict(code='server_error', message='Response generation failed.')
    if finish_reason == 'abort':
        return dict(code='server_error', message='Response generation was cancelled.')
    return None


def _make_response(*,
                   request: ResponsesRequest,
                   model_name: str,
                   created_time: int,
                   text: str | None,
                   tool_calls: list[Any] | None = None,
                   input_tokens: int,
                   output_tokens: int,
                   finish_reason: str | None,
                   message_id: str | None = None) -> ResponsesResponse:
    text = text or ''
    tool_calls = _filter_parallel_tool_calls(request, tool_calls)
    status = _response_status_from_finish_reason(finish_reason)
    message_status = 'incomplete' if status == 'incomplete' else 'completed'
    incomplete_details = None
    if status == 'incomplete':
        incomplete_details = ResponseIncompleteDetails(reason='max_output_tokens')
    output: list[ResponseOutputMessage | ResponseOutputFunctionCall] = []
    if text or not tool_calls:
        output.append(
            ResponseOutputMessage(
                id=message_id or f'msg_{shortuuid.random()}',
                status=message_status,
                content=[ResponseOutputText(text=text)],
            ))
    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, ResponseOutputFunctionCall):
                output.append(tool_call)
                continue
            function = getattr(tool_call, 'function', None)
            if function is None:
                continue
            call_id = getattr(tool_call, 'id', None) or f'call_{shortuuid.random()}'
            output.append(
                ResponseOutputFunctionCall(
                    id=call_id,
                    call_id=call_id,
                    name=function.name,
                    arguments=function.arguments or '',
                ))
    return ResponsesResponse(
        id=request.request_id,
        created_at=created_time,
        model=model_name,
        status=status,
        output=output,
        output_text=text,
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        error=_response_error_from_finish_reason(finish_reason),
        incomplete_details=incomplete_details,
        **_response_metadata_kwargs(request),
    )
