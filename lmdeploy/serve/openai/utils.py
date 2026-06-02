# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, TypeVar

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)

_ToolCallT = TypeVar('_ToolCallT')
_ChatCompletionResponseChoiceT = TypeVar('_ChatCompletionResponseChoiceT', ChatCompletionResponseChoice,
                                         ChatCompletionResponseStreamChoice)


def filter_parallel_tool_calls(tool_calls: list[_ToolCallT] | None,
                               parallel_tool_calls: bool | None) -> list[_ToolCallT] | None:
    """Filter to the first tool call only when parallel_tool_calls is false."""

    if parallel_tool_calls or not tool_calls:
        return tool_calls
    return tool_calls[:1]


def filter_parallel_tool_call_deltas(tool_calls: list[Any] | None,
                                     parallel_tool_calls: bool | None) -> list[Any] | None:
    """Filter to index zero tool deltas only when parallel_tool_calls is
    false."""

    if parallel_tool_calls or not tool_calls:
        return tool_calls
    return [tool_call for tool_call in tool_calls if tool_call.index == 0]


def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT,
    request: ChatCompletionRequest,
) -> _ChatCompletionResponseChoiceT:
    """Filter to the first tool call only when parallel_tool_calls is false."""

    if request.parallel_tool_calls:
        return choice

    if isinstance(choice, ChatCompletionResponseChoice) and choice.message.tool_calls:
        choice.message.tool_calls = filter_parallel_tool_calls(
            choice.message.tool_calls, request.parallel_tool_calls)
    elif isinstance(choice, ChatCompletionResponseStreamChoice) and choice.delta.tool_calls:
        choice.delta.tool_calls = filter_parallel_tool_call_deltas(
            choice.delta.tool_calls, request.parallel_tool_calls)

    return choice
