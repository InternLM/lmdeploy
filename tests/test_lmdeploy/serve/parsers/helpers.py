# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from lmdeploy.serve.openai.protocol import DeltaMessage, ToolCall

if TYPE_CHECKING:
    from lmdeploy.serve.parsers.tool_parser import ToolParser


def first_stream_delta(
    deltas: list[tuple[DeltaMessage, bool]],
) -> tuple[DeltaMessage | None, bool]:
    """Return the first delta from ``ResponseParser.stream_chunk``.

    Returns ``(None, False)`` when ``deltas`` is empty.
    """
    if not deltas:
        return None, False
    return deltas[0]


def final_tool_calls(parser: ToolParser, payload: str) -> list[ToolCall]:
    """Feed one final payload through the production tool-parser interface."""
    parser.begin_tool_block()
    deltas = []
    parser.feed_tool_block(payload, deltas, final=True)
    return parser.build_tool_calls(deltas)


def final_tool_call(parser: ToolParser, payload: str) -> ToolCall:
    """Return the first complete call emitted from one final payload."""
    return final_tool_calls(parser, payload)[0]
