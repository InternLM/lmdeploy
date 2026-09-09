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


def flatten_stream_deltas(deltas: list[tuple[DeltaMessage, bool]]) -> list[dict]:
    """Flatten response-parser deltas into values relevant to parser tests."""
    events = []
    for delta, tool_emitted in deltas:
        if delta is None:
            continue
        if delta.reasoning_content is not None:
            events.append({'reasoning_content': delta.reasoning_content, 'tool_emitted': tool_emitted})
        if delta.content is not None:
            events.append({'content': delta.content, 'tool_emitted': tool_emitted})
        for call in delta.tool_calls or []:
            events.append({
                'tool_emitted': tool_emitted,
                'type': call.type,
                'name': call.function.name if call.function else None,
                'arguments': call.function.arguments if call.function else None,
            })
    return events


def feed_tool_payload(parser: ToolParser, pending: str, *, final: bool) -> tuple[str, list]:
    """Consume as much buffered tool payload as the parser can decide."""
    deltas = []
    while pending and not parser.block_closed:
        consumed = parser.feed_tool_block(pending, deltas, final=final)
        if not consumed:
            break
        pending = pending[consumed:]
    return pending, deltas


def feed_tool_chunks(parser: ToolParser, chunks: list[str]) -> tuple[str, list, int]:
    """Feed chunks while retaining undecidable suffixes between calls."""
    parser.begin_tool_block()
    pending = ''
    deltas = []
    max_pending = 0
    for chunk in chunks:
        pending, chunk_deltas = feed_tool_payload(parser, pending + chunk, final=False)
        deltas.extend(chunk_deltas)
        max_pending = max(max_pending, len(pending))
    return pending, deltas, max_pending


def tool_arguments(deltas: list) -> str:
    """Join argument fragments from tool-call deltas."""
    return ''.join(
        delta.function.arguments or ''
        for delta in deltas
        if delta.function is not None
    )


def stream_tool_arguments_by_chunk(parser: ToolParser, chunks: list[str]) -> tuple[str, list[str]]:
    """Return reconstructed arguments and fragments emitted for each chunk."""
    parser.begin_tool_block()
    pending = ''
    argument_fragments = []
    per_chunk = []
    last_index = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        final = index == last_index
        text = pending + chunk
        if final:
            text += parser.get_tool_close_tag() or ''
        pending, deltas = feed_tool_payload(parser, text, final=final)
        chunk_fragments = [
            call.function.arguments for call in deltas
            if call.function and call.function.arguments is not None
        ]
        argument_fragments.extend(chunk_fragments)
        per_chunk.append(''.join(chunk_fragments))
    return ''.join(argument_fragments), per_chunk


def stream_tool_arguments(parser: ToolParser, chunks: list[str]) -> str:
    """Return arguments reconstructed from a complete sequence of chunks."""
    arguments, _ = stream_tool_arguments_by_chunk(parser, chunks)
    return arguments


def final_tool_calls(parser: ToolParser, payload: str) -> list[ToolCall]:
    """Feed one final payload through the production tool-parser interface."""
    parser.begin_tool_block()
    deltas = []
    parser.feed_tool_block(payload, deltas, final=True)
    return parser.build_tool_calls(deltas)


def final_tool_call(parser: ToolParser, payload: str) -> ToolCall:
    """Return the first complete call emitted from one final payload."""
    return final_tool_calls(parser, payload)[0]
