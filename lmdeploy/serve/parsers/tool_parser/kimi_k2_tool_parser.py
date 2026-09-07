# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from lmdeploy.serve.openai.protocol import DeltaToolCall

from .json_value_scanner import JsonValueScanner
from .tool_parser import ToolParser, ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@ToolParserManager.register_module(['kimi_k2', 'kimi-k2'])
class KimiK2ToolParser(ToolParser):
    """Incrementally parse the Kimi K2 tool-call section protocol.

    Complete responses use this same consumer through the response parser.
    """

    structural_tag_model = 'kimi'
    section_begin = '<|tool_calls_section_begin|>'
    section_end = '<|tool_calls_section_end|>'
    call_begin = '<|tool_call_begin|>'
    argument_begin = '<|tool_call_argument_begin|>'
    call_end = '<|tool_call_end|>'

    def __init__(self) -> None:
        super().__init__()
        self._phase = 'call_start'
        self._arguments_emitted = False
        self._value_scanner = JsonValueScanner()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request.skip_special_tokens = False
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)

    def begin_tool_block(self) -> None:
        super().begin_tool_block()
        self._phase = 'call_start'
        self._arguments_emitted = False
        self._value_scanner.reset()

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return cls.section_begin

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return cls.section_end

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        del final
        pos = 0
        while pos < len(text):
            if self._phase == 'call_start':
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos == len(text):
                    break
                if text.startswith(self.section_end, pos):
                    self._payload_closed = True
                    break
                if text.startswith(self.call_begin, pos):
                    pos += len(self.call_begin)
                    self._phase = 'call_header'
                    continue
                next_pos = self._next_marker(text, pos, (self.call_begin, self.section_end))
                if next_pos == pos:
                    break
                pos = next_pos
                continue

            if self._phase == 'call_header':
                argument_at = text.find(self.argument_begin, pos)
                if argument_at < 0:
                    break
                raw_id = text[pos:argument_at].strip()
                self._begin_call(raw_id)
                self._emit_delta(
                    deltas,
                    name=self._resolve_function_name(raw_id),
                )
                self._arguments_emitted = False
                self._value_scanner.reset()
                pos = argument_at + len(self.argument_begin)
                self._phase = 'arguments'
                continue

            if self._phase == 'arguments':
                if not self._value_scanner.started and text.startswith(self.call_end, pos):
                    self._emit_arguments(deltas, '{}')
                    self._phase = 'call_end'
                    continue

                if self._value_scanner.in_string and text.find('"', pos) < 0:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if pos > start:
                        self._emit_arguments(deltas, text[start:pos])
                    break

                marker_at = text.find(self.call_end, pos)
                scan_limit = marker_at if marker_at >= 0 else len(text)
                start = pos
                pos = self._value_scanner.feed(text, pos, scan_limit)
                if pos > start:
                    self._emit_arguments(deltas, text[start:pos])
                if marker_at >= 0 and pos == marker_at and self._value_scanner.finish_scalar():
                    self._phase = 'call_end'
                    continue
                if self._value_scanner.complete:
                    self._phase = 'call_end'
                    continue
                if marker_at >= 0:
                    if not self._value_scanner.in_string:
                        self._phase = 'call_end'
                        continue
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if pos > start:
                        self._emit_arguments(deltas, text[start:pos])
                    if self._value_scanner.complete:
                        self._phase = 'call_end'
                        continue
                elif scan_limit < len(text) and self._value_scanner.in_string:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if pos > start:
                        self._emit_arguments(deltas, text[start:pos])
                    if self._value_scanner.complete:
                        self._phase = 'call_end'
                        continue
                break

            if self._phase == 'call_end':
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos == len(text):
                    break
                if not text.startswith(self.call_end, pos):
                    next_pos = self._next_marker(text, pos, (self.call_end, ))
                    if next_pos == pos:
                        break
                    pos = next_pos
                    continue
                if not self._arguments_emitted:
                    self._emit_arguments(deltas, '{}')
                pos += len(self.call_end)
                self._phase = 'call_start'
                continue

        return pos

    def _emit_arguments(self, deltas: list[DeltaToolCall], arguments: str) -> None:
        if not arguments:
            return
        self._emit_delta(
            deltas,
            arguments=arguments,
        )
        self._arguments_emitted = True

    @staticmethod
    def _resolve_function_name(raw_id: str) -> str:
        name = raw_id
        prefix, separator, counter = raw_id.rpartition(':')
        if separator and counter.isdigit():
            name = prefix
        if name.startswith('functions.'):
            name = name[len('functions.'):]
        return name

    @staticmethod
    def _next_marker(text: str, pos: int, markers: tuple[str, ...]) -> int:
        """Return the next complete Kimi marker, whose markers are atomic
        tokens."""
        next_pos = len(text)
        for marker in markers:
            marker_at = text.find(marker, pos)
            if 0 <= marker_at < next_pos:
                next_pos = marker_at
        return next_pos
