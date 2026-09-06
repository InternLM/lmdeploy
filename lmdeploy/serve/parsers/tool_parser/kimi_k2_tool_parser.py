# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from lmdeploy.serve.openai.protocol import DeltaToolCall, FunctionCall, ToolCall

from .json_value_scanner import JsonValueScanner
from .tool_parser import ToolParser, ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@ToolParserManager.register_module(['kimi_k2', 'kimi-k2'])
class KimiK2ToolParser(ToolParser):
    """Tool parser for the Kimi K2 tool-call section protocol."""

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

    def parse_tool_call_complete(self, payload: str) -> list[ToolCall] | None:
        calls, _ = self._parse_complete_calls(payload, 0, section_close_tag=None)
        return calls or None

    def parse_tool_block(self, text: str, start: int, tool_calls: list[ToolCall]) -> int:
        calls, end = self._parse_complete_calls(text, start, section_close_tag=self.get_tool_close_tag())
        tool_calls.extend(calls)
        return end

    def _parse_complete_calls(
        self,
        text: str,
        start: int,
        *,
        section_close_tag: str | None,
    ) -> tuple[list[ToolCall], int]:
        calls: list[ToolCall] = []
        pos = start
        while pos < len(text):
            section_at = text.find(section_close_tag, pos) if section_close_tag is not None else -1
            call_at = text.find(self.call_begin, pos)
            if section_at >= 0 and (call_at < 0 or section_at < call_at):
                return calls, section_at + len(section_close_tag)
            if call_at < 0:
                return calls, len(text)

            id_start = call_at + len(self.call_begin)
            argument_at = text.find(self.argument_begin, id_start)
            nested_call = text.find(self.call_begin, id_start)
            if argument_at < 0:
                return calls, len(text)
            if nested_call >= 0 and nested_call < argument_at:
                pos = nested_call
                continue

            raw_id = text[id_start:argument_at].strip()
            args_start = argument_at + len(self.argument_begin)
            if text.startswith(self.call_end, args_start):
                arguments = '{}'
                call_end_at = args_start
            else:
                args_end, complete = self._scan_argument(text, args_start, self.call_end)
                call_end_at = text.find(self.call_end, args_end)
                if call_end_at < 0:
                    call_end_at = text.find(self.call_end, args_start)
                if call_end_at < 0:
                    arguments = text[args_start:args_end] or '{}'
                    calls.append(self._make_tool_call(raw_id, arguments))
                    return calls, len(text)
                arguments = text[args_start:(args_end if complete else call_end_at)] or '{}'

            calls.append(self._make_tool_call(raw_id, arguments))
            pos = call_end_at + len(self.call_end)

        return calls, pos

    @staticmethod
    def _scan_argument(text: str, start: int, marker: str) -> tuple[int, bool]:
        scanner = JsonValueScanner()
        marker_at = text.find(marker, start)
        if marker_at < 0:
            end = scanner.feed(text, start)
            return end, scanner.complete

        end = scanner.feed(text, start, marker_at)
        if scanner.complete or (end == marker_at and scanner.finish_scalar()):
            return end, True
        if not scanner.in_string:
            return end, False
        end = scanner.feed(text, end)
        return end, scanner.complete

    @staticmethod
    def _make_tool_call(raw_id: str, arguments: str) -> ToolCall:
        return ToolCall(
            id=raw_id,
            function=FunctionCall(
                name=KimiK2ToolParser._resolve_function_name(raw_id),
                arguments=arguments,
            ),
        )

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
