# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import ClassVar

from lmdeploy.serve.openai.protocol import DeltaToolCall, FunctionCall, ToolCall

from .json_value_scanner import JsonValueScanner
from .tool_parser import ToolParser


class JsonToolParser(ToolParser):
    """Incrementally extract a JSON tool-call envelope."""

    argument_field: ClassVar[str] = 'arguments'
    def __init__(self) -> None:
        super().__init__()
        self._phase = 'payload_start'
        self._json_key: str | None = None
        self._arguments_seen = False
        self._value_scanner = JsonValueScanner()

    def begin_tool_block(self) -> None:
        super().begin_tool_block()
        self._phase = 'payload_start'
        self._json_key = None
        self._arguments_seen = False
        self._value_scanner.reset()

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        pos = 0
        size = len(text)
        close_tag = self.get_tool_close_tag()

        while pos < size:
            if self._phase not in ('arguments_value', 'skip_value') and close_tag and text.startswith(close_tag, pos):
                self._finish_envelope(deltas)
                break

            if self._phase == 'payload_start':
                pos = self._skip_ws(text, pos)
                if pos == size:
                    break
                if text[pos] != '{':
                    pos += 1
                    continue
                pos += 1
                self._phase = 'key_or_end'
                continue

            if self._phase in ('key_or_end', 'key'):
                while pos < size and (text[pos].isspace() or text[pos] == ','):
                    pos += 1
                if pos == size:
                    break
                if text[pos] == '}':
                    pos += 1
                    self._finish_envelope(deltas)
                    continue
                if text[pos] != '"':
                    pos += 1
                    continue
                key, end = self._read_string(text, pos)
                if end < 0:
                    break
                self._json_key = key
                pos = end
                self._phase = 'colon'
                continue

            if self._phase == 'colon':
                pos = self._skip_ws(text, pos)
                if pos == size:
                    break
                if text[pos] != ':':
                    pos += 1
                    continue
                pos += 1
                self._phase = 'value_start'
                continue

            if self._phase == 'value_start':
                pos = self._skip_ws(text, pos)
                if pos == size:
                    break
                if self._json_key == 'name' and text[pos] == '"':
                    self._phase = 'name_value'
                else:
                    self._value_scanner.reset()
                    if self._json_key == self.argument_field:
                        self._arguments_seen = True
                        self._phase = 'arguments_value'
                    else:
                        self._phase = 'skip_value'
                continue

            if self._phase == 'name_value':
                name, end = self._read_string(text, pos)
                if end < 0:
                    break
                if not self._name_emitted:
                    self._emit_delta(
                        deltas,
                        index=self._active_tool_index,
                        tool_call_id=self._active_tool_call_id,
                        name=name,
                    )
                    self._name_emitted = True
                pos = end
                self._json_key = None
                self._phase = 'after_value'
                continue

            if self._phase in ('arguments_value', 'skip_value'):
                if self._value_scanner.in_string and text.find('"', pos) < 0:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if self._phase == 'arguments_value' and pos > start:
                        self._emit_delta(
                            deltas,
                            index=self._active_tool_index,
                            tool_call_id=self._active_tool_call_id,
                            arguments=text[start:pos],
                        )
                    break

                marker_at = text.find(close_tag, pos) if close_tag else -1
                if marker_at >= 0:
                    scan_limit = marker_at
                elif self.tool_close_prefixes:
                    scan_limit = self._stable_prefix_end(text, self.tool_close_prefixes, pos)
                else:
                    scan_limit = size
                start = pos
                pos = self._value_scanner.feed(text, pos, scan_limit)
                if self._phase == 'arguments_value' and pos > start:
                    self._emit_delta(
                        deltas,
                        index=self._active_tool_index,
                        tool_call_id=self._active_tool_call_id,
                        arguments=text[start:pos],
                    )
                if self._value_scanner.complete:
                    self._json_key = None
                    self._phase = 'after_value'
                    continue

                if marker_at >= 0 and pos == marker_at:
                    if not self._value_scanner.in_string:
                        self._value_scanner.finish()
                        self._finish_envelope(deltas)
                        continue
                    marker_end = marker_at + len(close_tag)
                    start = pos
                    pos = self._value_scanner.feed(text, pos, marker_end)
                    if self._phase == 'arguments_value':
                        self._emit_delta(
                            deltas,
                            index=self._active_tool_index,
                            tool_call_id=self._active_tool_call_id,
                            arguments=text[start:pos],
                        )
                    continue

                if scan_limit < size and pos == scan_limit and self._value_scanner.in_string:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if self._phase == 'arguments_value' and pos > start:
                        self._emit_delta(
                            deltas,
                            index=self._active_tool_index,
                            tool_call_id=self._active_tool_call_id,
                            arguments=text[start:pos],
                        )
                    if self._value_scanner.complete:
                        self._json_key = None
                        self._phase = 'after_value'
                        continue
                break

            if self._phase == 'after_value':
                pos = self._skip_ws(text, pos)
                if pos == size:
                    break
                if text[pos] == ',':
                    pos += 1
                    self._phase = 'key'
                elif text[pos] == '}':
                    pos += 1
                    self._finish_envelope(deltas)
                else:
                    pos += 1
                continue

            if self._phase == 'done':
                pos = self._skip_ws(text, pos)
                break

        if final and self._phase in ('arguments_value', 'skip_value'):
            self._value_scanner.finish()
        return pos

    def _finish_envelope(self, deltas: list[DeltaToolCall]) -> None:
        if not self._arguments_seen:
            self._emit_delta(
                deltas,
                index=self._active_tool_index,
                tool_call_id=self._active_tool_call_id,
                arguments='{}',
            )
        self._payload_closed = True
        self._phase = 'done'

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        parsed = self._parse_complete_envelope(payload)
        if parsed is None:
            return None
        name, arguments, _ = parsed
        return ToolCall(function=FunctionCall(name=name, arguments=arguments))

    def parse_tool_block(self, text: str, start: int, tool_calls: list[ToolCall]) -> int:
        close_tag = self.get_tool_close_tag()
        parsed = self._parse_complete_envelope(text[start:], close_tag=close_tag)
        if parsed is None:
            close_at = text.find(close_tag, start) if close_tag is not None else -1
            return close_at + len(close_tag) if close_at >= 0 else len(text)

        name, arguments, payload_end = parsed
        tool_calls.append(ToolCall(function=FunctionCall(name=name, arguments=arguments)))
        if close_tag is None:
            return start + payload_end
        close_at = text.find(close_tag, start + payload_end)
        return close_at + len(close_tag) if close_at >= 0 else len(text)

    def _parse_complete_envelope(self, payload: str, *, close_tag: str | None = None) -> tuple[str, str, int] | None:
        """Extract source fields without validating their JSON values."""
        pos = self._skip_ws(payload, 0)
        if pos >= len(payload) or payload[pos] != '{':
            return None
        pos += 1
        name: str | None = None
        arguments: list[str] = []
        arguments_seen = False

        while pos < len(payload):
            pos = self._skip_ws(payload, pos)
            if pos == len(payload):
                break
            if close_tag and payload.startswith(close_tag, pos):
                break
            if payload[pos] == '}':
                pos += 1
                break
            if payload[pos] != '"':
                pos += 1
                continue
            key, key_end = self._read_string(payload, pos)
            if key_end < 0:
                break
            pos = self._skip_ws(payload, key_end)
            if pos >= len(payload) or payload[pos] != ':':
                continue
            value_start = self._skip_ws(payload, pos + 1)
            if key == self.argument_field:
                arguments_seen = True
            if value_start >= len(payload) or (close_tag and payload.startswith(close_tag, value_start)):
                pos = value_start
                break

            value_end = self._scan_complete_value(payload, value_start, close_tag)
            if key == 'name' and name is None and payload[value_start] == '"':
                value, string_end = self._read_string(payload, value_start)
                if string_end >= 0 and string_end <= value_end:
                    name = value
            elif key == self.argument_field:
                arguments.append(payload[value_start:value_end])

            pos = value_end
            if close_tag and payload.startswith(close_tag, pos):
                break

        if name is None:
            return None
        return name, ''.join(arguments) if arguments_seen else '{}', pos

    @staticmethod
    def _scan_complete_value(text: str, start: int, close_tag: str | None) -> int:
        scanner = JsonValueScanner()
        pos = start
        while pos < len(text):
            marker_at = text.find(close_tag, pos) if close_tag else -1
            scan_end = marker_at if marker_at >= 0 else len(text)
            pos = scanner.feed(text, pos, scan_end)
            if scanner.complete or marker_at < 0:
                return pos
            if not scanner.in_string:
                scanner.finish()
                return pos
            pos = scanner.feed(text, pos, marker_at + len(close_tag))
        return pos

    @staticmethod
    def _read_string(text: str, start: int) -> tuple[str, int]:
        escaped = False
        pos = start + 1
        while pos < len(text):
            char = text[pos]
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                end = pos + 1
                try:
                    value = json.loads(text[start:end])
                except json.JSONDecodeError:
                    value = text[start + 1:pos]
                return value if isinstance(value, str) else str(value), end
            pos += 1
        return '', -1

    @staticmethod
    def _skip_ws(text: str, pos: int) -> int:
        while pos < len(text) and text[pos] in ' \t\r\n':
            pos += 1
        return pos
