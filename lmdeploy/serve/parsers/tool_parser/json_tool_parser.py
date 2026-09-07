# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import ClassVar

from lmdeploy.serve.openai.protocol import DeltaToolCall

from .json_value_scanner import JsonValueScanner
from .tool_parser import ToolParser


class JsonToolParser(ToolParser):
    """Incrementally extract a JSON tool-call envelope.

    Subclasses supply outer protocol tags. Complete parsing is inherited from
    :class:`ToolParser` and therefore uses this incremental state machine too.
    """

    argument_field: ClassVar[str] = 'arguments'

    def __init__(self) -> None:
        """Initialize the JSON envelope state machine."""
        super().__init__()
        self._phase = 'payload_start'
        self._json_key: str | None = None
        self._arguments_seen = False
        self._value_scanner = JsonValueScanner()

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        """Return no outer opening tag for the bare JSON base parser."""
        return None

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        """Return no outer closing tag for the bare JSON base parser."""
        return None

    def begin_tool_block(self) -> None:
        """Begin a logical call and reset its JSON envelope state."""
        super().begin_tool_block()
        self._begin_call()
        self._phase = 'payload_start'
        self._json_key = None
        self._arguments_seen = False
        self._value_scanner.reset()

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        """Consume the stable prefix of an incremental JSON envelope.

        The configured argument field is emitted verbatim as its value becomes
        available. Other fields are scanned only to locate the next envelope
        field. The value scanner carries incomplete value state across calls;
        incomplete envelope syntax and possible closing-marker suffixes remain
        in the input buffer.

        Args:
            text: Buffered payload text after the outer opening marker.
            deltas: Destination for parsed function-name and argument deltas.
            final: Whether no more generated text will follow.

        Returns:
            Number of leading characters that may be discarded from ``text``.
        """
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
                            arguments=text[start:pos],
                        )
                    continue

                if scan_limit < size and pos == scan_limit and self._value_scanner.in_string:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if self._phase == 'arguments_value' and pos > start:
                        self._emit_delta(
                            deltas,
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
        """Mark the envelope complete and emit default empty arguments."""
        if not self._arguments_seen:
            self._emit_delta(
                deltas,
                arguments='{}',
            )
        self._payload_closed = True
        self._phase = 'done'

    @staticmethod
    def _read_string(text: str, start: int) -> tuple[str, int]:
        """Decode a quoted JSON string and return its value and end offset.

        A completed but malformed string falls back to its raw inner text.
        ``('', -1)`` indicates that the closing quote is not buffered yet.
        """
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
        """Return the first position at or after ``pos`` that is not JSON
        whitespace."""
        while pos < len(text) and text[pos] in ' \t\r\n':
            pos += 1
        return pos
