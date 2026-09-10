# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import ClassVar, Literal

from lmdeploy.serve.openai.protocol import DeltaToolCall

from .json_value_scanner import JsonValueScanner
from .tool_parser import ToolParser

JsonParsePhase = Literal[
    'payload_start',
    'key',
    'colon',
    'value_start',
    'arguments',
    'skip_value',
    'after_value',
]


class JsonToolParser(ToolParser):
    """Incrementally extract a JSON tool-call envelope.

    Streaming follows this state machine::

        payload_start -- ``{`` --> key
        key ----------- field --> colon
        colon --------- ``:`` --> value_start
        value_start --- name_field -----------------------+
                    +-- argument_field -> arguments ------+--> after_value
                    +-- other ----------> skip_value -----+
        after_value --- ``,`` ----> key
        key / after_value -- ``}`` --> _payload_closed

    Each phase describes the syntax expected at the current cursor:

    - ``payload_start`` searches for the opening ``{`` of the envelope.
    - ``key`` follows the opening brace or a comma and accepts a field name
      or the envelope-closing ``}``.
    - ``colon`` follows a field name and expects its separating ``:``.
    - ``value_start`` reads and emits a complete quoted ``name_field`` value,
      or selects a scanner phase for other fields. Incomplete names remain
      in the caller's input buffer.
    - ``arguments`` scans and emits the configured ``argument_field`` value.
    - ``skip_value`` scans any other field without emitting it.
    - ``after_value`` follows a complete value and expects either ``,`` or
      the envelope-closing ``}``.

    ``_payload_closed`` records completion independently of the syntax phase
    and leaves any outer closing marker for :class:`ToolParser`.

    ``arguments`` is forwarded as it becomes lexically stable;
    ``skip_value`` advances over other envelope fields without emitting them.
    If arguments precede the function name, :class:`ToolParser` retains those
    deltas until the name is known. A recognized outer closing marker may also
    finish an incomplete envelope by setting ``_payload_closed``.

    Subclasses may configure ``name_field`` and ``argument_field`` and supply
    outer protocol tags. Both configured fields must be direct members of the
    JSON envelope; nested function objects require a different parser.
    Complete parsing is inherited from :class:`ToolParser` and therefore uses
    this incremental state machine too.
    """

    name_field: ClassVar[str] = 'name'
    argument_field: ClassVar[str] = 'arguments'

    def __init__(self) -> None:
        """Initialize the JSON envelope state machine."""
        super().__init__()
        self._phase: JsonParsePhase = 'payload_start'
        # Field name retained across chunks until its value is consumed.
        self._json_key: str | None = None
        self._arguments_seen = False
        self._value_scanner = JsonValueScanner()
        self._close_tag = self.get_tool_close_tag()

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return None

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
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
        in the input buffer. Once the envelope is done, any outer delimiter or
        trailing content remains for :class:`ToolParser` and the response
        parser respectively.

        Args:
            text: Buffered payload text after the outer opening marker.
            deltas: Destination for parsed function-name and argument deltas.
            final: Whether no more generated text will follow.

        Returns:
            Number of leading characters that may be discarded from ``text``.
        """
        pos = 0
        size = len(text)
        close_tag = self._close_tag

        while pos < size and not self._payload_closed:
            if self._phase not in ('arguments', 'skip_value') and close_tag and text.startswith(close_tag, pos):
                self._finish_envelope(deltas)
                break

            if self._phase == 'payload_start':
                pos = self._skip_ws(text, pos)
                if pos == size:
                    break
                if text[pos] != '{':
                    # Recover from a malformed prefix by looking for a later envelope opener.
                    pos += 1
                    continue
                pos += 1
                self._phase = 'key'
                continue

            if self._phase == 'key':
                # Whitespace is valid here; extra commas are ignored while recovering the next field.
                while pos < size and (text[pos].isspace() or text[pos] == ','):
                    pos += 1
                if pos == size:
                    break
                if text[pos] == '}':
                    pos += 1
                    self._finish_envelope(deltas)
                    continue
                if text[pos] != '"':
                    # Recover by scanning for the next quoted field name or closing brace.
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
                    # Recover by scanning for the missing key/value separator.
                    pos += 1
                    continue
                pos += 1
                self._phase = 'value_start'
                continue

            if self._phase == 'value_start':
                pos = self._skip_ws(text, pos)
                if pos == size:
                    break
                if self._json_key == self.name_field and text[pos] == '"':
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
                else:
                    self._value_scanner.reset()
                    if self._json_key == self.argument_field:
                        self._arguments_seen = True
                        self._phase = 'arguments'
                    else:
                        self._phase = 'skip_value'
                continue

            if self._phase in ('arguments', 'skip_value'):
                if self._value_scanner.in_string and text.find('"', pos) < 0:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if self._phase == 'arguments' and pos > start:
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
                if self._phase == 'arguments' and pos > start:
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
                    if self._phase == 'arguments':
                        self._emit_delta(
                            deltas,
                            arguments=text[start:pos],
                        )
                    continue

                if scan_limit < size and pos == scan_limit and self._value_scanner.in_string:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if self._phase == 'arguments' and pos > start:
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
                # Skipping JSON whitespace may expose an outer closing marker.
                # Treat it as the protocol boundary before malformed-envelope
                # recovery can consume the marker's leading character.
                if close_tag and text.startswith(close_tag, pos):
                    self._finish_envelope(deltas)
                    break
                if text[pos] == ',':
                    pos += 1
                    self._phase = 'key'
                elif text[pos] == '}':
                    pos += 1
                    self._finish_envelope(deltas)
                else:
                    # Recover by scanning for the next field separator or closing brace.
                    pos += 1
                continue

        if final and not self._payload_closed and self._phase in ('arguments', 'skip_value'):
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

    @staticmethod
    def _read_string(text: str, start: int) -> tuple[str, int]:
        """Decode a quoted JSON string and return its value and end offset.

        A completed but malformed string falls back to its raw inner text.
        ``('', -1)`` indicates that the closing quote is not buffered yet.
        """
        escaped = False
        # "+ 1" to skip the opening quote
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
