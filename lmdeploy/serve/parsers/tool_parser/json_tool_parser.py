# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from dataclasses import dataclass

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParser


@dataclass
class JsonToolSnapshot:
    func_name: str | None
    args_delta: str


class JsonToolParser(ToolParser):
    """Base class for JSON tool-call payload parsers."""

    def __init__(self):
        super().__init__()
        self._payload: str = ''
        self._phase: str = 'payload_start'
        self._json_key: str | None = None
        self._value_type: str | None = None
        self._value_depth: int = 0
        self._string_open_in_container: bool = False
        self._value_escaped: bool = False
        self._payload_closed: bool = False

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'json'

    def start_tool_call(self) -> None:
        super().start_tool_call()
        self._reset_stream_state()

    def finish_tool_call(self) -> None:
        super().finish_tool_call()
        self._reset_stream_state()

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Stream raw JSON tool argument text without requiring completion.

        ``_payload`` only keeps unconsumed syntax such as a partial key or
        delimiter. Argument value bytes are emitted as deltas once they are
        observed, while string/container state is tracked separately.
        """
        self._payload += added_text
        snapshot, consumed = self._consume_payload(self._payload, final=final)
        if consumed > 0:
            self._payload = self._payload[consumed:]

        out: list[DeltaToolCall] = []
        if snapshot.func_name and not self._name_emitted:
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type='function',
                    function=DeltaFunctionCall(name=snapshot.func_name),
                ))
            self._name_emitted = True
        if snapshot.args_delta:
            out.append(
                DeltaToolCall(
                    id=None,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=snapshot.args_delta),
                ))
        return out

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        if not payload:
            return None
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get('name')
        if not isinstance(name, str) or not name:
            return None
        args_obj = obj.get('arguments', obj.get('parameters', {}))
        args_json = json.dumps(args_obj, ensure_ascii=False)
        return ToolCall(function=FunctionCall(name=name, arguments=args_json))

    def _validate_tool_payload(self, payload: str) -> bool:
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return False
        if not isinstance(obj, dict):
            return False
        name = obj.get('name')
        return isinstance(name, str) and bool(name)

    def _consume_payload(self, payload: str, *, final: bool) -> tuple[JsonToolSnapshot, int]:
        pos = 0
        args_delta_parts: list[str] = []
        func_name: str | None = None
        n = len(payload)

        while pos < n:
            if self._phase == 'payload_start':
                pos = self._skip_ws(payload, pos)
                if pos >= n:
                    break
                if payload[pos] != '{':
                    break
                pos += 1
                self._phase = 'key'
                continue

            if self._phase == 'key':
                pos = self._skip_key_prefix(payload, pos)
                if pos >= n:
                    break
                if payload[pos] == '}':
                    pos += 1
                    self._phase = 'done'
                    self._payload_closed = True
                    break
                if payload[pos] != '"':
                    break
                key, end = self._read_string(payload, pos)
                if end < 0:
                    break
                self._json_key = key
                pos = end
                self._phase = 'colon'
                continue

            if self._phase == 'colon':
                pos = self._skip_ws(payload, pos)
                if pos >= n:
                    break
                if payload[pos] != ':':
                    break
                pos += 1
                self._phase = 'value_start'
                continue

            if self._phase == 'value_start':
                pos = self._skip_ws(payload, pos)
                if pos >= n:
                    break
                if self._json_key == 'name':
                    if payload[pos] == '"':
                        name, end = self._read_string(payload, pos)
                        if end < 0:
                            break
                        func_name = name
                        pos = end
                        self._json_key = None
                        self._phase = 'key'
                        continue
                    self._phase = 'skip_value'
                    continue
                if self._json_key in ('arguments', 'parameters'):
                    self._phase = 'args_value'
                    continue
                self._phase = 'skip_value'
                continue

            if self._phase == 'args_value':
                delta, consumed, complete = self._consume_value(payload, pos, emit=True)
                if delta:
                    args_delta_parts.append(delta)
                pos += consumed
                if complete:
                    self._json_key = None
                    self._phase = 'key'
                    self._reset_value_state()
                    continue
                break

            if self._phase == 'skip_value':
                _, consumed, complete = self._consume_value(payload, pos, emit=False)
                pos += consumed
                if complete:
                    self._json_key = None
                    self._phase = 'key'
                    self._reset_value_state()
                    continue
                break

            break

        return JsonToolSnapshot(func_name, ''.join(args_delta_parts)), pos

    def _consume_value(self, payload: str, start: int, *, emit: bool) -> tuple[str, int, bool]:
        if start >= len(payload):
            return '', 0, False

        i = start
        if self._value_type is None:
            ch = payload[i]
            if ch == '"':
                self._value_type = 'string'
                self._value_escaped = False
                i += 1
            elif ch in '{[':
                self._value_type = 'container'
                self._value_depth = 1
                self._string_open_in_container = False
                self._value_escaped = False
                i += 1
            else:
                self._value_type = 'scalar'

        complete = False
        while i < len(payload):
            ch = payload[i]
            if self._value_type == 'string':
                if self._value_escaped:
                    self._value_escaped = False
                elif ch == '\\':
                    self._value_escaped = True
                elif ch == '"':
                    i += 1
                    complete = True
                    break
            elif self._value_type == 'container':
                if self._string_open_in_container:
                    if self._value_escaped:
                        self._value_escaped = False
                    elif ch == '\\':
                        self._value_escaped = True
                    elif ch == '"':
                        self._string_open_in_container = False
                elif ch == '"':
                    self._string_open_in_container = True
                elif ch in '{[':
                    self._value_depth += 1
                elif ch in '}]':
                    self._value_depth -= 1
                    if self._value_depth == 0:
                        i += 1
                        complete = True
                        break
            elif ch in ',}]':
                complete = True
                break
            i += 1

        consumed = i - start
        return payload[start:i] if emit else '', consumed, complete

    @staticmethod
    def _read_string(payload: str, start: int) -> tuple[str, int]:
        chars: list[str] = []
        escaped = False
        i = start + 1
        while i < len(payload):
            ch = payload[i]
            if escaped:
                chars.append(ch)
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                return ''.join(chars), i + 1
            else:
                chars.append(ch)
            i += 1
        return '', -1

    @staticmethod
    def _skip_ws(payload: str, pos: int) -> int:
        while pos < len(payload) and payload[pos].isspace():
            pos += 1
        return pos

    @staticmethod
    def _skip_key_prefix(payload: str, pos: int) -> int:
        while pos < len(payload) and (payload[pos].isspace() or payload[pos] == ','):
            pos += 1
        return pos

    def _reset_stream_state(self) -> None:
        self._payload = ''
        self._phase = 'payload_start'
        self._json_key = None
        self._payload_closed = False
        self._reset_value_state()

    def _reset_value_state(self) -> None:
        self._value_type = None
        self._value_depth = 0
        self._string_open_in_container = False
        self._value_escaped = False
