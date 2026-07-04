# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import shortuuid
from mmengine import Registry

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

from ..response_parser import BaseResponseParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.parsers.tool_parser'])


@dataclass
class JsonToolSnapshot:
    func_name: str | None
    args_delta: str


class ToolParser:
    """Base class for model-specific tool parsers."""

    def __init__(self):
        self._tool_payload: str = ''
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False
        self._json_phase: str = 'payload_start'
        self._json_key: str | None = None
        self._json_args_key: str | None = None
        self._json_value_kind: str | None = None
        self._json_value_depth: int = 0
        self._json_value_in_string: bool = False
        self._json_value_escaped: bool = False
        self._json_payload_closed: bool = False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request payload before rendering, if needed."""
        return BaseResponseParser.dump_tools(request)

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        """Return tool opening tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_open_tag has not been implemented!')

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        """Return tool closing tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_close_tag has not been implemented!')

    @classmethod
    def get_tool_payload_format(cls) -> str:
        """Return payload format for tool call body."""
        raise NotImplementedError('ToolParser.get_tool_payload_format has not been implemented!')

    def start_tool_call(self) -> None:
        """Mark start of a tool-call block."""
        self._active_tool_index += 1
        self._active_tool_call_id = f'chatcmpl-tool-{shortuuid.random()}'
        self._name_emitted = False
        self._reset_json_stream_state()

    def finish_tool_call(self) -> None:
        """Mark end of a tool-call block."""
        self._active_tool_call_id = ''
        self._name_emitted = False
        self._reset_json_stream_state()

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental tool payload emitted between tool tags."""
        raise NotImplementedError('ToolParser.decode_tool_incremental has not been implemented!')

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        """Parse one complete tool payload into OpenAI tool call object."""
        raise NotImplementedError('ToolParser.parse_tool_call_complete has not been implemented!')

    def validate_complete(self, text: str) -> bool:
        """Return whether complete response text has valid tool calls."""
        open_tag = self.get_tool_open_tag()
        close_tag = self.get_tool_close_tag()

        pos = 0
        while True:
            open_idx = text.find(open_tag, pos)
            close_idx = text.find(close_tag, pos)
            if open_idx < 0:
                return close_idx < 0

            payload_start = open_idx + len(open_tag)
            if close_idx < payload_start:
                return False

            payload = text[payload_start:close_idx].strip()
            if not self._validate_tool_payload(payload):
                return False

            pos = close_idx + len(close_tag)
            if pos >= len(text):
                return True

    def _validate_tool_payload(self, payload: str) -> bool:
        """Return whether one complete JSON tool payload is structurally
        valid."""
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return False
        if not isinstance(obj, dict):
            return False
        name = obj.get('name')
        return isinstance(name, str) and bool(name)

    def _decode_tool_incremental_json(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Stream raw JSON tool argument text without requiring completion.

        ``_tool_payload`` only keeps unconsumed syntax such as a partial key or
        delimiter. Argument value bytes are emitted as deltas once they are
        observed, while string/container state is tracked separately.
        """
        self._tool_payload += added_text
        snapshot, consumed = self._consume_json_payload(self._tool_payload)
        if consumed > 0:
            self._tool_payload = self._tool_payload[consumed:]

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

    def _consume_json_payload(self, payload: str) -> tuple[JsonToolSnapshot, int]:
        pos = 0
        args_delta_parts: list[str] = []
        func_name: str | None = None
        n = len(payload)

        while pos < n:
            if self._json_phase == 'payload_start':
                pos = self._skip_json_ws(payload, pos)
                if pos >= n:
                    break
                if payload[pos] != '{':
                    break
                pos += 1
                self._json_phase = 'key'
                continue

            if self._json_phase == 'key':
                pos = self._skip_json_key_prefix(payload, pos)
                if pos >= n:
                    break
                if payload[pos] == '}':
                    pos += 1
                    self._json_phase = 'done'
                    self._json_payload_closed = True
                    break
                if payload[pos] != '"':
                    break
                key, end = self._read_json_string(payload, pos)
                if end < 0:
                    break
                self._json_key = key
                pos = end
                self._json_phase = 'colon'
                continue

            if self._json_phase == 'colon':
                pos = self._skip_json_ws(payload, pos)
                if pos >= n:
                    break
                if payload[pos] != ':':
                    break
                pos += 1
                self._json_phase = 'value_start'
                continue

            if self._json_phase == 'value_start':
                pos = self._skip_json_ws(payload, pos)
                if pos >= n:
                    break
                if self._json_key == 'name':
                    if payload[pos] == '"':
                        name, end = self._read_json_string(payload, pos)
                        if end < 0:
                            break
                        func_name = name
                        pos = end
                        self._json_key = None
                        self._json_phase = 'key'
                        continue
                    self._json_phase = 'skip_value'
                    continue
                if self._json_key in ('arguments', 'parameters') and self._json_args_key is None:
                    self._json_args_key = self._json_key
                    self._json_phase = 'args_value'
                    continue
                self._json_phase = 'skip_value'
                continue

            if self._json_phase == 'args_value':
                delta, consumed, complete = self._consume_json_value(payload, pos, emit=True)
                if delta:
                    args_delta_parts.append(delta)
                pos += consumed
                if complete:
                    self._json_key = None
                    self._json_phase = 'key'
                    self._reset_json_value_state()
                    continue
                break

            if self._json_phase == 'skip_value':
                _, consumed, complete = self._consume_json_value(payload, pos, emit=False)
                pos += consumed
                if complete:
                    self._json_key = None
                    self._json_phase = 'key'
                    self._reset_json_value_state()
                    continue
                break

            break

        return JsonToolSnapshot(func_name, ''.join(args_delta_parts)), pos

    def _consume_json_value(self, payload: str, start: int, *, emit: bool) -> tuple[str, int, bool]:
        if start >= len(payload):
            return '', 0, False

        i = start
        if self._json_value_kind is None:
            ch = payload[i]
            if ch == '"':
                self._json_value_kind = 'string'
                self._json_value_escaped = False
                i += 1
            elif ch in '{[':
                self._json_value_kind = 'container'
                self._json_value_depth = 1
                self._json_value_in_string = False
                self._json_value_escaped = False
                i += 1
            else:
                self._json_value_kind = 'scalar'

        complete = False
        while i < len(payload):
            ch = payload[i]
            if self._json_value_kind == 'string':
                if self._json_value_escaped:
                    self._json_value_escaped = False
                elif ch == '\\':
                    self._json_value_escaped = True
                elif ch == '"':
                    i += 1
                    complete = True
                    break
            elif self._json_value_kind == 'container':
                if self._json_value_in_string:
                    if self._json_value_escaped:
                        self._json_value_escaped = False
                    elif ch == '\\':
                        self._json_value_escaped = True
                    elif ch == '"':
                        self._json_value_in_string = False
                elif ch == '"':
                    self._json_value_in_string = True
                elif ch in '{[':
                    self._json_value_depth += 1
                elif ch in '}]':
                    self._json_value_depth -= 1
                    if self._json_value_depth == 0:
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
    def _read_json_string(payload: str, start: int) -> tuple[str, int]:
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
    def _skip_json_ws(payload: str, pos: int) -> int:
        while pos < len(payload) and payload[pos].isspace():
            pos += 1
        return pos

    @staticmethod
    def _skip_json_key_prefix(payload: str, pos: int) -> int:
        while pos < len(payload) and (payload[pos].isspace() or payload[pos] == ','):
            pos += 1
        return pos

    def _reset_json_stream_state(self) -> None:
        self._tool_payload = ''
        self._json_phase = 'payload_start'
        self._json_key = None
        self._json_args_key = None
        self._json_payload_closed = False
        self._reset_json_value_state()

    def _reset_json_value_state(self) -> None:
        self._json_value_kind = None
        self._json_value_depth = 0
        self._json_value_in_string = False
        self._json_value_escaped = False

    @staticmethod
    def _parse_tool_call_complete_json(payload: str) -> ToolCall | None:
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
