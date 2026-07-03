# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/tool_parsers
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import partial_json_parser
import shortuuid
from mmengine import Registry
from partial_json_parser.core.options import Allow

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


class ToolParser:
    """Base class for model-specific tool parsers."""

    def __init__(self):
        self._tool_payload: str = ''
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False
        self._args_emitted_len: int = 0
        self._args_payload_start: int = -1

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
        self._args_emitted_len = 0
        self._args_payload_start = -1
        self._tool_payload = ''

    def finish_tool_call(self) -> None:
        """Mark end of a tool-call block."""
        self._active_tool_call_id = ''
        self._name_emitted = False
        self._args_emitted_len = 0
        self._args_payload_start = -1
        self._tool_payload = ''

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
        self._tool_payload += added_text
        raw_payload = self._tool_payload.lstrip()
        payload = raw_payload.strip()
        if not payload:
            return []

        flags = Allow.ALL if self._name_emitted else Allow.ALL & ~Allow.STR
        try:
            obj = partial_json_parser.loads(payload, flags)
        except partial_json_parser.core.exceptions.MalformedJSON:
            return []
        if not isinstance(obj, dict):
            return []

        out: list[DeltaToolCall] = []
        if not self._name_emitted:
            fn_name = obj.get('name')
            if isinstance(fn_name, str) and fn_name:
                out.append(
                    DeltaToolCall(
                        id=self._active_tool_call_id,
                        index=self._active_tool_index,
                        type='function',
                        function=DeltaFunctionCall(name=fn_name),
                    ))
                self._name_emitted = True

        args_obj = obj.get('arguments', obj.get('parameters', None))
        if args_obj is None:
            return out

        if self._args_payload_start < 0:
            args_key = 'arguments' if 'arguments' in obj else 'parameters'
            self._args_payload_start = self._find_json_key_value_start(raw_payload, args_key)
        if self._args_payload_start < 0:
            return out

        args_payload_end = self._find_json_value_end(raw_payload, self._args_payload_start)
        if args_payload_end < 0:
            args_payload_end = len(raw_payload)

        args_payload = raw_payload[self._args_payload_start:args_payload_end]
        if len(args_payload) > self._args_emitted_len:
            diff = args_payload[self._args_emitted_len:]
            out.append(
                DeltaToolCall(
                    id=None,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=diff),
                ))
            self._args_emitted_len = len(args_payload)
        return out

    @staticmethod
    def _find_json_key_value_start(payload: str, key: str) -> int:
        depth = 0
        i = 0
        while i < len(payload):
            ch = payload[i]
            if ch == '"':
                value, end = ToolParser._read_json_string(payload, i)
                if end < 0:
                    return -1
                j = end
                while j < len(payload) and payload[j].isspace():
                    j += 1
                if depth == 1 and value == key and j < len(payload) and payload[j] == ':':
                    j += 1
                    while j < len(payload) and payload[j].isspace():
                        j += 1
                    return j if j < len(payload) else -1
                i = end
                continue
            if ch in '{[':
                depth += 1
            elif ch in '}]':
                depth -= 1
            i += 1
        return -1

    @staticmethod
    def _find_json_value_end(payload: str, start: int) -> int:
        depth = 0
        i = start
        while i < len(payload):
            ch = payload[i]
            if ch == '"':
                _, end = ToolParser._read_json_string(payload, i)
                if end < 0:
                    return -1
                i = end
                continue
            if ch in '{[':
                depth += 1
            elif ch in '}]':
                if depth == 0:
                    return i
                depth -= 1
                if depth == 0:
                    return i + 1
            elif depth == 0 and ch in ',':
                return i
            i += 1
        return -1

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
