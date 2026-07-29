# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
)

from .tool_parser import ToolParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@dataclass
class XmlParseState:
    """Syntax state shared by XML-like parser implementations."""

    phase: str = 'function'
    func_name: str | None = None
    arg_name: str | None = None


@dataclass
class XmlParseResult:
    """One explicit syntax transition returned by a format adapter."""

    next_pos: int | None
    next_phase: str | None = None
    func_name: str | None = None
    arg_name: str | None = None
    arg_delta: str = ''
    arg_closed: bool = False
    payload_closed: bool = False
    should_stop: bool = False


@dataclass
class XmlArgState:
    """State needed to decide whether a value can be streamed safely."""

    mode: str = 'undecided'
    pending_ws: str = ''
    buffered_parts: list[str] = field(default_factory=list)


@dataclass
class XmlToolSnapshot:
    func_name: str | None
    args_delta: str
    payload_closed: bool


class XmlToolParser(ToolParser):
    """Base class for incremental XML-like tool parsers.

    Format adapters identify syntax boundaries and return ``XmlParseResult``.
    This class owns JSON emission, schema coercion, and all stream lifecycle
    state. Unquoted string values are emitted immediately and discarded;
    only undecided syntax, trailing whitespace, and non-streamable values are
    retained.
    """

    def __init__(self):
        super().__init__()
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_names: set[str] = set()
        self._payload_parts: list[str] = []
        self._state = XmlParseState()
        self._arg_state = XmlArgState()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        self._function_param_schemas = self._build_function_param_schemas(request)
        return super().adjust_request(request)

    def start_tool_call(self) -> None:
        super().start_tool_call()
        self._reset_stream_state()

    def finish_tool_call(self) -> None:
        super().finish_tool_call()
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_names.clear()
        self._payload_parts.clear()
        self._state = XmlParseState()
        self._arg_state = XmlArgState()

    def _consume_payload(self, payload: str, *, final: bool) -> tuple[XmlToolSnapshot, int]:
        pos = 0
        json_fragments: list[str] = []

        while pos < len(payload):
            if self._state.phase == 'function':
                result = self._consume_function(payload, pos, final)
            elif self._state.phase == 'arg_start':
                result = self._consume_arg_start(payload, pos)
            elif self._state.phase == 'arg_name':
                result = self._consume_arg_name(payload, pos)
            elif self._state.phase == 'arg_value':
                result = self._consume_arg_value(payload, pos)
            else:
                break

            if result.next_pos is None:
                break

            if result.func_name is not None:
                self._state.func_name = result.func_name
            if result.arg_name is not None:
                self._state.arg_name = result.arg_name
                self._arg_state = XmlArgState()
            if result.arg_delta:
                self._consume_arg_delta(result.arg_delta, json_fragments)
            if result.arg_closed:
                self._finish_arg(json_fragments)
                self._state.arg_name = None
            if result.payload_closed:
                self._payload_closed = True
            if result.next_phase is not None:
                self._state.phase = result.next_phase

            pos = result.next_pos
            if result.should_stop:
                break

        return XmlToolSnapshot(self._state.func_name, ''.join(json_fragments), self._payload_closed), pos

    def _consume_function(self, payload: str, pos: int, final: bool) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_function has not been implemented!')

    def _consume_arg_start(self, payload: str, pos: int) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_arg_start has not been implemented!')

    def _consume_arg_name(self, payload: str, pos: int) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_arg_name has not been implemented!')

    def _consume_arg_value(self, payload: str, pos: int) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_arg_value has not been implemented!')

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._payload_parts.append(added_text)
        payload = ''.join(self._payload_parts)
        snapshot, consumed = self._consume_payload(payload, final=final)

        if consumed > 0:
            left = payload[consumed:]
            self._payload_parts.clear()
            if left:
                self._payload_parts.append(left)

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

        json_fragments = [snapshot.args_delta] if snapshot.args_delta else []
        should_close = snapshot.payload_closed or (final and self._close_json_on_final())
        if should_close and not self._has_emitted_json_start:
            json_fragments.append('{')
            self._has_emitted_json_start = True
        if should_close and self._has_emitted_json_start and not self._json_closed:
            json_fragments.append('}')
            self._json_closed = True

        if json_fragments:
            out.append(
                DeltaToolCall(
                    id=None,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=''.join(json_fragments)),
                ))
        return out

    def _consume_arg_delta(self, raw: str, json_fragments: list[str]) -> None:
        arg_name = self._state.arg_name
        if arg_name is None:
            return

        arg_state = self._arg_state
        if arg_state.mode == 'buffered':
            arg_state.buffered_parts.append(raw)
            return

        if arg_state.mode == 'streaming':
            self._stream_string_delta(raw, json_fragments)
            return

        schema_type = self._get_param_schema_type(self._state.func_name, arg_name)
        if schema_type not in (None, 'string'):
            arg_state.mode = 'buffered'
            arg_state.buffered_parts.append(raw)
            return

        text = arg_state.pending_ws + raw
        arg_state.pending_ws = ''
        stripped = text.lstrip()
        if not stripped:
            arg_state.pending_ws = text
            return
        if stripped.startswith('"'):
            arg_state.mode = 'buffered'
            arg_state.buffered_parts.append(stripped)
            return

        arg_state.mode = 'streaming'
        self._stream_string_delta(stripped, json_fragments)

    def _stream_string_delta(self, text: str, json_fragments: list[str]) -> None:
        arg_state = self._arg_state
        text = arg_state.pending_ws + text
        stable = text.rstrip()
        arg_state.pending_ws = text[len(stable):]
        if not stable:
            return

        arg_name = self._state.arg_name
        if arg_name is None:
            return
        if arg_name not in self._emitted_arg_names:
            self._append_json_start(json_fragments)
            prefix = ', ' if self._emitted_arg_names else ''
            json_fragments.append(f'{prefix}{json.dumps(arg_name, ensure_ascii=False)}: "')
            self._emitted_arg_names.add(arg_name)
        json_fragments.append(json.dumps(stable, ensure_ascii=False)[1:-1])

    def _finish_arg(self, json_fragments: list[str]) -> None:
        arg_name = self._state.arg_name
        if arg_name is None:
            self._arg_state = XmlArgState()
            return

        if self._arg_state.mode == 'streaming':
            json_fragments.append('"')
        else:
            if self._arg_state.mode == 'buffered':
                raw_value = ''.join(self._arg_state.buffered_parts)
            else:
                raw_value = self._arg_state.pending_ws
            schema_type = self._get_param_schema_type(self._state.func_name, arg_name)
            value = self._coerce_value(raw_value, schema_type)
            self._append_completed_arg(json_fragments, arg_name, value)
        self._arg_state = XmlArgState()

    def _append_json_start(self, json_fragments: list[str]) -> None:
        if not self._has_emitted_json_start:
            json_fragments.append('{')
            self._has_emitted_json_start = True

    def _append_completed_arg(self, json_fragments: list[str], arg_name: str, value: Any) -> None:
        if arg_name in self._emitted_arg_names:
            return
        self._append_json_start(json_fragments)
        prefix = ', ' if self._emitted_arg_names else ''
        key = json.dumps(arg_name, ensure_ascii=False)
        json_fragments.append(f'{prefix}{key}: {json.dumps(value, ensure_ascii=False)}')
        self._emitted_arg_names.add(arg_name)

    def _get_param_schema_type(self, func_name: str | None, param_name: str) -> str | None:
        if func_name is None:
            return None
        param_schema = self._function_param_schemas.get(func_name, {}).get(param_name)
        if not isinstance(param_schema, dict):
            return None
        return self._resolve_schema_type(param_schema)

    @staticmethod
    def _trim_partial_close_tag_suffix(payload: str, start: int, close_tag: str) -> int:
        """Return safe value end before any partial close-tag suffix."""
        max_len = min(len(payload) - start, len(close_tag) - 1)
        for suffix_len in range(max_len, 0, -1):
            suffix_start = len(payload) - suffix_len
            if close_tag.startswith(payload[suffix_start:]):
                return suffix_start
        return len(payload)

    def _build_function_param_schemas(self, request: ChatCompletionRequest) -> dict[str, dict[str, dict[str, Any]]]:
        """Build function->parameter schema map from request tools."""
        if not request.tools:
            return {}

        out: dict[str, dict[str, dict[str, Any]]] = {}
        for tool in request.tools:
            parameters = tool.function.parameters
            if not isinstance(parameters, dict):
                continue
            properties = parameters.get('properties')
            if not isinstance(properties, dict):
                continue

            param_schemas = {name: schema for name, schema in properties.items() if isinstance(schema, dict)}
            if param_schemas:
                out[tool.function.name] = param_schemas
        return out

    @staticmethod
    def _resolve_schema_type(param_schema: dict[str, Any]) -> str | None:
        schema_type = param_schema.get('type')
        if isinstance(schema_type, str):
            return schema_type
        if isinstance(schema_type, list):
            for item in schema_type:
                if isinstance(item, str) and item != 'null':
                    return item
            for item in schema_type:
                if isinstance(item, str):
                    return item
        return None

    @staticmethod
    def _coerce_value(raw_value: str, schema_type: str | None) -> Any:
        raw_value = raw_value.strip()
        if schema_type is None or schema_type == 'string':
            if not raw_value.startswith('"'):
                return raw_value
            try:
                parsed_val = json.loads(raw_value)
                return parsed_val if isinstance(parsed_val, str) else raw_value
            except json.JSONDecodeError:
                return raw_value

        if schema_type == 'integer':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return raw_value
            if isinstance(parsed_val, int):
                return parsed_val
            return raw_value

        if schema_type == 'number':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return raw_value
            if isinstance(parsed_val, (int, float)):
                return parsed_val
            return raw_value

        if schema_type == 'boolean':
            lowered = raw_value.lower()
            if lowered == 'true':
                return True
            if lowered == 'false':
                return False
            return raw_value

        if schema_type == 'null':
            return None if raw_value.lower() == 'null' else raw_value

        if schema_type == 'array':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                return raw_value
            return parsed_val if isinstance(parsed_val, list) else raw_value

        if schema_type == 'object':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                return raw_value
            return parsed_val if isinstance(parsed_val, dict) else raw_value

        return raw_value

    def _get_coerced_args(self, func_name: str | None, raw_args_dict: dict[str, str]) -> dict[str, Any]:
        if not func_name or not raw_args_dict:
            return raw_args_dict
        param_schemas = self._function_param_schemas.get(func_name, {})

        coerced: dict[str, Any] = {}
        for key, value in raw_args_dict.items():
            schema = param_schemas.get(key)
            schema_type = self._resolve_schema_type(schema) if isinstance(schema, dict) else None
            coerced[key] = self._coerce_value(value, schema_type)
        return coerced

    def _close_json_on_final(self) -> bool:
        return True
