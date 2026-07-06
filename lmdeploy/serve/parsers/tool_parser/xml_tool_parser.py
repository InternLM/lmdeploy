# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
)

from .tool_parser import ToolParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@dataclass
class XmlToolSnapshot:
    func_name: str | None
    completed_args: dict[str, str]
    arg_name: str | None
    arg_delta: str
    payload_closed: bool


@dataclass
class XmlParseState:
    phase: str
    func_name: str | None
    args: dict[str, str]
    arg_name: str | None
    payload_closed: bool


@dataclass
class XmlParseResult:
    next_pos: int | None
    next_phase: str | None = None
    func_name: str | None = None
    arg_name: str | None = None
    raw_arg_delta: str = ''
    completed_arg_value: str | None = None
    payload_closed: bool = False
    should_stop: bool = False


class XmlToolParser(ToolParser):
    """Base class for XML-like tool parsers.

    Subclasses only need to implement XML payload extraction.
    """

    def __init__(self):
        super().__init__()
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_names: set[str] = set()
        self._payload_parts: list[str] = []
        self._coerced_args: dict[str, Any] = {}
        self._streamed_arg_name: str | None = None
        self._streamed_arg_emitted_len = 0
        self._streamed_arg_quote_opened = False
        self._xml_state = XmlParseState('function', None, {}, None, False)

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
        self._coerced_args.clear()
        self._xml_state = XmlParseState('function', None, {}, None, False)
        self._reset_arg()
        self._reset_incremental_state()

    def _reset_incremental_state(self) -> None:
        """Reset subclass-specific incremental parse state."""

    def _consume_payload(self, payload: str, *, final: bool) -> tuple[XmlToolSnapshot, int]:
        pos = 0
        arg_delta_parts: list[str] = []
        state = self._xml_state

        while pos < len(payload):
            if state.phase == 'function':
                result = self._consume_function(payload, pos, final)
            elif state.phase == 'arg_start':
                result = self._consume_arg_start(payload, pos)
            elif state.phase == 'arg_name':
                result = self._consume_arg_name(payload, pos)
            elif state.phase == 'arg_value':
                result = self._consume_arg_value(payload, pos)
            else:
                break

            if result.next_pos is None:
                break

            if result.func_name is not None:
                state.func_name = result.func_name
            if result.arg_name is not None:
                state.arg_name = result.arg_name
            if result.raw_arg_delta:
                stream_delta = self._stream_arg_delta(result.raw_arg_delta)
                if stream_delta:
                    arg_delta_parts.append(stream_delta)
            if result.completed_arg_value is not None:
                if state.arg_name is None:
                    raise RuntimeError('XML parser completed an argument without an active argument name')
                state.args[state.arg_name] = result.completed_arg_value
                state.arg_name = None
            if result.payload_closed:
                state.payload_closed = True
            if result.next_phase is not None:
                state.phase = result.next_phase

            pos = result.next_pos
            if result.should_stop:
                break

        return (
            XmlToolSnapshot(
                state.func_name,
                dict(state.args),
                state.arg_name,
                ''.join(arg_delta_parts),
                state.payload_closed,
            ),
            pos,
        )

    def _consume_function(self, payload: str, pos: int, final: bool) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_function has not been implemented!')

    def _consume_arg_start(self, payload: str, pos: int) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_arg_start has not been implemented!')

    def _consume_arg_name(self, payload: str, pos: int) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_arg_name has not been implemented!')

    def _consume_arg_value(self, payload: str, pos: int) -> XmlParseResult:
        raise NotImplementedError('XmlToolParser._consume_arg_value has not been implemented!')

    def _stream_arg_delta(self, raw: str) -> str:
        state = self._xml_state
        if state.arg_name is None:
            raise RuntimeError('XML parser streamed an argument value without an active argument name')

        schema_type = self._get_param_schema_type(state.func_name, state.arg_name)
        if schema_type not in (None, 'string'):
            self._block_stream_arg_delta()
            return ''
        return self._normalize_stream_arg_delta(raw, schema_type)

    def _block_stream_arg_delta(self) -> None:
        raise NotImplementedError('XmlToolParser._block_stream_arg_delta has not been implemented!')

    def _normalize_stream_arg_delta(self, raw: str, schema_type: str | None) -> str:
        raise NotImplementedError('XmlToolParser._normalize_stream_arg_delta has not been implemented!')

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

        should_close = snapshot.payload_closed or (final and self._close_json_on_final())

        json_fragments: list[str] = []
        completed_args = self._get_coerced_args(snapshot.func_name, snapshot.completed_args)
        self._append_finished_arg(json_fragments, completed_args)
        self._append_completed_args(json_fragments, completed_args)
        self._append_open_arg(json_fragments, snapshot)

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

    def _append_json_start(self, json_fragments: list[str]) -> None:
        if not self._has_emitted_json_start:
            json_fragments.append('{')
            self._has_emitted_json_start = True

    def _append_finished_arg(self, json_fragments: list[str], completed_args: dict[str, Any]) -> None:
        arg_name = self._streamed_arg_name
        if arg_name is None or arg_name not in completed_args or arg_name not in self._emitted_arg_names:
            return
        value = completed_args[arg_name]
        if self._streamed_arg_quote_opened:
            if isinstance(value, str) and len(value) > self._streamed_arg_emitted_len:
                diff = value[self._streamed_arg_emitted_len:]
                json_fragments.append(json.dumps(diff, ensure_ascii=False)[1:-1])
            json_fragments.append('"')
        else:
            value_text = json.dumps(value, ensure_ascii=False)
            if len(value_text) > self._streamed_arg_emitted_len:
                json_fragments.append(value_text[self._streamed_arg_emitted_len:])
        self._reset_arg()

    def _append_completed_args(self, json_fragments: list[str], completed_args: dict[str, Any]) -> None:
        for key, value in completed_args.items():
            if key in self._emitted_arg_names:
                continue
            self._append_json_start(json_fragments)
            prefix = ', ' if len(self._emitted_arg_names) > 0 else ''
            json_fragments.append(f'{prefix}"{key}": {json.dumps(value, ensure_ascii=False)}')
            self._emitted_arg_names.add(key)

    def _append_open_arg(self, json_fragments: list[str], snapshot: XmlToolSnapshot) -> None:
        if snapshot.arg_name is None or not snapshot.arg_delta:
            return

        if self._streamed_arg_name == snapshot.arg_name:
            json_fragments.append(json.dumps(snapshot.arg_delta, ensure_ascii=False)[1:-1])
            self._streamed_arg_emitted_len += len(snapshot.arg_delta)
            return

        if snapshot.arg_name in self._emitted_arg_names:
            return

        schema_type = self._get_param_schema_type(snapshot.func_name, snapshot.arg_name)
        if schema_type not in (None, 'string'):
            return

        self._append_json_start(json_fragments)
        prefix = ', ' if len(self._emitted_arg_names) > 0 else ''
        json_fragments.append(f'{prefix}"{snapshot.arg_name}": "')
        diff = json.dumps(snapshot.arg_delta, ensure_ascii=False)[1:-1]
        json_fragments.append(diff)
        self._emitted_arg_names.add(snapshot.arg_name)
        self._streamed_arg_name = snapshot.arg_name
        self._streamed_arg_emitted_len = len(snapshot.arg_delta)
        self._streamed_arg_quote_opened = True

    def _reset_arg(self) -> None:
        self._streamed_arg_name = None
        self._streamed_arg_emitted_len = 0
        self._streamed_arg_quote_opened = False

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

    def _get_coerced_args(self,
                          func_name: str | None,
                          raw_args_dict: dict[str, str],
                          *,
                          use_cache: bool = True) -> dict[str, Any]:
        if not func_name or not raw_args_dict:
            return raw_args_dict
        param_schemas = self._function_param_schemas.get(func_name, {})

        coerced = dict(self._coerced_args) if use_cache else {}
        for key, value in raw_args_dict.items():
            if use_cache and key in self._coerced_args:
                continue
            schema = param_schemas.get(key)
            schema_type = self._resolve_schema_type(schema) if isinstance(schema, dict) else None
            coerced_value = self._coerce_value(value, schema_type)
            if use_cache:
                self._coerced_args[key] = coerced_value
            coerced[key] = coerced_value
        return coerced

    def _close_json_on_final(self) -> bool:
        return True
