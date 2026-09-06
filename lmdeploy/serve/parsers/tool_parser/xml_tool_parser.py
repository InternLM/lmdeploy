# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from json.encoder import encode_basestring
from typing import TYPE_CHECKING, Any

from lmdeploy.serve.openai.protocol import (
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
class XmlArgState:
    """State needed to decide whether a value can be streamed safely."""

    mode: str = 'undecided'
    pending_prefix: str = ''
    pending_suffix: str = ''
    buffered_parts: list[str] = field(default_factory=list)
    header_emitted: bool = False
    prefix_done: bool = False


class XmlToolParser(ToolParser):
    """Base class for incremental XML-like tool parsers.

    Format adapters advance the shared syntax state directly. This class owns JSON emission, schema coercion, and stream
    lifecycle state. Unquoted string values are emitted immediately and discarded; only undecided syntax and non-
    streamable typed values are retained.
    """

    strip_value_newlines = False

    def __init__(self):
        super().__init__()
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_count = 0
        self._state = XmlParseState()
        self._arg_state = XmlArgState()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        self._function_param_schemas = self._build_function_param_schemas(request)
        return super().adjust_request(request)

    def begin_tool_block(self) -> None:
        super().begin_tool_block()
        self._begin_call()
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_count = 0
        self._state = XmlParseState()
        self._arg_state = XmlArgState()

    def _consume_payload(self, payload: str, json_fragments: list[str], *, final: bool) -> int:
        pos = 0

        while pos < len(payload):
            phase = self._state.phase
            if self._state.phase == 'function':
                next_pos = self._consume_function(payload, pos, final)
            elif self._state.phase == 'arg_start':
                next_pos = self._consume_arg_start(payload, pos)
            elif self._state.phase == 'arg_name':
                next_pos = self._consume_arg_name(payload, pos)
            elif self._state.phase == 'arg_value':
                next_pos = self._consume_arg_value(payload, pos, json_fragments)
            elif self._state.phase == 'done':
                while pos < len(payload) and payload[pos].isspace():
                    pos += 1
                break
            else:
                break

            if next_pos is None:
                break
            if next_pos == pos and self._state.phase == phase:
                break
            pos = next_pos

        return pos

    def _consume_function(self, payload: str, pos: int, final: bool) -> int | None:
        raise NotImplementedError('XmlToolParser._consume_function has not been implemented!')

    def _consume_arg_start(self, payload: str, pos: int) -> int | None:
        raise NotImplementedError('XmlToolParser._consume_arg_start has not been implemented!')

    def _consume_arg_name(self, payload: str, pos: int) -> int | None:
        raise NotImplementedError('XmlToolParser._consume_arg_name has not been implemented!')

    def _consume_arg_value(self, payload: str, pos: int, json_fragments: list[str]) -> int | None:
        raise NotImplementedError('XmlToolParser._consume_arg_value has not been implemented!')

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        json_fragments: list[str] = []
        consumed = self._consume_payload(text, json_fragments, final=final)

        if self._state.func_name is not None and not self._name_emitted:
            self._emit_delta(
                deltas,
                name=self._state.func_name,
            )
            self._name_emitted = True

        should_close = self._payload_closed or (final and self._close_json_on_final())
        if should_close and not self._has_emitted_json_start:
            json_fragments.append('{')
            self._has_emitted_json_start = True
        if should_close and self._has_emitted_json_start and not self._json_closed:
            json_fragments.append('}')
            self._json_closed = True

        if json_fragments:
            self._emit_delta(
                deltas,
                arguments=''.join(json_fragments),
            )
        if final and should_close:
            self._payload_closed = True
        return consumed

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

        if not arg_state.prefix_done:
            text = arg_state.pending_prefix + raw
            arg_state.pending_prefix = ''
            if self.strip_value_newlines:
                if text == '\r':
                    arg_state.pending_prefix = text
                    return
                if text.startswith('\r\n'):
                    text = text[2:]
                elif text.startswith('\n'):
                    text = text[1:]
            arg_state.prefix_done = True
        else:
            text = raw

        arg_state.pending_prefix += text
        if not arg_state.pending_prefix.lstrip():
            return
        text = arg_state.pending_prefix
        arg_state.pending_prefix = ''

        param_schema = self._get_param_schema(self._state.func_name, arg_name)
        schema_type = self._get_schema_type(param_schema)
        if param_schema is not None and schema_type != 'string':
            arg_state.mode = 'buffered'
            arg_state.buffered_parts.append(text)
            return
        if text.lstrip().startswith('"'):
            arg_state.mode = 'buffered'
            arg_state.buffered_parts.append(text)
            return

        arg_state.mode = 'streaming'
        self._stream_string_delta(text, json_fragments)

    def _stream_string_delta(self, text: str, json_fragments: list[str]) -> None:
        arg_state = self._arg_state
        text = arg_state.pending_suffix + text
        arg_state.pending_suffix = ''
        if self.strip_value_newlines and text:
            last = text[-1]
            if last == '\n':
                if len(text) > 1 and text[-2] == '\r':
                    stable = text[:-2]
                    arg_state.pending_suffix = '\r\n'
                else:
                    stable = text[:-1]
                    arg_state.pending_suffix = '\n'
            elif last == '\r':
                stable = text[:-1]
                arg_state.pending_suffix = '\r'
            else:
                stable = text
        else:
            stable = text
        if not stable:
            return

        self._append_streaming_arg_header(json_fragments)
        json_fragments.append(encode_basestring(stable)[1:-1])

    def _finish_arg(self, json_fragments: list[str]) -> None:
        arg_name = self._state.arg_name
        if arg_name is None:
            self._arg_state = XmlArgState()
            return

        if self._arg_state.mode == 'streaming':
            self._append_streaming_arg_header(json_fragments)
            if self._arg_state.pending_suffix == '\r':
                json_fragments.append('\\r')
            json_fragments.append('"')
        else:
            if self._arg_state.mode == 'buffered':
                raw_value = ''.join(self._arg_state.buffered_parts)
            else:
                raw_value = self._arg_state.pending_prefix
            raw_value = self._normalize_complete_value(raw_value)
            func_name = self._state.func_name
            schema = self._get_param_schema(func_name, arg_name)
            value = self._coerce_arg_value(raw_value, schema)
            self._append_completed_arg(json_fragments, arg_name, value)
        self._arg_state = XmlArgState()

    def _append_json_start(self, json_fragments: list[str]) -> None:
        if not self._has_emitted_json_start:
            json_fragments.append('{')
            self._has_emitted_json_start = True

    def _append_completed_arg(self, json_fragments: list[str], arg_name: str, value: Any) -> None:
        self._append_json_start(json_fragments)
        prefix = ', ' if self._emitted_arg_count else ''
        key = json.dumps(arg_name, ensure_ascii=False)
        json_fragments.append(f'{prefix}{key}: {json.dumps(value, ensure_ascii=False)}')
        self._emitted_arg_count += 1

    def _append_streaming_arg_header(self, json_fragments: list[str]) -> None:
        arg_state = self._arg_state
        if arg_state.header_emitted:
            return
        arg_name = self._state.arg_name
        if arg_name is None:
            return
        self._append_json_start(json_fragments)
        prefix = ', ' if self._emitted_arg_count else ''
        json_fragments.append(f'{prefix}{json.dumps(arg_name, ensure_ascii=False)}: "')
        self._emitted_arg_count += 1
        arg_state.header_emitted = True

    def _normalize_complete_value(self, raw_value: str) -> str:
        if not self.strip_value_newlines:
            return raw_value
        if raw_value.startswith('\r\n'):
            raw_value = raw_value[2:]
        elif raw_value.startswith('\n'):
            raw_value = raw_value[1:]
        if raw_value.endswith('\r\n'):
            raw_value = raw_value[:-2]
        elif raw_value.endswith('\n'):
            raw_value = raw_value[:-1]
        return raw_value

    def _get_param_schema(self, func_name: str | None, param_name: str) -> dict[str, Any] | None:
        if func_name is None:
            return None
        param_schema = self._function_param_schemas.get(func_name, {}).get(param_name)
        return param_schema if isinstance(param_schema, dict) else None

    @staticmethod
    def _get_schema_type(schema: dict[str, Any] | None) -> str | None:
        return XmlToolParser._resolve_schema_type(schema) if schema is not None else None

    def _get_param_schema_type(self, func_name: str | None, param_name: str) -> str | None:
        return self._get_schema_type(self._get_param_schema(func_name, param_name))

    @staticmethod
    def _trim_partial_close_tag_suffix(payload: str, start: int, close_tag: str) -> int:
        """Return safe value end before any partial close-tag suffix."""
        max_len = min(len(payload) - start, len(close_tag) - 1)
        for suffix_len in range(max_len, 0, -1):
            suffix_start = len(payload) - suffix_len
            if close_tag.startswith(payload[suffix_start:]):
                return suffix_start
        return len(payload)

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
        original = raw_value
        raw_value = raw_value.strip()
        if schema_type is None or schema_type == 'string':
            if not raw_value.startswith('"'):
                return original
            try:
                parsed_val = json.loads(raw_value)
                return parsed_val if isinstance(parsed_val, str) else original
            except json.JSONDecodeError:
                return original

        if schema_type == 'integer':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return original
            if isinstance(parsed_val, int):
                return parsed_val
            return original

        if schema_type == 'number':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return original
            if isinstance(parsed_val, (int, float)):
                return parsed_val
            return original

        if schema_type == 'boolean':
            lowered = raw_value.lower()
            if lowered == 'true':
                return True
            if lowered == 'false':
                return False
            return original

        if schema_type == 'null':
            return None if raw_value.lower() == 'null' else original

        if schema_type == 'array':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                return original
            return parsed_val if isinstance(parsed_val, list) else original

        if schema_type == 'object':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                return original
            return parsed_val if isinstance(parsed_val, dict) else original

        return original

    def _coerce_arg_value(self, raw_value: Any, schema: dict[str, Any] | None) -> Any:
        if not isinstance(raw_value, str):
            return raw_value
        return self._coerce_value(raw_value, self._get_schema_type(schema)) if schema is not None else raw_value

    def _get_coerced_args(self, func_name: str | None, raw_arg_pairs: list[tuple[str, str]]) -> list[tuple[str, Any]]:
        if not func_name or not raw_arg_pairs:
            return raw_arg_pairs
        param_schemas = self._function_param_schemas.get(func_name, {})
        if not param_schemas:
            return raw_arg_pairs

        coerced: list[tuple[str, Any]] = []
        for key, value in raw_arg_pairs:
            schema = param_schemas.get(key)
            coerced.append((key, self._coerce_arg_value(value, schema)))
        return coerced

    @staticmethod
    def _dump_argument_pairs(arg_pairs: list[tuple[str, Any]]) -> str:
        fields = (
            f'{json.dumps(name, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}'
            for name, value in arg_pairs
        )
        return '{' + ', '.join(fields) + '}'

    def _close_json_on_final(self) -> bool:
        return True
