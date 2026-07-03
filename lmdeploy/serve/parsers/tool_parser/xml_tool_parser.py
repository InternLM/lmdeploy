# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
)

from .tool_parser import ToolParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


class XmlToolParser(ToolParser):
    """Base class for XML-like tool parsers.

    Subclasses only need to implement XML payload extraction.
    """

    def __init__(self):
        super().__init__()
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._xml_has_emitted_json_start = False
        self._xml_json_closed = False
        self._xml_emitted_param_names: set[str] = set()
        self._payload_parts: list[str] = []
        self._coerced_args: dict[str, Any] = {}
        self._in_progress_value = False
        self._xml_streaming_param_name: str | None = None
        self._xml_streaming_param_emitted_raw_len = 0
        self._xml_streaming_param_schema_type: str | None = None
        self._xml_streaming_param_quote_opened = False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        self._function_param_schemas = self._build_function_param_schemas(request)
        return super().adjust_request(request)

    def start_tool_call(self) -> None:
        super().start_tool_call()
        self._reset_xml_stream_state()

    def finish_tool_call(self) -> None:
        super().finish_tool_call()
        self._reset_xml_stream_state()

    def _reset_xml_stream_state(self) -> None:
        self._xml_has_emitted_json_start = False
        self._xml_json_closed = False
        self._xml_emitted_param_names.clear()
        self._payload_parts.clear()
        self._coerced_args.clear()
        self._in_progress_value = False
        self._reset_streaming_param()
        self._reset_incremental_state()

    def _reset_incremental_state(self) -> None:
        """Reset subclass-specific incremental parse state."""

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._payload_parts.append(added_text)
        payload = ''.join(self._payload_parts)
        func_name, raw_args_dict, is_closed = self._extract_incremental_state(payload, final=final)
        args_dict = self._get_coerced_args(func_name, raw_args_dict)
        streaming_param_name, streaming_raw_value, is_param_closed = self._extract_streaming_param(payload)

        out: list[DeltaToolCall] = []
        if func_name and not self._name_emitted:
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type='function',
                    function=DeltaFunctionCall(name=func_name),
                ))
            self._name_emitted = True

        should_close = is_closed or (final and self._close_json_on_final())

        json_fragments: list[str] = []
        self._append_completed_streaming_param_close(json_fragments, args_dict)
        has_streaming_param = self._prepare_streaming_param(
            func_name,
            streaming_param_name,
            is_param_closed,
        )
        has_streaming_update = self._has_streaming_param_update(streaming_raw_value, is_param_closed)
        if not self._xml_has_emitted_json_start and (args_dict or has_streaming_update or should_close):
            json_fragments.append('{')
            self._xml_has_emitted_json_start = True

        for key, value in args_dict.items():
            if key in self._xml_emitted_param_names:
                continue
            prefix = ', ' if len(self._xml_emitted_param_names) > 0 else ''
            json_fragments.append(f'{prefix}\"{key}\": {json.dumps(value, ensure_ascii=False)}')
            self._xml_emitted_param_names.add(key)
            if key == self._xml_streaming_param_name:
                self._reset_streaming_param()

        if has_streaming_param and has_streaming_update:
            self._append_streaming_param_fragments(json_fragments, streaming_raw_value, is_param_closed)

        if should_close and self._xml_has_emitted_json_start and not self._xml_json_closed:
            json_fragments.append('}')
            self._xml_json_closed = True

        if json_fragments:
            out.append(
                DeltaToolCall(
                    id=None,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=''.join(json_fragments)),
                ))
        return out

    def _append_completed_streaming_param_close(self, json_fragments: list[str], args_dict: dict[str, Any]) -> None:
        param_name = self._xml_streaming_param_name
        if param_name is None or param_name not in args_dict or param_name not in self._xml_emitted_param_names:
            return
        if self._xml_streaming_param_quote_opened:
            json_fragments.append('"')
        self._reset_streaming_param()

    def _prepare_streaming_param(self,
                                 func_name: str | None,
                                 param_name: str | None,
                                 is_param_closed: bool) -> bool:
        if param_name is None:
            return self._xml_streaming_param_name is not None
        if self._xml_streaming_param_name == param_name:
            return True
        if is_param_closed or param_name in self._xml_emitted_param_names:
            return False

        self._xml_streaming_param_name = param_name
        self._xml_streaming_param_emitted_raw_len = 0
        self._xml_streaming_param_schema_type = self._get_param_schema_type(func_name, param_name)
        self._xml_streaming_param_quote_opened = False
        return True

    def _has_streaming_param_update(self, raw_value: str, is_param_closed: bool) -> bool:
        param_name = self._xml_streaming_param_name
        if param_name is None:
            return False
        if not self._can_stream_raw_param_value(raw_value):
            return False
        if param_name not in self._xml_emitted_param_names:
            return bool(raw_value) or is_param_closed
        return len(raw_value) > self._xml_streaming_param_emitted_raw_len or is_param_closed

    def _can_stream_raw_param_value(self, raw_value: str) -> bool:
        if self._xml_streaming_param_schema_type not in (None, 'string'):
            return False
        stripped_value = raw_value.lstrip()
        return bool(stripped_value) and not stripped_value.startswith('"')

    def _append_streaming_param_fragments(self,
                                          json_fragments: list[str],
                                          raw_value: str,
                                          is_param_closed: bool) -> None:
        param_name = self._xml_streaming_param_name
        if param_name is None:
            return

        is_string = self._xml_streaming_param_schema_type in (None, 'string')
        if param_name not in self._xml_emitted_param_names:
            prefix = ', ' if len(self._xml_emitted_param_names) > 0 else ''
            json_fragments.append(f'{prefix}\"{param_name}\": ')
            if is_string:
                json_fragments.append('"')
                self._xml_streaming_param_quote_opened = True
            self._xml_emitted_param_names.add(param_name)

        if len(raw_value) > self._xml_streaming_param_emitted_raw_len:
            diff = raw_value[self._xml_streaming_param_emitted_raw_len:]
            if is_string:
                diff = json.dumps(diff, ensure_ascii=False)[1:-1]
            json_fragments.append(diff)
            self._xml_streaming_param_emitted_raw_len = len(raw_value)

        if is_param_closed:
            if is_string and self._xml_streaming_param_quote_opened:
                json_fragments.append('"')
            self._reset_streaming_param()

    def _reset_streaming_param(self) -> None:
        self._xml_streaming_param_name = None
        self._xml_streaming_param_emitted_raw_len = 0
        self._xml_streaming_param_schema_type = None
        self._xml_streaming_param_quote_opened = False

    def _get_param_schema_type(self, func_name: str | None, param_name: str) -> str | None:
        if func_name is None:
            return None
        param_schema = self._function_param_schemas.get(func_name, {}).get(param_name)
        if not isinstance(param_schema, dict):
            return None
        return self._resolve_schema_type(param_schema)

    @staticmethod
    def _strip_partial_xml_close_suffix(raw_value: str, close_tag: str) -> str:
        max_len = min(len(raw_value), len(close_tag) - 1)
        for suffix_len in range(max_len, 0, -1):
            if close_tag.startswith(raw_value[-suffix_len:]):
                return raw_value[:-suffix_len]
        return raw_value

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
                          raw_args_dict: dict[str, Any],
                          *,
                          use_cache: bool = True) -> dict[str, Any]:
        if not func_name or not raw_args_dict:
            return raw_args_dict
        param_schemas = self._function_param_schemas.get(func_name, {})
        if not param_schemas:
            return raw_args_dict

        coerced = dict(self._coerced_args) if use_cache else {}
        for key, value in raw_args_dict.items():
            if use_cache and key in self._coerced_args:
                continue
            if not isinstance(value, str):
                coerced_value = value
            else:
                schema = param_schemas.get(key)
                if not isinstance(schema, dict):
                    coerced_value = value
                else:
                    schema_type = self._resolve_schema_type(schema)
                    coerced_value = self._coerce_value(value, schema_type)
            if use_cache:
                self._coerced_args[key] = coerced_value
            coerced[key] = coerced_value
        return coerced

    def _close_json_on_final(self) -> bool:
        return True

    def _extract_streaming_param(self, payload: str) -> tuple[str | None, str, bool]:
        return None, '', False

    def _extract_incremental_state(self,
                                 payload: str,
                                 final: bool = False) -> tuple[str | None, dict[str, Any], bool]:
        """Parse accumulated inner tool payload and return the current
        snapshot.

        Subclasses update their incremental state from ``payload`` and return
        ``(func_name, raw_args_dict, is_closed)`` for delta emission.
        """
        raise NotImplementedError
