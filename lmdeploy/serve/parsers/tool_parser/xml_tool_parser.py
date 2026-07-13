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
        self._reset_incremental_state()

    def _reset_incremental_state(self) -> None:
        """Reset subclass-specific incremental parse state."""

    def _should_buffer_value_chunk(self, added_text: str, final: bool) -> bool:
        """Fast-path plain value fragments that cannot close an XML tag."""
        if final or not self._in_progress_value:
            return False
        return not any(ch in added_text for ch in '<>/')

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._payload_parts.append(added_text)
        if self._should_buffer_value_chunk(added_text, final):
            return []

        func_name, raw_args_dict, is_closed = self._extract_incremental_state(
            ''.join(self._payload_parts),
            final=final,
        )
        args_dict = self._get_coerced_args(func_name, raw_args_dict)

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
        if not self._xml_has_emitted_json_start and (args_dict or should_close):
            json_fragments.append('{')
            self._xml_has_emitted_json_start = True

        for key, value in args_dict.items():
            if key in self._xml_emitted_param_names:
                continue
            prefix = ', ' if len(self._xml_emitted_param_names) > 0 else ''
            json_fragments.append(f'{prefix}\"{key}\": {json.dumps(value, ensure_ascii=False)}')
            self._xml_emitted_param_names.add(key)

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

    def _extract_incremental_state(self,
                                 payload: str,
                                 final: bool = False) -> tuple[str | None, dict[str, Any], bool]:
        """Parse accumulated inner tool payload and return the current
        snapshot.

        Subclasses update their incremental state from ``payload`` and return
        ``(func_name, raw_args_dict, is_closed)`` for delta emission.
        """
        raise NotImplementedError
