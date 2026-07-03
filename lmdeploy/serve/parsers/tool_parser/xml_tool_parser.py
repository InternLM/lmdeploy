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
    arg_value: str
    payload_closed: bool


class XmlToolParser(ToolParser):
    """Base class for XML-like tool parsers.

    Subclasses only need to implement XML payload extraction.
    """

    def __init__(self):
        super().__init__()
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._xml_has_emitted_json_start = False
        self._xml_json_closed = False
        self._xml_emitted_arg_names: set[str] = set()
        self._payload_parts: list[str] = []
        self._coerced_args: dict[str, Any] = {}
        self._in_progress_value = False
        self._xml_arg_name: str | None = None
        self._xml_arg_emitted_len = 0
        self._xml_arg_schema_type: str | None = None
        self._xml_arg_quote_opened = False

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
        self._xml_emitted_arg_names.clear()
        self._payload_parts.clear()
        self._coerced_args.clear()
        self._in_progress_value = False
        self._reset_arg()
        self._reset_incremental_state()

    def _reset_incremental_state(self) -> None:
        """Reset subclass-specific incremental parse state."""

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._payload_parts.append(added_text)
        payload = ''.join(self._payload_parts)
        snapshot = self._parse_payload(payload, final=final)

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

        if should_close and not self._xml_has_emitted_json_start:
            json_fragments.append('{')
            self._xml_has_emitted_json_start = True
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

    def _append_json_start(self, json_fragments: list[str]) -> None:
        if not self._xml_has_emitted_json_start:
            json_fragments.append('{')
            self._xml_has_emitted_json_start = True

    def _append_finished_arg(self, json_fragments: list[str], completed_args: dict[str, Any]) -> None:
        arg_name = self._xml_arg_name
        if arg_name is None or arg_name not in completed_args or arg_name not in self._xml_emitted_arg_names:
            return
        value = completed_args[arg_name]
        if self._xml_arg_quote_opened:
            if isinstance(value, str) and len(value) > self._xml_arg_emitted_len:
                diff = value[self._xml_arg_emitted_len:]
                json_fragments.append(json.dumps(diff, ensure_ascii=False)[1:-1])
            json_fragments.append('"')
        else:
            value_text = json.dumps(value, ensure_ascii=False)
            if len(value_text) > self._xml_arg_emitted_len:
                json_fragments.append(value_text[self._xml_arg_emitted_len:])
        self._reset_arg()

    def _append_completed_args(self, json_fragments: list[str], completed_args: dict[str, Any]) -> None:
        for key, value in completed_args.items():
            if key in self._xml_emitted_arg_names:
                continue
            self._append_json_start(json_fragments)
            prefix = ', ' if len(self._xml_emitted_arg_names) > 0 else ''
            json_fragments.append(f'{prefix}"{key}": {json.dumps(value, ensure_ascii=False)}')
            self._xml_emitted_arg_names.add(key)

    def _append_open_arg(self, json_fragments: list[str], snapshot: XmlToolSnapshot) -> None:
        if snapshot.arg_name is None:
            return

        if self._xml_arg_name == snapshot.arg_name:
            value_text = self._streaming_value_text(snapshot.arg_value, self._xml_arg_schema_type)
            if value_text is None or len(value_text) <= self._xml_arg_emitted_len:
                return
            diff = value_text[self._xml_arg_emitted_len:]
            json_fragments.append(json.dumps(diff, ensure_ascii=False)[1:-1])
            self._xml_arg_emitted_len = len(value_text)
            return

        if snapshot.arg_name in self._xml_emitted_arg_names:
            return

        schema_type = self._get_param_schema_type(snapshot.func_name, snapshot.arg_name)
        value_text = self._streaming_value_text(snapshot.arg_value, schema_type)
        if value_text is None or value_text == '':
            return

        self._append_json_start(json_fragments)
        prefix = ', ' if len(self._xml_emitted_arg_names) > 0 else ''
        json_fragments.append(f'{prefix}"{snapshot.arg_name}": "')
        diff = json.dumps(value_text, ensure_ascii=False)[1:-1]
        json_fragments.append(diff)
        self._xml_emitted_arg_names.add(snapshot.arg_name)
        self._xml_arg_name = snapshot.arg_name
        self._xml_arg_emitted_len = len(value_text)
        self._xml_arg_schema_type = schema_type
        self._xml_arg_quote_opened = True

    @staticmethod
    def _streaming_value_text(raw_value: str, schema_type: str | None) -> str | None:
        stripped_value = raw_value.lstrip()
        if not stripped_value:
            return None
        if stripped_value.startswith('"'):
            return None
        if schema_type == 'string' and raw_value != raw_value.strip():
            return None
        if schema_type in (None, 'string'):
            return raw_value
        return None

    def _reset_arg(self) -> None:
        self._xml_arg_name = None
        self._xml_arg_emitted_len = 0
        self._xml_arg_schema_type = None
        self._xml_arg_quote_opened = False

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

    def _parse_payload(self, payload: str, *, final: bool) -> XmlToolSnapshot:
        """Parse accumulated inner tool payload into raw XML tool state."""
        raise NotImplementedError
