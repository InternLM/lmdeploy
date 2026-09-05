# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import Any

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
)

from .tool_parser import ToolParser


class XmlToolParser(ToolParser):
    """Base class for XML-like tool parsers.

    Subclasses only need to implement XML payload extraction.
    """

    def __init__(self):
        super().__init__()
        self._xml_has_emitted_json_start = False
        self._xml_json_closed = False
        self._xml_emitted_param_names: set[str] = set()
        self._payload_parts: list[str] = []
        self._coerced_args: dict[str, Any] = {}
        self._in_progress_value = False

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

    @staticmethod
    def _coerce_value(raw_value: str, schema_type: str) -> Any:
        """Convert direct JSON types without constructing a schema
        validator."""
        raw_value = raw_value.strip()
        if schema_type == 'string':
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
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value
        if schema_type == 'integer':
            return parsed if type(parsed) is int or isinstance(parsed, float) and parsed.is_integer() else raw_value
        if schema_type == 'number':
            return parsed if type(parsed) in (int, float) else raw_value
        if schema_type == 'array':
            return parsed if isinstance(parsed, list) else raw_value
        if schema_type == 'object':
            return parsed if isinstance(parsed, dict) else raw_value
        return raw_value

    def _coerce_schema_value(self, raw_value: str, schema: dict | bool, func_name: str) -> Any:
        """Convert complex-schema arguments using the function's validator."""
        validator = self._get_function_validator(func_name)
        raw_value = raw_value.strip()
        if not any(validator.descend(raw_value, schema)):
            return raw_value
        json_value = raw_value.lower() if raw_value.lower() in ('true', 'false', 'null') else raw_value
        try:
            parsed = json.loads(json_value)
        except json.JSONDecodeError:
            return raw_value
        return parsed if not any(validator.descend(parsed, schema)) else raw_value

    def _get_coerced_args(self,
                          func_name: str | None,
                          raw_args_dict: dict[str, Any],
                          *,
                          use_cache: bool = True) -> dict[str, Any]:
        if not func_name or not raw_args_dict:
            return raw_args_dict
        root_schema = self._function_schemas.get(func_name)
        if not isinstance(root_schema, dict):
            return raw_args_dict
        param_schemas = root_schema.get('properties')
        if not isinstance(param_schemas, dict):
            return raw_args_dict

        coerced = dict(self._coerced_args) if use_cache else {}
        for key, value in raw_args_dict.items():
            if use_cache and key in self._coerced_args:
                continue
            if not isinstance(value, str):
                coerced_value = value
            else:
                schema = param_schemas.get(key)
                schema_type = schema.get('type') if isinstance(schema, dict) else None
                if isinstance(schema_type, str):
                    coerced_value = self._coerce_value(value, schema_type)
                elif schema is not None:
                    coerced_value = self._coerce_schema_value(value, schema, func_name)
                else:
                    coerced_value = value
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
