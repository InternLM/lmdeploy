# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParser, ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


def _parse_tool_call_arguments_dict(arguments: Any) -> dict[str, Any] | None:
    """Return dict-like tool arguments for Qwen3Coder request normalization."""
    if not isinstance(arguments, str):
        return None

    try:
        parsed_arguments = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed_arguments, dict):
        return parsed_arguments
    return None


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(ToolParser):
    """Tool parser for Qwen3Coder XML tool-call payloads."""

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'
        self.func_prefix = '<function='
        self.func_end_token = '</function>'
        self.param_prefix = '<parameter='
        self.param_end_token = '</parameter>'
        self.coder_has_emitted_name = False
        self.coder_has_emitted_json_start = False
        self.coder_json_closed = False
        self.coder_emitted_param_names: set[str] = set()
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}

    def _normalize_request_messages(self, messages: list[dict]) -> list[dict] | None:
        """Return a render-safe copy of request messages when needed."""
        normalized_messages = None

        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            tool_calls = message.get('tool_calls')
            if not isinstance(tool_calls, list):
                continue

            normalized_tool_calls = None
            for tool_idx, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get('function')
                if not isinstance(function, dict) or isinstance(function.get('arguments'), dict):
                    continue

                parsed_arguments = _parse_tool_call_arguments_dict(function.get('arguments'))
                if parsed_arguments is None:
                    continue

                if normalized_messages is None:
                    normalized_messages = list(messages)
                if normalized_tool_calls is None:
                    normalized_tool_calls = list(tool_calls)
                    normalized_message = dict(message)
                    normalized_message['tool_calls'] = normalized_tool_calls
                    normalized_messages[msg_idx] = normalized_message

                normalized_function = dict(function)
                normalized_function['arguments'] = parsed_arguments

                normalized_tool_call = dict(tool_call)
                normalized_tool_call['function'] = normalized_function
                normalized_tool_calls[tool_idx] = normalized_tool_call

        return normalized_messages

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'xml'

    def start_tool_call(self) -> None:
        super().start_tool_call()
        self.coder_has_emitted_name = False
        self.coder_has_emitted_json_start = False
        self.coder_json_closed = False
        self.coder_emitted_param_names.clear()

    def finish_tool_call(self) -> None:
        super().finish_tool_call()
        self.coder_has_emitted_name = False
        self.coder_has_emitted_json_start = False
        self.coder_json_closed = False
        self.coder_emitted_param_names.clear()

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental XML tool payload."""
        self._tool_payload += added_text
        func_name, raw_args_dict, is_func_closed = self._extract_params(self._tool_payload)
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)

        out: list[DeltaToolCall] = []
        if func_name and not self.coder_has_emitted_name:
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type='function',
                    function=DeltaFunctionCall(name=func_name),
                ))
            self.coder_has_emitted_name = True

        json_fragments: list[str] = []
        if not self.coder_has_emitted_json_start and (args_dict or is_func_closed):
            json_fragments.append('{')
            self.coder_has_emitted_json_start = True

        for k, v in args_dict.items():
            if k in self.coder_emitted_param_names:
                continue
            prefix = ', ' if len(self.coder_emitted_param_names) > 0 else ''
            json_fragments.append(f'{prefix}\"{k}\": {json.dumps(v, ensure_ascii=False)}')
            self.coder_emitted_param_names.add(k)

        if is_func_closed and self.coder_has_emitted_json_start and not self.coder_json_closed:
            json_fragments.append('}')
            self.coder_json_closed = True

        if json_fragments:
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=''.join(json_fragments)),
                ))
        return out

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict, _ = self._extract_params(payload)
        if not func_name:
            return None
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
        args_json = json.dumps(args_dict, ensure_ascii=False) if args_dict else '{}'
        return ToolCall(function=FunctionCall(name=func_name, arguments=args_json))

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        self._function_param_schemas = self._build_function_param_schemas(request)
        messages = request.messages
        if not isinstance(messages, list):
            return request

        normalized_messages = self._normalize_request_messages(messages)
        if normalized_messages is None:
            return request
        return request.model_copy(update={'messages': normalized_messages})

    def _build_function_param_schemas(self, request: ChatCompletionRequest) -> dict[str, dict[str, dict[str, Any]]]:
        """Build function->parameter schema map from request tools."""
        tools = request.tools
        if not isinstance(tools, list):
            return {}

        out: dict[str, dict[str, dict[str, Any]]] = {}
        for tool in tools:
            function = getattr(tool, 'function', None)
            if function is None:
                continue
            function_name = getattr(function, 'name', None)
            parameters = getattr(function, 'parameters', None)
            if not isinstance(function_name, str) or not function_name:
                continue
            if not isinstance(parameters, dict):
                continue
            properties = parameters.get('properties')
            if not isinstance(properties, dict):
                continue

            param_schemas = {name: schema for name, schema in properties.items() if isinstance(schema, dict)}
            if param_schemas:
                out[function_name] = param_schemas
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
        """Coerce one XML parameter string by schema type."""
        if schema_type is None or schema_type == 'string':
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

    def _coerce_args_by_schema(self, func_name: str | None, args_dict: dict[str, Any]) -> dict[str, Any]:
        if not func_name or not args_dict:
            return args_dict
        param_schemas = self._function_param_schemas.get(func_name)
        if not param_schemas:
            return args_dict

        coerced: dict[str, Any] = {}
        for key, value in args_dict.items():
            if not isinstance(value, str):
                coerced[key] = value
                continue
            schema = param_schemas.get(key)
            if not isinstance(schema, dict):
                coerced[key] = value
                continue
            schema_type = self._resolve_schema_type(schema)
            coerced[key] = self._coerce_value(value, schema_type)
        return coerced

    def _extract_params(self, content: str) -> tuple[str | None, dict[str, Any], bool]:
        """Extract function name, parameter map, and close status from XML."""
        content = content.replace(self.tool_start_token, '').replace(self.tool_end_token, '').strip()

        func_name = None
        func_start = content.find(self.func_prefix)
        if func_start != -1:
            name_start = func_start + len(self.func_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if terminators:
                func_name = content[name_start:min(terminators)].strip()

        args_dict = {}
        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                break

            name_start = param_start + len(self.param_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if not terminators:
                break

            name_end = min(terminators)
            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_end_token, val_start)
            if val_end == -1:
                break

            param_val_str = content[val_start:val_end].strip()

            # Qwen3Coder XML payloads do not carry explicit type metadata.
            # Keep parameter values as strings to avoid implicit type coercion
            # (e.g., zip codes like 77004 being parsed into integers).
            try:
                parsed_val = json.loads(param_val_str)
                val = parsed_val if isinstance(parsed_val, str) else param_val_str
            except json.JSONDecodeError:
                val = param_val_str
            args_dict[param_name] = val
            search_idx = val_end + len(self.param_end_token)

        is_func_closed = self.func_end_token in content
        return func_name, args_dict, is_func_closed
