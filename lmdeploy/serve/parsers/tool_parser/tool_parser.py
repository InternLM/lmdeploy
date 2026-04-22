# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/tool_parsers
from __future__ import annotations

import json
from functools import cached_property
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.parsers.tool_parser'])


class ToolParser:
    """Base class for model-specific tool parsers."""

    def __init__(self, tokenizer: object):
        self.model_tokenizer = tokenizer
        self._tool_payload: str = ''
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False
        self._args_emitted_len: int = 0

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request payload before rendering, if needed."""
        if request.tools is None or request.tool_choice == 'none':
            return request.model_copy(update={'tools': None})
        if not isinstance(request.tool_choice, str):
            tools = [
                item.function.model_dump() for item in request.tools
                if item.function.name == request.tool_choice.function.name
            ]
        else:
            tools = [item.function.model_dump() for item in request.tools]
        return request.model_copy(update={'tools': tools})

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
        param_schemas = getattr(self, '_function_param_schemas', {}).get(func_name, {})
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

    def get_tool_open_tag(self) -> str | None:
        """Return tool opening tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_open_tag has not been implemented!')

    def get_tool_close_tag(self) -> str | None:
        """Return tool closing tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_close_tag has not been implemented!')

    def get_tool_payload_format(self) -> str:
        """Return payload format for tool call body."""
        raise NotImplementedError('ToolParser.get_tool_payload_format has not been implemented!')

    def start_tool_call(self) -> None:
        """Mark start of a tool-call block."""
        self._active_tool_index += 1
        self._active_tool_call_id = f'chatcmpl-tool-{shortuuid.random()}'
        self._name_emitted = False
        self._args_emitted_len = 0
        self._tool_payload = ''

    def finish_tool_call(self) -> None:
        """Mark end of a tool-call block."""
        self._active_tool_call_id = ''
        self._name_emitted = False
        self._args_emitted_len = 0
        self._tool_payload = ''

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental tool payload emitted between tool tags."""
        raise NotImplementedError('ToolParser.decode_tool_incremental has not been implemented!')

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        """Parse one complete tool payload into OpenAI tool call object."""
        raise NotImplementedError('ToolParser.parse_tool_call_complete has not been implemented!')

    def _decode_tool_incremental_json(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._tool_payload += added_text
        payload = self._tool_payload.strip()
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

        args_json = json.dumps(args_obj, ensure_ascii=False)
        if args_json in ('{}', '[]'):
            return out

        # Emit argument text only when the tool payload is complete. This keeps
        # streamed argument chunks valid JSON and avoids malformed intermediate
        # fragments when partial parsers expose transient dict states.
        if final and len(args_json) > self._args_emitted_len:
            diff = args_json[self._args_emitted_len:]
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=diff),
                ))
            self._args_emitted_len = len(args_json)
        return out

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
