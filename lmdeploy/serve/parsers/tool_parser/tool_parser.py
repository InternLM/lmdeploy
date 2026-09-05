# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/tool_parsers
from __future__ import annotations

import json
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

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


def dump_tools(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """Dump tools to a list of dicts to fit jinja chat template."""
    from lmdeploy.serve.openai.protocol import AllowedToolChoice

    if isinstance(request.tool_choice, AllowedToolChoice):
        allowed_names: set[str] = set()
        allowed_functions: list[dict] = []
        for t in request.tool_choice.allowed_tools.tools:
            func = t.get('function', {})
            if isinstance(func, dict) and 'name' in func:
                allowed_names.add(func['name'])
                allowed_functions.append(func)

        if not request.tools:
            return request.model_copy(update={'tools': allowed_functions or None})

        request_tool_names = {item.function.name for item in request.tools}
        missing = sorted(allowed_names - request_tool_names)
        if missing:
            raise ValueError(f'Allowed tool(s) not found in request.tools: {missing}')

        tools = [item.function.model_dump() for item in request.tools if item.function.name in allowed_names]
        return request.model_copy(update={'tools': tools})

    if not request.tools:
        return request.model_copy(update={'tools': None})

    if not isinstance(request.tool_choice, str):
        tools = [
            item.function.model_dump() for item in request.tools
            if item.function.name == request.tool_choice.function.name
        ]
    else:
        tools = [item.function.model_dump() for item in request.tools]
    return request.model_copy(update={'tools': tools})


@lru_cache(maxsize=128)
def _check_required_tool_grammar(serialized_format: str) -> None:
    """Check grammar compilability, caching only successful preflight
    results."""
    import xgrammar as xgr

    try:
        # Construct the grammar before the request reaches the shared engine loop.
        xgr.Grammar.from_structural_tag(serialized_format)
    except RuntimeError as err:
        raise ValueError(f'Unsupported required-tool grammar: {err}') from err


class ToolParser:
    """Base class for model-specific tool parsers."""

    validate_tool_names: ClassVar[bool] = False
    # XGrammar builtin structural-tag keys for required tool calls.
    structural_tag_model: str | None = None
    reasoning_structural_tag_model: str | None = None

    def __init__(self):
        self._function_schemas: dict[str, dict | bool] = {}
        self._function_validators: dict[str, Any] = {}
        self._tool_payload: str = ''
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False
        self._args_emitted_len: int = 0
        self._allowed_tool_names: set[str] = set()
        self._stream_tool_indices: dict[int, int | None] = {}
        self._next_stream_tool_index = 0

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request payload before rendering, if needed."""
        self._function_schemas.clear()
        self._function_validators.clear()
        # Only XML coercion and required-call validation consume parameter schemas.
        if request.tools and (request.tool_choice == 'required' or self.get_tool_payload_format() == 'xml'):
            for tool in request.tools:
                function = tool.function
                self._function_schemas[function.name] = function.parameters if function.parameters is not None else True
        request = dump_tools(request)
        if self.validate_tool_names:
            self._allowed_tool_names = self._get_allowed_tool_names(request)
        return request

    def prepare_required_tools(self, tools: list[Any], *, reasoning: bool) -> dict[str, Any]:
        """Prepare schema validators and a compilable required-tool grammar."""
        from .schema import create_schema_validator

        structural_tag_model = (self.reasoning_structural_tag_model
                                if reasoning and self.reasoning_structural_tag_model is not None
                                else self.structural_tag_model)
        if structural_tag_model is None:
            raise ValueError(f'Tool parser {type(self).__name__!r} does not support `tool_choice="required"`.')

        self._function_validators = {
            name: create_schema_validator(schema) for name, schema in self._function_schemas.items()
        }
        import xgrammar as xgr

        response_format = xgr.get_model_structural_tag(
            structural_tag_model,
            [tool.model_dump() for tool in tools],
            tool_choice='required',
            reasoning=reasoning,
        ).model_dump(mode='json')
        _check_required_tool_grammar(json.dumps(response_format, separators=(',', ':')))
        return response_format

    def _get_function_validator(self, name: str) -> Any:
        """Reuse a prepared validator or create one lazily for auto
        coercion."""
        validator = self._function_validators.get(name)
        if validator is None:
            from jsonschema.validators import validator_for
            from referencing import Registry

            schema = self._function_schemas[name]
            validator = validator_for(schema)(schema, registry=Registry())
            self._function_validators[name] = validator
        return validator

    def validate_tool_calls(self, tool_calls: list[ToolCall] | None) -> bool:
        """Validate required calls against the prepared tool schemas."""
        if not tool_calls:
            return False
        for call in tool_calls:
            validator = self._function_validators.get(call.function.name)
            if validator is None:
                return False
            try:
                arguments = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                return False
            if not validator.is_valid(arguments):
                return False
        return True

    @staticmethod
    def _get_allowed_tool_names(request: ChatCompletionRequest) -> set[str]:
        """Return names exposed by the effective request tool list."""
        if request.tools is None:
            return set()

        names: set[str] = set()
        for tool in request.tools:
            if isinstance(tool, dict):
                function = tool.get('function', tool)
                name = function.get('name') if isinstance(function, dict) else None
            else:
                name = tool.function.name
            if isinstance(name, str):
                names.add(name)
        return names

    def is_valid_tool_name(self, name: str) -> bool:
        """Return whether a name is allowed by the effective request tools."""
        if not self.validate_tool_names:
            return True
        return name in self._allowed_tool_names

    def filter_tool_call_deltas(self, calls: list[DeltaToolCall]) -> list[DeltaToolCall]:
        """Drop streamed calls whose names are absent from request tools."""
        if not self.validate_tool_names:
            return calls

        filtered: list[DeltaToolCall] = []
        for call in calls:
            function = call.function
            if function is not None and function.name and call.index not in self._stream_tool_indices:
                # Assign accepted calls a contiguous client-visible index.
                # Mark rejected calls as None to filter their later argument deltas.
                if self.is_valid_tool_name(function.name):
                    self._stream_tool_indices[call.index] = self._next_stream_tool_index
                    self._next_stream_tool_index += 1
                else:
                    self._stream_tool_indices[call.index] = None
            visible_index = self._stream_tool_indices.get(call.index)
            if visible_index is not None:
                call.index = visible_index
                filtered.append(call)
        return filtered

    def filter_tool_calls(self, calls: list[ToolCall]) -> list[ToolCall]:
        """Drop complete calls whose names are absent from request tools."""
        return [call for call in calls if self.is_valid_tool_name(call.function.name)]

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

    def parse_tool_call_complete(self, payload: str) -> ToolCall | list[ToolCall] | None:
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
        payload = self._tool_payload.strip()
        if not payload:
            return []

        # After the function name is emitted, arguments are only surfaced at
        # final=True. Skip repeated partial_json_parser.loads on growing payload.
        if self._name_emitted and not final:
            return []

        flags = Allow.ALL if final else Allow.ALL & ~Allow.STR
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
                    id=None,
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
