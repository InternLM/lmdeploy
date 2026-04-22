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
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import (
        ChatCompletionRequest,
    )


@ToolParserManager.register_module(['glm47'])
class Glm47ToolParser(ToolParser):
    """Tool parser for GLM-4.7 XML-like tool-call payloads.

    Expected format inside ``<tool_call>...</tool_call>``:
    ``function_name<arg_key>k</arg_key><arg_value>v</arg_value>...``
    """

    tool_start_token = '<tool_call>'
    tool_end_token = '</tool_call>'
    arg_key_start_token = '<arg_key>'
    arg_key_end_token = '</arg_key>'
    arg_value_start_token = '<arg_value>'
    arg_value_end_token = '</arg_value>'

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        self._function_param_schemas = self._build_function_param_schemas(request)
        # Preserve tool-call tags in decoded text so ResponseParser can detect
        # tool blocks and hand them to this parser.
        if request.tools and request.tool_choice != 'none':
            request.skip_special_tokens = False
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'xml'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._tool_payload += added_text
        func_name, raw_args_dict = self._parse_payload(self._tool_payload)
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)

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

        if not final:
            return out

        args_json = json.dumps(args_dict, ensure_ascii=False)
        if len(args_json) > self._args_emitted_len:
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=args_json[self._args_emitted_len:]),
                ))
            self._args_emitted_len = len(args_json)
        return out

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict = self._parse_payload(payload)
        if not func_name:
            return None
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
        return ToolCall(function=FunctionCall(name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False)))

    def _parse_payload(self, payload: str) -> tuple[str | None, dict[str, str]]:
        payload = payload.strip()
        if not payload:
            return None, {}

        args_start_idx = payload.find(self.arg_key_start_token)
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            args_text = payload[args_start_idx:]
        else:
            func_name = payload.strip()
            args_text = ''
        if not func_name:
            return None, {}

        args_dict: dict[str, str] = {}
        search_idx = 0
        while True:
            key_start = args_text.find(self.arg_key_start_token, search_idx)
            if key_start < 0:
                break
            key_content_start = key_start + len(self.arg_key_start_token)
            key_end = args_text.find(self.arg_key_end_token, key_content_start)
            if key_end < 0:
                break
            key = args_text[key_content_start:key_end].strip()
            value_start = args_text.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
            if value_start < 0:
                break
            value_content_start = value_start + len(self.arg_value_start_token)
            value_end = args_text.find(self.arg_value_end_token, value_content_start)
            if value_end < 0:
                break
            if key:
                args_dict[key] = args_text[value_content_start:value_end]
            search_idx = value_end + len(self.arg_value_end_token)
        return func_name, args_dict
