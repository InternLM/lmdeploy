# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser  # type: ignore[reportMissingImports]


@ToolParserManager.register_module(['glm47'])
class Glm47ToolParser(XmlToolParser):
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

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'xml'

    def _extract_incremental_state(self, payload: str, final: bool = False) -> tuple[str | None, dict[str, str], bool]:
        func_name, args_dict = self._parse_payload(payload, final=final)
        return func_name, args_dict, False

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict = self._parse_payload(payload, final=True)
        if not func_name:
            return None
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
        return ToolCall(function=FunctionCall(name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False)))

    def _parse_payload(self, payload: str, *, final: bool = False) -> tuple[str | None, dict[str, str]]:
        payload = payload.strip()
        if not payload:
            return None, {}

        args_start_idx = payload.find(self.arg_key_start_token)
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            args_text = payload[args_start_idx:]
        else:
            # Do not treat a growing prefix as the callee until ``<arg_key>`` or
            # end-of-payload (``final``); the latter covers zero-argument tools.
            if not final:
                return None, {}
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
