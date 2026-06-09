# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

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

    arg_key_start_token = '<arg_key>'
    arg_key_end_token = '</arg_key>'
    arg_value_start_token = '<arg_value>'
    arg_value_end_token = '</arg_value>'
    _complete_payload_pattern = re.compile(
        r'^\s*[^\s<]+(?:\s*<arg_key>[^<]+</arg_key>\s*<arg_value>.*?</arg_value>)*\s*$',
        re.DOTALL,
    )

    def _value_close_token(self) -> str:
        return self.arg_value_end_token

    def _reset_incremental_state(self) -> None:
        self._glm_func_name: str | None = None
        self._glm_args: dict[str, str] = {}
        self._glm_args_scan_pos = 0
        self._glm_open_arg_key: str | None = None
        self._glm_value_start = -1

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'xml'

    def _extract_incremental_state(self,
                                   payload: str,
                                   final: bool = False,
                                   added_text: str = '') -> tuple[str | None, dict[str, str], bool]:
        payload = payload.strip()
        if not payload:
            return None, {}, False

        args_start_idx = payload.find(self.arg_key_start_token)
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            if func_name:
                self._glm_func_name = func_name
            self._parse_args_incremental(payload[args_start_idx:])
        elif final:
            func_name = payload.strip()
            if func_name:
                self._glm_func_name = func_name

        return self._glm_func_name, dict(self._glm_args), False

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict = self._parse_payload(payload, final=True)
        if not func_name:
            return None
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
        return ToolCall(function=FunctionCall(name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False)))

    def _validate_tool_payload(self, payload: str) -> bool:
        return bool(self._complete_payload_pattern.fullmatch(payload))

    def _complete_open_arg_if_ready(self, args_text: str) -> bool:
        if self._glm_value_start < 0 or not self._glm_open_arg_key:
            return False
        value_end = args_text.find(self.arg_value_end_token, self._glm_value_start)
        if value_end < 0:
            self._in_progress_value = True
            return False
        self._glm_args[self._glm_open_arg_key] = args_text[self._glm_value_start:value_end]
        self._glm_args_scan_pos = value_end + len(self.arg_value_end_token)
        self._glm_open_arg_key = None
        self._glm_value_start = -1
        self._in_progress_value = False
        return True

    def _parse_args_incremental(self, args_text: str) -> None:
        if self._glm_value_start >= 0:
            if not self._complete_open_arg_if_ready(args_text):
                return

        search_idx = 0
        while True:
            key_start = args_text.find(self.arg_key_start_token, search_idx)
            if key_start < 0:
                self._glm_args_scan_pos = len(args_text)
                self._in_progress_value = False
                return

            key_content_start = key_start + len(self.arg_key_start_token)
            key_end = args_text.find(self.arg_key_end_token, key_content_start)
            if key_end < 0:
                self._glm_args_scan_pos = key_start
                self._in_progress_value = True
                return

            key = args_text[key_content_start:key_end].strip()
            value_start = args_text.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
            if value_start < 0:
                self._glm_args_scan_pos = key_start
                self._in_progress_value = True
                return

            value_content_start = value_start + len(self.arg_value_start_token)
            value_end = args_text.find(self.arg_value_end_token, value_content_start)
            if value_end < 0:
                self._glm_open_arg_key = key
                self._glm_value_start = value_content_start
                self._glm_args_scan_pos = value_content_start
                self._in_progress_value = True
                return

            if key in self._glm_args:
                search_idx = value_end + len(self.arg_value_end_token)
                self._glm_args_scan_pos = search_idx
                continue

            if key:
                self._glm_args[key] = args_text[value_content_start:value_end]
            search_idx = value_end + len(self.arg_value_end_token)
            self._glm_args_scan_pos = search_idx
            self._in_progress_value = False

    def _parse_payload(self, payload: str, *, final: bool = False) -> tuple[str | None, dict[str, str]]:
        payload = payload.strip()
        if not payload:
            return None, {}

        args_start_idx = payload.find(self.arg_key_start_token)
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            args_text = payload[args_start_idx:]
        else:
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
