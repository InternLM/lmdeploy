# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser, XmlToolSnapshot  # type: ignore[reportMissingImports]


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

    def _reset_incremental_state(self) -> None:
        self._func_name: str | None = None
        self._args: dict[str, str] = {}
        self._arg_name: str | None = None
        self._value_start = -1
        self._scan_pos = 0

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'xml'

    def _parse_payload(self, payload: str, *, final: bool) -> XmlToolSnapshot:
        payload = payload.strip()
        if not payload:
            return XmlToolSnapshot(self._func_name, dict(self._args), None, '', False)

        args_start_idx = payload.find(self.arg_key_start_token)
        args_text = ''
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            if func_name:
                self._func_name = func_name
            args_text = payload[args_start_idx:]
            self._parse_args(args_text)
        elif final:
            func_name = payload.strip()
            if func_name:
                self._func_name = func_name

        arg_name, arg_value = self._open_arg(args_text)
        return XmlToolSnapshot(self._func_name, dict(self._args), arg_name, arg_value, False)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict = self._extract_complete_args(payload)
        if not func_name:
            return None
        args_dict = self._get_coerced_args(func_name, raw_args_dict, use_cache=False)
        return ToolCall(function=FunctionCall(name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False)))

    def _validate_tool_payload(self, payload: str) -> bool:
        return bool(self._complete_payload_pattern.fullmatch(payload))

    def _complete_open_arg(self, args_text: str) -> bool:
        if self._value_start < 0 or not self._arg_name:
            return False
        value_end = args_text.find(self.arg_value_end_token, self._value_start)
        if value_end < 0:
            self._in_progress_value = True
            return False
        self._args[self._arg_name] = args_text[self._value_start:value_end]
        self._arg_name = None
        self._value_start = -1
        self._in_progress_value = False
        self._scan_pos = value_end + len(self.arg_value_end_token)
        return True

    def _parse_args(self, args_text: str) -> None:
        if self._value_start >= 0 and not self._complete_open_arg(args_text):
            return

        while True:
            key_start = args_text.find(self.arg_key_start_token, self._scan_pos)
            if key_start < 0:
                self._in_progress_value = False
                return

            key_content_start = key_start + len(self.arg_key_start_token)
            key_end = args_text.find(self.arg_key_end_token, key_content_start)
            if key_end < 0:
                self._in_progress_value = True
                return

            arg_name = args_text[key_content_start:key_end].strip()
            value_start = args_text.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
            if value_start < 0:
                self._in_progress_value = True
                return

            value_content_start = value_start + len(self.arg_value_start_token)
            value_end = args_text.find(self.arg_value_end_token, value_content_start)
            if value_end < 0:
                self._arg_name = arg_name
                self._value_start = value_content_start
                self._in_progress_value = True
                return

            next_pos = value_end + len(self.arg_value_end_token)
            if arg_name and arg_name not in self._args:
                self._args[arg_name] = args_text[value_content_start:value_end]
            self._scan_pos = next_pos
            self._in_progress_value = False

    def _open_arg(self, args_text: str) -> tuple[str | None, str]:
        if self._arg_name is None or self._value_start < 0:
            return None, ''
        raw_value = self._strip_partial_xml_close_suffix(args_text[self._value_start:], self.arg_value_end_token)
        return self._arg_name, raw_value

    def _extract_complete_args(self, payload: str) -> tuple[str | None, dict[str, str]]:
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
