# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlParseResult, XmlToolParser


@ToolParserManager.register_module(['glm47'])
class Glm47ToolParser(XmlToolParser):
    """Tool parser for GLM-4.7 XML-like tool-call payloads.

    Expected format inside ``<tool_call>...</tool_call>``:
    ``function_name<arg_key>k</arg_key><arg_value>v</arg_value>...``
    """

    _complete_payload_pattern = re.compile(
        r'^\s*[^\s<]+(?:\s*<arg_key>[^<]+</arg_key>\s*<arg_value>.*?</arg_value>)*\s*$',
        re.DOTALL,
    )

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'xml'

    def _consume_function(self, payload: str, pos: int, final: bool) -> XmlParseResult:
        arg_key_start = payload.find('<arg_key>', pos)
        if arg_key_start >= 0:
            name = payload[pos:arg_key_start].strip()
            return XmlParseResult(
                arg_key_start,
                next_phase='arg_start',
                func_name=name or None,
            )

        remaining = payload[pos:]
        if final and remaining.strip():
            return XmlParseResult(len(payload), func_name=remaining.strip())
        return XmlParseResult(None)

    def _consume_arg_start(self, payload: str, pos: int) -> XmlParseResult:
        arg_key_start = payload.find('<arg_key>', pos)
        if arg_key_start < 0:
            return XmlParseResult(None)

        return XmlParseResult(arg_key_start + len('<arg_key>'), next_phase='arg_name')

    def _consume_arg_name(self, payload: str, pos: int) -> XmlParseResult:
        key_end = payload.find('</arg_key>', pos)
        if key_end < 0:
            return XmlParseResult(None)

        value_start = payload.find('<arg_value>', key_end + len('</arg_key>'))
        if value_start < 0:
            return XmlParseResult(None)

        return XmlParseResult(
            value_start + len('<arg_value>'),
            next_phase='arg_value',
            arg_name=payload[pos:key_end].strip(),
        )

    def _consume_arg_value(self, payload: str, pos: int) -> XmlParseResult:
        value_end = payload.find('</arg_value>', pos)

        if value_end >= 0:
            return XmlParseResult(
                value_end + len('</arg_value>'),
                next_phase='function',
                arg_delta=payload[pos:value_end],
                arg_closed=True,
            )

        # Open value: keep any partial "</arg_value>" suffix buffered instead
        # of emitting it as argument text.
        raw_end = self._trim_partial_close_tag_suffix(payload, pos, '</arg_value>')
        if raw_end == pos:
            return XmlParseResult(None)

        return XmlParseResult(raw_end, arg_delta=payload[pos:raw_end], should_stop=True)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict = self._extract_complete_args(payload)
        if not func_name:
            return None
        args_dict = self._get_coerced_args(func_name, raw_args_dict)
        return ToolCall(function=FunctionCall(name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False)))

    def _validate_tool_payload(self, payload: str) -> bool:
        return bool(self._complete_payload_pattern.fullmatch(payload))

    def _extract_complete_args(self, payload: str) -> tuple[str | None, dict[str, str]]:
        payload = payload.strip()
        if not payload:
            return None, {}

        args_start_idx = payload.find('<arg_key>')
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
            key_start = args_text.find('<arg_key>', search_idx)
            if key_start < 0:
                break
            key_content_start = key_start + len('<arg_key>')
            key_end = args_text.find('</arg_key>', key_content_start)
            if key_end < 0:
                break
            key = args_text[key_content_start:key_end].strip()
            value_start = args_text.find('<arg_value>', key_end + len('</arg_key>'))
            if value_start < 0:
                break
            value_content_start = value_start + len('<arg_value>')
            value_end = args_text.find('</arg_value>', value_content_start)
            if value_end < 0:
                break
            if key:
                args_dict[key] = args_text[value_content_start:value_end]
            search_idx = value_end + len('</arg_value>')
        return func_name, args_dict
