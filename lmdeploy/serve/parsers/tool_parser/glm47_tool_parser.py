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

    _complete_payload_pattern = re.compile(
        r'^\s*[^\s<]+(?:\s*<arg_key>[^<]+</arg_key>\s*<arg_value>.*?</arg_value>)*\s*$',
        re.DOTALL,
    )

    def _reset_incremental_state(self) -> None:
        self._func_name: str | None = None
        self._args: dict[str, str] = {}
        self._arg_name: str | None = None
        self._value_parts: list[str] = []
        self._phase = 'function'
        self._stream_started = False
        self._stream_pending_ws = ''
        self._stream_blocked = False

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'xml'

    def _reset_value_stream_state(self) -> None:
        self._stream_started = False
        self._stream_pending_ws = ''
        self._stream_blocked = False

    def _stream_arg_delta(self, raw: str) -> str:
        if self._stream_blocked:
            return ''

        schema_type = self._get_param_schema_type(self._func_name, self._arg_name or '')
        if schema_type not in (None, 'string'):
            self._stream_blocked = True
            return ''

        text = self._stream_pending_ws + raw
        self._stream_pending_ws = ''

        if schema_type == 'string':
            if not self._stream_started:
                text = text.lstrip()
                if not text:
                    return ''
                if text.startswith('"'):
                    self._stream_blocked = True
                    return ''

            stable = text.rstrip()
            self._stream_pending_ws = text[len(stable):]
            if not stable:
                return ''

            self._stream_started = True
            return stable

        if not self._stream_started:
            stripped = text.lstrip()
            if not stripped:
                self._stream_pending_ws = text
                return ''
            if stripped.startswith('"'):
                self._stream_blocked = True
                return ''

        self._stream_started = True
        return text

    def _consume_payload(self, payload: str, *, final: bool) -> tuple[XmlToolSnapshot, int]:
        pos = 0
        arg_delta_parts: list[str] = []
        n = len(payload)

        while pos < n:
            remaining = payload[pos:]

            if self._phase == 'function':
                arg_key = '<arg_key>'
                idx = remaining.find(arg_key)
                if idx >= 0:
                    name = remaining[:idx].strip()
                    if name:
                        self._func_name = name
                    pos += idx + len(arg_key)
                    self._phase = 'arg_key_name'
                    continue
                if final and remaining.strip():
                    self._func_name = remaining.strip()
                    pos = n
                    break
                if len(remaining) < len(arg_key) and arg_key.startswith(remaining):
                    break
                break

            if self._phase == 'arg_key_name':
                close = '</arg_key>'
                close_idx = remaining.find(close)
                if close_idx < 0:
                    break
                self._arg_name = remaining[:close_idx].strip()
                pos += close_idx + len(close)
                self._phase = 'arg_value_open'
                continue

            if self._phase == 'arg_value_open':
                token = '<arg_value>'
                if not remaining.startswith(token):
                    idx = remaining.find(token)
                    if idx < 0:
                        if len(remaining) < len(token) and token.startswith(remaining):
                            break
                        break
                    pos += idx
                    continue
                pos += len(token)
                self._value_parts.clear()
                self._reset_value_stream_state()
                self._phase = 'arg_value'
                continue

            if self._phase == 'arg_value':
                close = '</arg_value>'
                close_idx = remaining.find(close)
                if close_idx >= 0:
                    raw = remaining[:close_idx]
                    if raw:
                        self._value_parts.append(raw)
                    if self._arg_name:
                        self._args[self._arg_name] = ''.join(self._value_parts)
                    self._arg_name = None
                    self._value_parts.clear()
                    self._reset_value_stream_state()
                    pos += close_idx + len(close)
                    self._phase = 'function'
                    continue

                raw_delta = self._strip_partial_xml_close_suffix(remaining, close)
                if not raw_delta:
                    break
                pos += len(raw_delta)
                self._value_parts.append(raw_delta)
                stream_delta = self._stream_arg_delta(raw_delta)
                if stream_delta:
                    arg_delta_parts.append(stream_delta)
                break

            break

        return (
            XmlToolSnapshot(
                self._func_name,
                dict(self._args),
                self._arg_name,
                ''.join(arg_delta_parts),
                False,
            ),
            pos,
        )

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict = self._extract_complete_args(payload)
        if not func_name:
            return None
        args_dict = self._get_coerced_args(func_name, raw_args_dict, use_cache=False)
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
