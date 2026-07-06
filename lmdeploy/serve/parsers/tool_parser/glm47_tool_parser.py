# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser


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

    def _consume_function(self, payload: str, pos: int, final: bool) -> int | None:
        arg_key_start = payload.find('<arg_key>', pos)
        if arg_key_start >= 0:
            name = payload[pos:arg_key_start].strip()
            if name:
                self._func_name = name
            self._phase = 'arg_start'
            return arg_key_start

        remaining = payload[pos:]
        if final and remaining.strip():
            self._func_name = remaining.strip()
            return len(payload)
        return None

    def _consume_arg_start(self, payload: str, pos: int) -> int | None:
        arg_key_start = payload.find('<arg_key>', pos)
        if arg_key_start < 0:
            return None

        self._phase = 'arg_name'
        return arg_key_start + len('<arg_key>')

    def _consume_arg_name(self, payload: str, pos: int) -> int | None:
        key_end = payload.find('</arg_key>', pos)
        if key_end < 0:
            return None

        value_start = payload.find('<arg_value>', key_end + len('</arg_key>'))
        if value_start < 0:
            return None

        self._arg_name = payload[pos:key_end].strip()
        self._value_parts.clear()
        self._reset_value_stream_state()
        self._phase = 'arg_value'
        return value_start + len('<arg_value>')

    def _consume_arg_value(self, payload: str, pos: int, arg_delta_parts: list[str]) -> tuple[int | None, bool]:
        """Consume an argument value.

        Returns ``(next_pos, should_stop)``. ``should_stop`` is true after
        streaming an open value delta, because the next bytes may be the
        argument close tag and must be checked with the next chunk.
        """
        value_end = payload.find('</arg_value>', pos)

        if value_end >= 0:
            raw = payload[pos:value_end]
            if raw:
                self._value_parts.append(raw)
            if self._arg_name:
                self._args[self._arg_name] = ''.join(self._value_parts)
            self._arg_name = None
            self._value_parts.clear()
            self._reset_value_stream_state()
            self._phase = 'function'
            return value_end + len('</arg_value>'), False

        # Open value: keep any partial "</arg_value>" suffix buffered instead
        # of emitting it as argument text.
        raw_end = self._trim_partial_close_tag_suffix(payload, pos, '</arg_value>')
        if raw_end == pos:
            return None, True

        raw_delta = payload[pos:raw_end]
        self._value_parts.append(raw_delta)
        stream_delta = self._stream_arg_delta(raw_delta)
        if stream_delta:
            arg_delta_parts.append(stream_delta)
        return raw_end, True

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
