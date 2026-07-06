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

    def _reset_incremental_state(self) -> None:
        self._value_parts: list[str] = []
        self._reset_value_stream_state()

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

    def _block_stream_arg_delta(self) -> None:
        self._stream_blocked = True

    def _normalize_stream_arg_delta(self, raw: str, schema_type: str | None) -> str:
        if self._stream_blocked:
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

    def _consume_function(self, payload: str, pos: int, final: bool) -> XmlParseResult:
        """Read the GLM function name before the first ``<arg_key>``.

        Returns no position when the function name is still split across chunks. A final no-argument payload can
        complete with only a function name.
        """
        arg_key_start = payload.find('<arg_key>', pos)
        if arg_key_start >= 0:
            name = payload[pos:arg_key_start].strip()
            return XmlParseResult(
                next_pos=arg_key_start,
                next_phase='arg_start',
                func_name=name or None,
            )

        remaining = payload[pos:]
        if final and remaining.strip():
            return XmlParseResult(next_pos=len(payload), func_name=remaining.strip())
        return XmlParseResult(next_pos=None)

    def _consume_arg_start(self, payload: str, pos: int) -> XmlParseResult:
        """Find the next GLM ``<arg_key>`` marker and enter arg-name
        parsing."""
        arg_key_start = payload.find('<arg_key>', pos)
        if arg_key_start < 0:
            return XmlParseResult(next_pos=None)

        return XmlParseResult(
            next_pos=arg_key_start + len('<arg_key>'),
            next_phase='arg_name',
        )

    def _consume_arg_name(self, payload: str, pos: int) -> XmlParseResult:
        """Read ``<arg_key>`` content and advance to the following value body.

        The method waits for both ``</arg_key>`` and ``<arg_value>`` so the base
        class only receives an argument name once the value phase can start.
        """
        key_end = payload.find('</arg_key>', pos)
        if key_end < 0:
            return XmlParseResult(next_pos=None)

        value_start = payload.find('<arg_value>', key_end + len('</arg_key>'))
        if value_start < 0:
            return XmlParseResult(next_pos=None)

        self._value_parts.clear()
        self._reset_value_stream_state()
        return XmlParseResult(
            next_pos=value_start + len('<arg_value>'),
            next_phase='arg_value',
            arg_name=payload[pos:key_end].strip(),
        )

    def _consume_arg_value(self, payload: str, pos: int) -> XmlParseResult:
        """Consume GLM argument value text.

        Closed values return the completed raw value for the base class to attach
        to the active argument name. Open values return only safe raw deltas and
        leave a possible partial ``</arg_value>`` suffix buffered for the next
        chunk.
        """
        value_end = payload.find('</arg_value>', pos)

        if value_end >= 0:
            raw = payload[pos:value_end]
            if raw:
                self._value_parts.append(raw)
            value = ''.join(self._value_parts)
            self._value_parts.clear()
            self._reset_value_stream_state()
            return XmlParseResult(
                next_pos=value_end + len('</arg_value>'),
                next_phase='function',
                completed_arg_value=value,
            )

        raw_end = self._trim_partial_close_tag_suffix(payload, pos, '</arg_value>')
        if raw_end == pos:
            return XmlParseResult(next_pos=None, should_stop=True)

        raw_delta = payload[pos:raw_end]
        self._value_parts.append(raw_delta)
        return XmlParseResult(
            next_pos=raw_end,
            raw_arg_delta=raw_delta,
            should_stop=True,
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
