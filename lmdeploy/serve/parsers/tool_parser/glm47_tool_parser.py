# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from lmdeploy.serve.openai.protocol import FunctionCall, ToolCall

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlArgState, XmlToolParser


@ToolParserManager.register_module(['glm47'])
class Glm47ToolParser(XmlToolParser):
    """Tool parser for GLM-4.7 XML-like tool-call payloads."""

    structural_tag_model = 'glm_4_7'
    arg_key_start_token = '<arg_key>'
    arg_key_end_token = '</arg_key>'
    arg_value_start_token = '<arg_value>'
    arg_value_end_token = '</arg_value>'

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    def _consume_function(self, payload: str, pos: int, final: bool) -> int | None:
        arg_key_start = payload.find(self.arg_key_start_token, pos)
        block_end = payload.find(self.get_tool_close_tag(), pos)
        if block_end >= 0 and (arg_key_start < 0 or block_end < arg_key_start):
            if self._state.func_name is None:
                self._state.func_name = payload[pos:block_end].strip()
            self._state.phase = 'done'
            self._payload_closed = True
            return block_end
        if arg_key_start >= 0:
            if self._state.func_name is None:
                self._state.func_name = payload[pos:arg_key_start].strip()
            self._state.phase = 'arg_start'
            return arg_key_start

        if final and pos < len(payload):
            if self._state.func_name is None:
                self._state.func_name = payload[pos:].strip()
            return len(payload)
        return None

    def _consume_arg_start(self, payload: str, pos: int) -> int | None:
        arg_key_start = payload.find(self.arg_key_start_token, pos)
        if arg_key_start < 0:
            return None
        self._state.phase = 'arg_name'
        return arg_key_start + len(self.arg_key_start_token)

    def _consume_arg_name(self, payload: str, pos: int) -> int | None:
        key_end = payload.find(self.arg_key_end_token, pos)
        if key_end < 0:
            return None
        value_start = payload.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
        if value_start < 0:
            return None

        self._state.arg_name = payload[pos:key_end].strip()
        self._arg_state = XmlArgState()
        self._state.phase = 'arg_value'
        return value_start + len(self.arg_value_start_token)

    def _consume_arg_value(self, payload: str, pos: int, json_fragments: list[str]) -> int | None:
        value_end = payload.find(self.arg_value_end_token, pos)
        if value_end >= 0:
            self._consume_arg_delta(payload[pos:value_end], json_fragments)
            self._finish_arg(json_fragments)
            self._state.arg_name = None
            self._state.phase = 'function'
            return value_end + len(self.arg_value_end_token)

        raw_end = self._trim_partial_close_tag_suffix(payload, pos, self.arg_value_end_token)
        if raw_end == pos:
            return None
        self._consume_arg_delta(payload[pos:raw_end], json_fragments)
        return raw_end

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_arg_pairs = self._extract_complete_args(payload)
        if func_name is None:
            return None
        arg_pairs = self._get_coerced_args(func_name, raw_arg_pairs)
        return ToolCall(function=FunctionCall(name=func_name, arguments=self._dump_argument_pairs(arg_pairs)))

    def parse_tool_block(self, text: str, start: int, tool_calls: list[ToolCall]) -> int:
        pos = start
        close_tag = self.get_tool_close_tag()
        while True:
            arg_start = text.find(self.arg_key_start_token, pos)
            close_at = text.find(close_tag, pos)
            if close_at < 0:
                return len(text)
            if arg_start < 0 or close_at < arg_start:
                break

            key_end = text.find(self.arg_key_end_token, arg_start + len(self.arg_key_start_token))
            if key_end < 0:
                return len(text)
            value_start = text.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
            if value_start < 0:
                return len(text)
            value_end = text.find(self.arg_value_end_token, value_start + len(self.arg_value_start_token))
            if value_end < 0:
                return len(text)
            pos = value_end + len(self.arg_value_end_token)

        parsed = self.parse_tool_call_complete(text[start:close_at])
        if parsed is not None:
            tool_calls.append(parsed)
        return close_at + len(close_tag)

    def _extract_complete_args(self, payload: str) -> tuple[str | None, list[tuple[str, str]]]:
        payload = payload.strip()
        if not payload:
            return None, []

        args_start_idx = payload.find(self.arg_key_start_token)
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            args_text = payload[args_start_idx:]
        else:
            func_name = payload
            args_text = ''

        arg_pairs: list[tuple[str, str]] = []
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
            arg_pairs.append((key, args_text[value_content_start:value_end]))
            search_idx = value_end + len(self.arg_value_end_token)
        return func_name, arg_pairs
