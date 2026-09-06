# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from lmdeploy.serve.openai.protocol import FunctionCall, ToolCall

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlArgState, XmlToolParser


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(XmlToolParser):
    """Tool parser for Qwen3Coder XML tool-call payloads."""

    structural_tag_model = 'qwen_3_coder'
    reasoning_structural_tag_model = 'qwen_3_5'
    strip_value_newlines = True

    func_prefix = '<function='
    func_suffix = '</function>'
    param_prefix = '<parameter='
    param_suffix = '</parameter>'

    # Qwen tokenizers emit ``</parameter>`` as the three complete token
    # pieces ``</``, ``parameter``, and ``>``.  These are therefore the only
    # incomplete suffixes that can occur at an incremental decode boundary.
    _param_close_start = '</'
    _param_close_without_end = '</parameter'

    def _close_json_on_final(self) -> bool:
        return False

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    def _consume_function(self, payload: str, pos: int, final: bool) -> int | None:
        del final
        start = payload.find(self.func_prefix, pos)
        if start < 0:
            return None
        name_start = start + len(self.func_prefix)
        name_end = payload.find('>', name_start)
        if name_end < 0:
            return None

        self._state.func_name = payload[name_start:name_end].strip()
        self._state.phase = 'arg_start'
        return name_end + 1

    def _consume_arg_start(self, payload: str, pos: int) -> int | None:
        param_start = payload.find(self.param_prefix, pos)
        func_end = payload.find(self.func_suffix, pos)
        if func_end >= 0 and (param_start < 0 or func_end < param_start):
            self._state.phase = 'done'
            self._payload_closed = True
            return func_end + len(self.func_suffix)
        if param_start < 0:
            return None

        self._state.phase = 'arg_name'
        return param_start + len(self.param_prefix)

    def _consume_arg_name(self, payload: str, pos: int) -> int | None:
        name_end = payload.find('>', pos)
        if name_end < 0:
            return None

        self._state.arg_name = payload[pos:name_end].strip()
        self._arg_state = XmlArgState()
        self._state.phase = 'arg_value'
        return name_end + 1

    def _consume_arg_value(self, payload: str, pos: int, json_fragments: list[str]) -> int | None:
        value_end = payload.find(self.param_suffix, pos)
        if value_end >= 0:
            self._consume_arg_delta(payload[pos:value_end], json_fragments)
            self._finish_arg(json_fragments)
            self._state.arg_name = None
            self._state.phase = 'arg_start'
            return value_end + len(self.param_suffix)

        raw_end = self._trim_partial_param_close_suffix(payload, pos)
        if raw_end == pos:
            return None
        self._consume_arg_delta(payload[pos:raw_end], json_fragments)
        return raw_end

    @classmethod
    def _trim_partial_param_close_suffix(cls, payload: str, start: int) -> int:
        """Return the value end at a Qwen token-aligned decode boundary."""
        end = len(payload)
        if end <= start:
            return end

        last = payload[-1]
        if last == '/' and payload.endswith(cls._param_close_start, start):
            return end - len(cls._param_close_start)
        if last == 'r' and payload.endswith(cls._param_close_without_end, start):
            return end - len(cls._param_close_without_end)
        return end

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        parsed = self._parse_complete_payload(payload, require_eof=True)
        if parsed is None:
            return None
        func_name, raw_arg_pairs, _ = parsed
        arg_pairs = self._get_coerced_args(func_name, raw_arg_pairs)
        return ToolCall(function=FunctionCall(name=func_name, arguments=self._dump_argument_pairs(arg_pairs)))

    def parse_tool_block(self, text: str, start: int, tool_calls: list[ToolCall]) -> int:
        parsed = self._parse_complete_payload(text[start:], require_eof=False)
        close_tag = self.get_tool_close_tag()
        if parsed is None:
            close_at = text.find(close_tag, start)
            return close_at + len(close_tag) if close_at >= 0 else len(text)

        func_name, raw_arg_pairs, payload_end = parsed
        arg_pairs = self._get_coerced_args(func_name, raw_arg_pairs)
        tool_calls.append(
            ToolCall(function=FunctionCall(name=func_name, arguments=self._dump_argument_pairs(arg_pairs))))
        close_at = text.find(close_tag, start + payload_end)
        return close_at + len(close_tag) if close_at >= 0 else len(text)

    def _parse_complete_payload(
        self,
        content: str,
        *,
        require_eof: bool,
    ) -> tuple[str, list[tuple[str, str]], int] | None:
        """Extract one inner function and its consumed prefix."""
        pos = content.find(self.func_prefix)
        if pos < 0:
            return None
        name_start = pos + len(self.func_prefix)
        name_end = content.find('>', name_start)
        if name_end < 0:
            return None
        func_name = content[name_start:name_end].strip()
        pos = name_end + 1
        arg_pairs: list[tuple[str, str]] = []

        while True:
            func_end = content.find(self.func_suffix, pos)
            param_start = content.find(self.param_prefix, pos)
            if func_end >= 0 and (param_start < 0 or func_end < param_start):
                pos = func_end + len(self.func_suffix)
                break
            if param_start < 0:
                return None

            name_start = param_start + len(self.param_prefix)
            name_end = content.find('>', name_start)
            if name_end < 0:
                return None
            param_name = content[name_start:name_end].strip()
            value_start = name_end + 1
            value_end = content.find(self.param_suffix, value_start)
            if value_end < 0:
                return None
            raw_value = self._normalize_complete_value(content[value_start:value_end])
            arg_pairs.append((param_name, raw_value))
            pos = value_end + len(self.param_suffix)

        pos = self._skip_ws(content, pos)
        if require_eof and pos != len(content):
            return None
        return func_name, arg_pairs, pos

    @staticmethod
    def _skip_ws(text: str, pos: int) -> int:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos
