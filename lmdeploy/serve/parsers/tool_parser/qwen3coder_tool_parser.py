# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Literal

from lmdeploy.serve.openai.protocol import ToolCall

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(XmlToolParser):
    """Parse Qwen function and parameter tags with the XML state machine.

    ``<function=...>`` resolves ``function``; ``<parameter=...>`` and
    ``</function>`` are handled by ``arg_start``; the remainder of the
    parameter opener resolves ``arg_name``.
    """

    structural_tag_model = 'qwen_3_coder'
    reasoning_structural_tag_model = 'qwen_3_5'
    strip_value_newlines = True

    func_prefix = '<function='
    func_suffix = '</function>'
    param_prefix = '<parameter='
    param_suffix = '</parameter>'
    arg_value_close_tag = param_suffix
    _param_close_start = '</'
    _param_close_without_end = '</parameter'
    arg_value_close_prefixes = (_param_close_without_end, _param_close_start)

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    def _consume_function(
        self,
        payload: str,
        pos: int,
        final: bool,
    ) -> tuple[int, str, Literal['arg_start', 'done']] | None:
        """Consume a complete ``<function=name>`` opener from ``function``."""
        start = payload.find(self.func_prefix, pos)
        if start < 0:
            return None
        name_start = start + len(self.func_prefix)
        name_end = payload.find('>', name_start)
        if name_end < 0:
            return None

        return name_end + 1, payload[name_start:name_end].strip(), 'arg_start'

    def _consume_arg_start(self, payload: str, pos: int) -> tuple[int, Literal['arg_name', 'done']] | None:
        """Enter ``arg_name`` or finish at the inner function close tag."""
        param_start = payload.find(self.param_prefix, pos)
        func_end = payload.find(self.func_suffix, pos)
        if func_end >= 0 and (param_start < 0 or func_end < param_start):
            return func_end + len(self.func_suffix), 'done'
        if param_start < 0:
            return None

        return param_start + len(self.param_prefix), 'arg_name'

    def _consume_arg_name(self, payload: str, pos: int) -> tuple[int, str] | None:
        """Read the parameter name and return the raw-value start."""
        name_end = payload.find('>', pos)
        if name_end < 0:
            return None

        return name_end + 1, payload[pos:name_end].strip()

    def _stable_arg_value_end(self, payload: str, start: int) -> int:
        """Dispatch Qwen's two partial markers by their final character."""
        end = len(payload)
        if end <= start:
            return end

        last = payload[-1]
        if last == '/' and payload.endswith(self._param_close_start, start):
            # '/' means the last character of "</"
            return end - len(self._param_close_start)
        if last == 'r' and payload.endswith(self._param_close_without_end, start):
            # 'r' means the last character of "</parameter"
            return end - len(self._param_close_without_end)
        return end

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        parsed = self._parse_complete_payload(payload, require_eof=True)
        if parsed is None:
            return None
        func_name, raw_arg_pairs, _ = parsed
        return self._build_tool_call(func_name, raw_arg_pairs)

    def parse_tool_block(self, text: str, start: int, tool_calls: list[ToolCall]) -> int:
        parsed = self._parse_complete_payload(text[start:], require_eof=False)
        close_tag = self.get_tool_close_tag()
        if parsed is None:
            close_at = text.find(close_tag, start)
            return close_at + len(close_tag) if close_at >= 0 else len(text)

        func_name, raw_arg_pairs, payload_end = parsed
        tool_calls.append(self._build_tool_call(func_name, raw_arg_pairs))
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
