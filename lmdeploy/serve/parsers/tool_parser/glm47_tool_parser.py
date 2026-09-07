# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Literal

from lmdeploy.serve.openai.protocol import ToolCall

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser


@ToolParserManager.register_module(['glm47'])
class Glm47ToolParser(XmlToolParser):
    """Parse GLM-4.7 mixed text and tags with the XML state machine.

    Leading plain text before the first ``<arg_key>`` or outer close resolves
    ``function``. After arguments begin, ``arg_start`` distinguishes the next
    key from ``</tool_call>``; the key close and value opener are consumed while
    resolving ``arg_name``.
    """

    structural_tag_model = 'glm_4_7'
    arg_key_start_token = '<arg_key>'
    arg_key_end_token = '</arg_key>'
    arg_value_start_token = '<arg_value>'
    arg_value_end_token = '</arg_value>'
    arg_value_close_tag = arg_value_end_token
    # The complete closing marker is one token in supported GLM tokenizers.
    arg_value_close_prefixes = ()

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        """Return the outer GLM tool-call opening tag."""
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        """Return the outer GLM tool-call closing tag."""
        return '</tool_call>'

    def _consume_function(
        self,
        payload: str,
        pos: int,
        final: bool,
    ) -> tuple[int, str, Literal['arg_start', 'done']] | None:
        """Resolve the leading plain-text function name once."""
        arg_key_start = payload.find(self.arg_key_start_token, pos)
        block_end = payload.find(self.get_tool_close_tag(), pos)
        if block_end >= 0 and (arg_key_start < 0 or block_end < arg_key_start):
            return block_end, payload[pos:block_end].strip(), 'done'
        if arg_key_start >= 0:
            return arg_key_start, payload[pos:arg_key_start].strip(), 'arg_start'

        if final and pos < len(payload):
            return len(payload), payload[pos:].strip(), 'done'
        return None

    def _consume_arg_start(self, payload: str, pos: int) -> tuple[int, Literal['arg_name', 'done']] | None:
        """Enter ``arg_name`` or stop before the outer closing tag."""
        arg_key_start = payload.find(self.arg_key_start_token, pos)
        block_end = payload.find(self.get_tool_close_tag(), pos)
        if block_end >= 0 and (arg_key_start < 0 or block_end < arg_key_start):
            return block_end, 'done'
        if arg_key_start < 0:
            return None
        return arg_key_start + len(self.arg_key_start_token), 'arg_name'

    def _consume_arg_name(self, payload: str, pos: int) -> tuple[int, str] | None:
        """Read the argument key and return the raw-value start."""
        key_end = payload.find(self.arg_key_end_token, pos)
        if key_end < 0:
            return None
        value_start = payload.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
        if value_start < 0:
            return None

        return value_start + len(self.arg_value_start_token), payload[pos:key_end].strip()

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        """Parse one complete GLM inner payload."""
        func_name, raw_arg_pairs = self._extract_complete_args(payload)
        if func_name is None:
            return None
        return self._build_tool_call(func_name, raw_arg_pairs)

    def parse_tool_block(self, text: str, start: int, tool_calls: list[ToolCall]) -> int:
        """Parse a GLM block without treating close-tag text in values as its
        end."""
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
        """Extract a function name and ordered raw argument pairs."""
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
