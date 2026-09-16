# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Literal

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
    arg_key_open_tag = '<arg_key>'
    arg_key_close_tag = '</arg_key>'
    arg_value_open_tag = '<arg_value>'
    arg_value_close_tag = '</arg_value>'
    # The complete closing marker is one token in supported GLM tokenizers.
    arg_value_close_prefixes = ()

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
        """Resolve the leading plain-text function name once."""
        arg_key_start = payload.find(self.arg_key_open_tag, pos)
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
        arg_key_start = payload.find(self.arg_key_open_tag, pos)
        block_end = payload.find(self.get_tool_close_tag(), pos)
        if block_end >= 0 and (arg_key_start < 0 or block_end < arg_key_start):
            return block_end, 'done'
        if arg_key_start < 0:
            return None
        return arg_key_start + len(self.arg_key_open_tag), 'arg_name'

    def _consume_arg_name(self, payload: str, pos: int) -> tuple[int, str] | None:
        """Read the argument key and return the raw-value start."""
        key_end = payload.find(self.arg_key_close_tag, pos)
        if key_end < 0:
            return None
        value_start = payload.find(self.arg_value_open_tag, key_end + len(self.arg_key_close_tag))
        if value_start < 0:
            return None

        return value_start + len(self.arg_value_open_tag), payload[pos:key_end].strip()
