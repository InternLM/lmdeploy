# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import html
import re
from typing import Any

from .qwen3coder_tool_parser import Qwen3CoderToolParser
from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser


@ToolParserManager.register_module('mimo')
class MiMoToolParser(Qwen3CoderToolParser):
    """Parse MiMo XML tool calls.

    MiMo uses the same function/parameter tags as Qwen3-Coder, but its prompt contract preserves parameter whitespace.
    HTML entities are decoded before values are converted according to the request's JSON schema.
    """

    strip_value_newlines = False
    # Keep at most one bounded, potentially incomplete entity between chunks.
    _incomplete_entity = re.compile(
        r'&(?:#[xX][0-9A-Fa-f]{0,8}|#\d{0,10}|[A-Za-z][A-Za-z0-9]{0,31})?$')

    def __init__(self):
        super().__init__()
        self._html_entity_suffix = ''

    def _reset_stream_state(self) -> None:
        """Reset MiMo's cross-chunk HTML entity buffer."""
        super()._reset_stream_state()
        self._html_entity_suffix = ''

    def _stream_string_delta(self, text: str, json_fragments: list[str]) -> None:
        """Decode complete entities while retaining a split entity suffix."""
        text = self._html_entity_suffix + text
        self._html_entity_suffix = ''
        match = self._incomplete_entity.search(text)
        if match is not None:
            self._html_entity_suffix = match.group()
            text = text[:match.start()]
        if text:
            super()._stream_string_delta(html.unescape(text), json_fragments)

    def _finish_arg(self, json_fragments: list[str]) -> None:
        """Flush a trailing entity candidate before closing the value."""
        if self._html_entity_suffix:
            XmlToolParser._stream_string_delta(
                self,
                html.unescape(self._html_entity_suffix),
                json_fragments,
            )
        self._html_entity_suffix = ''
        super()._finish_arg(json_fragments)

    def _normalize_complete_value(self, raw_value: str) -> str:
        """Decode entities without stripping MiMo value whitespace."""
        return html.unescape(super()._normalize_complete_value(raw_value))

    @staticmethod
    def _coerce_value(raw_value: str, schema_type: str | None) -> Any:
        """Keep string whitespace and coerce non-string schema values."""
        if schema_type is None or schema_type == 'string':
            return raw_value
        return XmlToolParser._coerce_value(raw_value, schema_type)
