# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import html
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

    validate_tool_names = True

    @staticmethod
    def _parse_param_value(param_val_str: str) -> str:
        """Preserve MiMo parameter text while decoding HTML entities."""
        return html.unescape(param_val_str)

    @staticmethod
    def _coerce_value(raw_value: str, schema_type: str | None) -> Any:
        """Keep string whitespace and coerce non-string schema values."""
        if schema_type is None or schema_type == 'string':
            return raw_value
        return XmlToolParser._coerce_value(raw_value, schema_type)
