# Copyright (c) OpenMMLab. All rights reserved.

from .json_tool_parser import JsonToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module(['qwen', 'qwen2d5', 'qwen3'])
class Qwen3ToolParser(JsonToolParser):
    """Tool parser for Qwen2.5 and Qwen3 JSON tool-call payloads."""

    structural_tag_model = 'qwen_3'

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'
