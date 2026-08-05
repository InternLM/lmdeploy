# Copyright (c) OpenMMLab. All rights reserved.

from .json_tool_parser import JsonToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module(['qwen2d5'])
class Qwen2d5ToolParser(JsonToolParser):
    """Tool parser for Qwen2.5 JSON tool-call payloads."""

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'
