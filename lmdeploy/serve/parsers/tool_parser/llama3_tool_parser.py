# Copyright (c) OpenMMLab. All rights reserved.

from .json_tool_parser import JsonToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module('llama3')
class Llama3JsonToolParser(JsonToolParser):
    """Tool parser for Llama3 JSON tool-call payloads."""

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<|python_tag|>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return None

    def validate_complete(self, text: str) -> bool:
        return True
