# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)

from .tool_parser import ToolParser, ToolParserManager


@ToolParserManager.register_module('llama3')
class Llama3JsonToolParser(ToolParser):
    """Tool parser for Llama3 JSON tool-call payloads."""

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<|python_tag|>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return None

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental JSON tool payload."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
