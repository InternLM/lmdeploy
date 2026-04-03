# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)

from . import ToolParserManager
from .tool_parser import ToolParser


@ToolParserManager.register_module('llama3')
class Llama3JsonToolParser(ToolParser):
    """Tool parser for Llama3 JSON tool-call payloads."""

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.bot_token = '<|python_tag|>'

    def get_tool_open_tag(self) -> str | None:
        return self.bot_token

    def get_tool_close_tag(self) -> str | None:
        return None

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental JSON tool payload."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
