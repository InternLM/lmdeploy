# Copyright (c) OpenMMLab. All rights reserved.


from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)

from .tool_parser import ToolParser, ToolParserManager


@ToolParserManager.register_module(['qwen2d5'])
class Qwen2d5ToolParser(ToolParser):
    """Tool parser for Qwen2.5 JSON tool-call payloads."""

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental JSON tool payload."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
