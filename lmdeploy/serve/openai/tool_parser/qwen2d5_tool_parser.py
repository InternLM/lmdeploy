# Copyright (c) OpenMMLab. All rights reserved.


from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')


@ToolParserManager.register_module(['qwen2d5'])
class Qwen2d5ToolParser(ToolParser):

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'
        self.pattern = r'<tool_call>(.*?)</tool_call>'
        self.parse_cursor = 0
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool: list[str] = []
        self.prev_tool_call_arr: list[dict] = []

    def get_argments(self, obj):
        if 'parameters' in obj:
            return obj.get('parameters')
        elif 'arguments' in obj:
            return obj.get('arguments')
        return None

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Qwen2.5 tool payload is JSON; reuse shared JSON incremental
        decoder."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
