# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

logger = get_logger('lmdeploy')


@ToolParserManager.register_module(['internlm', 'intern-s1'])
class Internlm2ToolParser(ToolParser):

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.parse_cursor = 0
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool: list[str] = []
        self.prev_tool_call_arr: list[dict] = []

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

    def get_argments(self, obj):
        if 'parameters' in obj:
            return obj.get('parameters')
        elif 'arguments' in obj:
            return obj.get('arguments')
        return None

    def get_tool_open_tag(self) -> str | None:
        return '<|action_start|><|plugin|>'

    def get_tool_close_tag(self) -> str | None:
        return '<|action_end|>'

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """InternLM2 tool payload is JSON; reuse shared JSON incremental
        decoder."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
