# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from . import ToolParserManager
from .tool_parser import ToolParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import (
        ChatCompletionRequest,
        DeltaToolCall,
        ToolCall,
    )

@ToolParserManager.register_module(['internlm', 'intern-s1'])
class Internlm2ToolParser(ToolParser):
    """Tool parser for InternLM JSON tool-call payloads."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

    def get_tool_open_tag(self) -> str | None:
        return '<|action_start|><|plugin|>'

    def get_tool_close_tag(self) -> str | None:
        return '<|action_end|>'

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental JSON tool payload."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
