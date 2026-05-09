# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from .qwen3coder_tool_parser import Qwen3CoderToolParser
from .tool_parser import ToolParserManager

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@ToolParserManager.register_module(['interns2-preview'])
class InternS2PreviewToolParser(Qwen3CoderToolParser):
    """Tool parser for InternS2-Preview XML-style tool calls."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)
