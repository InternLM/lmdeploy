# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from .qwen3coder_tool_parser import Qwen3CoderToolParser
from .tool_parser import ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@ToolParserManager.register_module(['interns2-preview'])
class InternS2PreviewToolParser(Qwen3CoderToolParser):
    """Reuse the Qwen XML grammar with InternS2 request rendering policy."""

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Disable spaces between special tokens before common adjustments."""
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)
