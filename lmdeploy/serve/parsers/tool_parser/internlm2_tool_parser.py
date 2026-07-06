# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from .json_tool_parser import JsonToolParser
from .tool_parser import ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@ToolParserManager.register_module(['internlm', 'intern-s1'])
class Internlm2ToolParser(JsonToolParser):
    """Tool parser for InternLM JSON tool-call payloads."""

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<|action_start|><|plugin|>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '<|action_end|>'
