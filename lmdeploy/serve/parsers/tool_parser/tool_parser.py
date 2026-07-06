# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import shortuuid
from mmengine import Registry

from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)

from ..response_parser import BaseResponseParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.parsers.tool_parser'])


class ToolParser:
    """Base class for model-specific tool parsers."""

    def __init__(self):
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request payload before rendering, if needed."""
        return BaseResponseParser.dump_tools(request)

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        """Return tool opening tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_open_tag has not been implemented!')

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        """Return tool closing tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_close_tag has not been implemented!')

    @classmethod
    def get_tool_payload_format(cls) -> str:
        """Return payload format for tool call body."""
        raise NotImplementedError('ToolParser.get_tool_payload_format has not been implemented!')

    def start_tool_call(self) -> None:
        """Mark start of a tool-call block."""
        self._active_tool_index += 1
        self._active_tool_call_id = f'chatcmpl-tool-{shortuuid.random()}'
        self._name_emitted = False

    def finish_tool_call(self) -> None:
        """Mark end of a tool-call block."""
        self._active_tool_call_id = ''
        self._name_emitted = False

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental tool payload emitted between tool tags."""
        raise NotImplementedError('ToolParser.decode_tool_incremental has not been implemented!')

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        """Parse one complete tool payload into OpenAI tool call object."""
        raise NotImplementedError('ToolParser.parse_tool_call_complete has not been implemented!')

    def validate_complete(self, text: str) -> bool:
        """Return whether complete response text has valid tool calls."""
        open_tag = self.get_tool_open_tag()
        close_tag = self.get_tool_close_tag()

        pos = 0
        while True:
            open_idx = text.find(open_tag, pos)
            close_idx = text.find(close_tag, pos)
            if open_idx < 0:
                return close_idx < 0

            payload_start = open_idx + len(open_tag)
            if close_idx < payload_start:
                return False

            payload = text[payload_start:close_idx].strip()
            if not self._validate_tool_payload(payload):
                return False

            pos = close_idx + len(close_tag)
            if pos >= len(text):
                return True

    def _validate_tool_payload(self, payload: str) -> bool:
        """Return whether one complete tool payload is structurally valid."""
        raise NotImplementedError('ToolParser._validate_tool_payload has not been implemented!')
