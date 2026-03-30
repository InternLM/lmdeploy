# Copyright (c) OpenMMLab. All rights reserved.
"""Unified streaming accumulation and façade for reasoning + tool call
parsing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from transformers import PreTrainedTokenizerBase

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import ReasoningParser
    from lmdeploy.serve.openai.tool_parser.tool_parser import ToolParser

logger = get_logger(__name__)


@dataclass
class StreamBuffer:
    """Cumulative decode snapshot (``ResponseParser.stream_buffer``); also
    passed as ``stream_buffer=``."""

    previous_text: str = ''
    current_text: str = ''
    previous_token_ids: list[int] = field(default_factory=list)
    current_token_ids: list[int] = field(default_factory=list)

    def update(self, delta_text: str, delta_token_ids: list[int]) -> None:
        self.current_text += delta_text
        self.current_token_ids.extend(delta_token_ids)

    def step(self) -> None:
        self.previous_text = self.current_text
        self.previous_token_ids = self.current_token_ids


class ResponseParser:
    """Single entry for streaming / complete post-processing (tool then
    reasoning).

    Parser *types* are configured at process start via :func:`lmdeploy.serve.openai.api_server.set_parsers`,
    which sets the class attributes below. Tests may assign those attributes on a subclass or temporarily on
    ``ResponseParser`` before construction.

    Streaming text/token accumulation lives on this instance (``current_text``, ``previous_token_ids``, etc.).
    """

    reasoning_parser_cls: ClassVar[type[ReasoningParser] | None] = None
    tool_parser_cls: ClassVar[type[ToolParser] | None] = None

    @classmethod
    def chat_template_kwargs_from_request(cls, request: ChatCompletionRequest) -> dict:
        """Merge ``request.enable_thinking`` into ``chat_template_kwargs``
        (deprecated field path)."""
        chat_template_kwargs = request.chat_template_kwargs or {}
        if request.enable_thinking is not None:
            logger.warning('`enable_thinking` will be deprecated in the future, '
                           'please use `chat_template_kwargs` instead.')
            if chat_template_kwargs.get('enable_thinking') is None:
                chat_template_kwargs['enable_thinking'] = request.enable_thinking
            else:
                logger.warning(
                    '`enable_thinking` in `chat_template_kwargs` will override the value in request.')
        return chat_template_kwargs

    def __init__(
        self,
        request: ChatCompletionRequest,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self._kwargs = type(self).chat_template_kwargs_from_request(request)
        self.enable_thinking: bool | None = self._kwargs.get('enable_thinking', None)
        rcls = type(self).reasoning_parser_cls
        tcls = type(self).tool_parser_cls
        self.reasoning_parser: ReasoningParser | None = (
            rcls(tokenizer, **self._kwargs) if rcls else None
        )
        self.tool_parser: ToolParser | None = (
            tcls(tokenizer, **self._kwargs) if tcls else None
        )
        if self.tool_parser is not None:
            self.request = self.tool_parser.adjust_request(request)
        else:
            self.request = request
        self.stream_buffer = StreamBuffer()

    def _stream_update(self, delta_text: str, delta_token_ids: list[int]) -> None:
        self.stream_buffer.update(delta_text, delta_token_ids)

    def _stream_step(self) -> None:
        self.stream_buffer.step()

    def stream_chunk(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        **kwargs,
    ) -> tuple[DeltaMessage, bool]:
        """Update state, run tool then reasoning parsers.

        Returns:
            (delta_message, tool_calls_emitted) — the latter is True if this chunk
            carries non-empty ``tool_calls`` (for finish_reason handling).
        """
        req = self.request
        self._stream_update(delta_text, delta_token_ids)

        delta_message = DeltaMessage(role='assistant', content=None)
        tool_calls_emitted = False

        if req.tool_choice != 'none' and self.tool_parser is not None:
            tool_delta = self.tool_parser.extract_tool_calls_streaming(
                delta_text=delta_text,
                delta_token_ids=delta_token_ids,
                request=req,
                stream_buffer=self.stream_buffer,
                **kwargs,
            )
            if tool_delta is not None:
                if tool_delta.tool_calls is not None:
                    delta_message.tool_calls = tool_delta.tool_calls
                if tool_delta.content is not None:
                    delta_message.content = tool_delta.content
                if isinstance(tool_delta.tool_calls, list) and len(tool_delta.tool_calls):
                    tool_calls_emitted = True
        elif req.tool_choice != 'none' and req.tools is not None and self.tool_parser is None:
            pass  # caller logs error

        if self.reasoning_parser is not None and self.enable_thinking is not False:
            reasoning_delta = self.reasoning_parser.extract_reasoning_streaming(
                delta_text=delta_message.content or '',
                delta_token_ids=delta_token_ids,
                request=req,
                stream_buffer=self.stream_buffer,
                **kwargs,
            )
            if reasoning_delta is not None:
                delta_message.reasoning_content = reasoning_delta.reasoning_content
                delta_message.content = reasoning_delta.content

        self._stream_step()
        return delta_message, tool_calls_emitted

    def parse_complete(
        self,
        text: str,
        **kwargs,
    ) -> tuple[str, list | None, str | None]:
        """Non-streaming: strip tools then reasoning. Returns (text, tool_calls, reasoning_content)."""
        req = self.request
        tool_calls = None
        reasoning_content = None
        out_text = text

        if req.tool_choice != 'none' and self.tool_parser is not None:
            tool_call_info = self.tool_parser.extract_tool_calls(out_text, request=req)
            out_text, tool_calls = tool_call_info.content, tool_call_info.tool_calls
        elif req.tool_choice != 'none' and req.tools is not None and self.tool_parser is None:
            pass

        if self.reasoning_parser is not None and self.enable_thinking is not False:
            reasoning_content, out_text = self.reasoning_parser.extract_reasoning(out_text, req)

        return out_text, tool_calls, reasoning_content
