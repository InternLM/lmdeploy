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
        rcls = type(self).reasoning_parser_cls
        tcls = type(self).tool_parser_cls
        if rcls is None and tcls is None:
            self.reasoning_parser = None
            self.tool_parser = None
            self.request = request
        else:
            self._kwargs = type(self).chat_template_kwargs_from_request(request)
            self.enable_thinking: bool | None = self._kwargs.get('enable_thinking', None)

            self.reasoning_parser: ReasoningParser | None = (
                rcls(tokenizer, **self._kwargs) if rcls else None
            )
            self.tool_parser: ToolParser | None = (
                tcls(tokenizer) if tcls else None
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
    ) -> tuple[DeltaMessage | None, bool]:
        """Update state, run tool then reasoning parsers.

        Returns:
            (delta_message, tool_calls_emitted) — the latter is True if this chunk
            carries non-empty ``tool_calls`` (for finish_reason handling).
        """
        # Special-case: some backends emit a leading empty delta (no text, no
        # tokens) before any actual content. Tests treat this as a visible empty
        # content delta.
        if (
            not delta_text
            and not delta_token_ids
            and getattr(self, 'stream_buffer', None) is not None
            and self.stream_buffer.current_text == ''
        ):
            return DeltaMessage(role='assistant', content=''), False

        if self.tool_parser is None and self.reasoning_parser is None:
            return DeltaMessage(role='assistant', content=delta_text), False

        delta_message = DeltaMessage(role='assistant')
        req = self.request
        # 1. Update cumulative buffer first so tool parsers can inspect full text.
        self._stream_update(delta_text, delta_token_ids)

        # 2. Run tool call parser first.
        reasoning_text = delta_text
        tool_text = delta_text
        tool_calls_emitted = False
        if req.tool_choice != 'none' and self.tool_parser is not None:
            # 2.1. Ask tool_parser (if any) where tool-call protocol starts in this chunk.
            start_idx = self.tool_parser.detect_tool_start_tag(
                delta_text=delta_text,
                delta_token_ids=delta_token_ids,
                stream_buffer=self.stream_buffer,
                request=req,
            )
            if start_idx is not None:
                # Everything before start_idx is outside the tool-call block.
                reasoning_text = delta_text[:start_idx]
                tool_text = delta_text[start_idx:]

            # 2.2. Run tool parser on tool_text (which may be the whole chunk or just the suffix).
            tool_delta = self.tool_parser.extract_tool_calls_streaming(
                delta_text=tool_text,
                delta_token_ids=delta_token_ids,
                request=req,
                stream_buffer=self.stream_buffer,
                **kwargs,
            )
            if tool_delta is not None and tool_delta.tool_calls:
                delta_message.tool_calls = tool_delta.tool_calls
                tool_calls_emitted = True
                if tool_delta.content is not None:
                    delta_message.content = tool_delta.content

        # 4. Run reasoning parser on reasoning_text only (tool protocol is excluded).
        if self.reasoning_parser is not None and reasoning_text:
            if self.enable_thinking is not False:
                reasoning_delta = self.reasoning_parser.extract_reasoning_streaming(
                    delta_text=reasoning_text,
                    delta_token_ids=delta_token_ids,
                    request=req,
                    stream_buffer=self.stream_buffer,
                    **kwargs,
                )
                if reasoning_delta is not None:
                    delta_message.reasoning_content = reasoning_delta.reasoning_content
                    # Only set content from reasoning if tool_parser did not already.
                    if reasoning_delta.content is not None and delta_message.content is None:
                        delta_message.content = reasoning_delta.content
            else:
                delta_message.content = (delta_message.content or '') + reasoning_text

        # 5. Special case: a trailing empty delta (delta_text == '') after non-empty
        # output should be surfaced as an explicit empty content delta so that
        # streaming clients see the final "no-op" chunk (some backends do this).
        if (
            delta_text == ''
            and delta_message.content is None
            and delta_message.reasoning_content is None
            and not delta_message.tool_calls
            and self.stream_buffer.current_text != ''
        ):
            delta_message.content = ''

        self._stream_step()

        # 6. If there is no reasoning, no tool_calls, and no visible content
        # change, treat this chunk as a non-delta.
        if (
            delta_message.reasoning_content is None
            and not delta_message.tool_calls
            and (delta_message.content is None or delta_message.content == '')
        ):
            return None, tool_calls_emitted

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
