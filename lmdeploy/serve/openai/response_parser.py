# Copyright (c) OpenMMLab. All rights reserved.
"""Unified profile-driven streaming parser for reasoning/content/tool calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from transformers import PreTrainedTokenizerBase

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    ToolCall,
)
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


@dataclass
class ProtocolProfile:
    reasoning_open_tag: str | None = None
    reasoning_close_tag: str | None = None
    tool_open_tag: str | None = None
    tool_close_tag: str | None = None
    tool_payload_format: str = 'json'
    starts_in_reasoning_mode: bool = True


@dataclass
class _QueuedDelta:
    delta: DeltaMessage
    tool_calls_emitted: bool = False


class ResponseParser:
    """Unified parser for streaming and complete assistant responses.

    It separates model output into:
    - plain assistant content
    - reasoning content
    - tool-call deltas

    Parsing is protocol/profile-driven and supports mixed chunks where one
    ``delta_text`` may contain multiple segments (for example reasoning close
    plus plain text plus tool open tag).
    """

    reasoning_parser_cls: ClassVar[type[ReasoningParser] | None] = None
    tool_parser_cls: ClassVar[type[ToolParser] | None] = None
    MODE_PLAIN: ClassVar[str] = 'plain'
    MODE_REASONING: ClassVar[str] = 'reasoning'
    MODE_TOOL: ClassVar[str] = 'tool'

    @classmethod
    def chat_template_kwargs_from_request(cls, request: ChatCompletionRequest) -> dict:
        """Normalize parser-related template kwargs from the request.

        ``enable_thinking`` is a deprecated top-level field. This helper maps
        it into ``chat_template_kwargs`` so downstream parser behavior can rely
        on one normalized source.
        """
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

        self.profile = self._build_profile()
        if (self.reasoning_parser is not None and self.enable_thinking is not False
                and self.profile.starts_in_reasoning_mode):
            self._mode = self.MODE_REASONING
        else:
            self._mode = self.MODE_PLAIN
        self._pending = ''
        self._queued_deltas: list[_QueuedDelta] = []

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
        """Parse one streamed chunk into delta message channels.

        Args:
            delta_text: New text fragment produced in this stream step.
            delta_token_ids: Token ids corresponding to ``delta_text``.

        Returns:
            ``(delta_message, tool_calls_emitted)`` where:
            - ``delta_message`` is ``None`` when this step has no visible delta.
            - ``tool_calls_emitted`` is ``True`` if at least one tool-call
              delta is emitted in this step.
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

        self._stream_update(delta_text, delta_token_ids)
        self._pending += delta_text
        produced_any = False

        while True:
            progressed = False
            if self._mode == self.MODE_PLAIN:
                emitted, progressed = self._consume_plain()
                if emitted:
                    self._queued_deltas.append(_QueuedDelta(DeltaMessage(role='assistant', content=emitted), False))
                    produced_any = True
            elif self._mode == self.MODE_REASONING:
                emitted, progressed = self._consume_reasoning()
                if emitted:
                    if self.enable_thinking is False:
                        self._queued_deltas.append(_QueuedDelta(DeltaMessage(role='assistant', content=emitted), False))
                    else:
                        self._queued_deltas.append(
                            _QueuedDelta(DeltaMessage(role='assistant', reasoning_content=emitted), False))
                    produced_any = True
            if self._mode == self.MODE_TOOL:
                # self._consume_plain() might change the mode to MODE_TOOL
                # so we need to check the mode again
                new_calls, progressed = self._consume_tool()
                if new_calls:
                    self._queued_deltas.append(
                        _QueuedDelta(DeltaMessage(role='assistant', tool_calls=new_calls), True))
                    produced_any = True
            if not progressed:
                break

        # 5. Special case: a trailing empty delta (delta_text == '') after non-empty
        # output should be surfaced as an explicit empty content delta so that
        # streaming clients see the final "no-op" chunk (some backends do this).
        if (
            delta_text == ''
            and not produced_any
            and self.stream_buffer.current_text != ''
        ):
            self._queued_deltas.append(_QueuedDelta(DeltaMessage(role='assistant', content=''), False))

        self._stream_step()
        if not self._queued_deltas:
            return None, False
        queued = self._queued_deltas.pop(0)
        return queued.delta, queued.tool_calls_emitted

    def _consume_plain(self) -> tuple[str | None, bool]:
        """Consume buffered text while in plain mode.

        Behavior:
        - Finds the earliest protocol opening tag (reasoning/tool) in
          ``self._pending``.
        - If no full tag is present, emits only the safe plain-text prefix and
          preserves possible partial-tag suffix for the next chunk.
        - If a tag is found, emits text before the tag as plain content,
          consumes the tag, and switches mode:
          - reasoning open tag -> ``MODE_REASONING``
          - tool open tag -> ``MODE_TOOL`` (also initializes tool-call state)

        Returns:
            ``(emitted_text, progressed)`` where ``emitted_text`` is the plain
            content produced in this step (or ``None``), and ``progressed``
            indicates whether parser state/input was consumed.
        """
        tags = [t for t in (self.profile.reasoning_open_tag, self.profile.tool_open_tag) if t]
        if not tags:
            if not self._pending:
                return None, False
            out = self._pending
            self._pending = ''
            return out, True

        # Find the earliest protocol open tag.
        earliest_idx = -1
        earliest_tag = None
        for tag in tags:
            idx = self._pending.find(tag)
            if idx >= 0 and (earliest_idx < 0 or idx < earliest_idx):
                earliest_idx = idx
                earliest_tag = tag

        # No protocol open tag found, treat the whole pending text as plain content.
        if earliest_idx < 0:
            if not self._pending:
                return None, False
            out = self._pending
            self._pending = ''
            return out, True

        # Emit content before protocol open tag.
        prefix = self._pending[:earliest_idx]
        self._pending = self._pending[earliest_idx + len(earliest_tag):]
        if earliest_tag == self.profile.reasoning_open_tag:
            self._mode = self.MODE_REASONING
        else:
            self._mode = self.MODE_TOOL
            if self.tool_parser is not None:
                self.tool_parser.start_tool_call()
        return (prefix if prefix else None), True

    def _consume_reasoning(self) -> tuple[str | None, bool]:
        """Consume buffered text while in reasoning mode.

        Behavior:
        - Drops the explicit open tag if model emits it.
        - If no close tag is present, emits only the safe reasoning-text prefix and
          preserves possible partial-tag suffix for the next chunk.
        - If a close tag is found, emits text before the close tag as reasoning content,
          consumes the close tag, and switches mode to ``MODE_PLAIN``.

        Returns:
            ``(emitted_text, progressed)`` where ``emitted_text`` is the reasoning
            content produced in this step (or ``None``), and ``progressed``
            indicates whether parser state/input was consumed.
        """

        open_tag = self.profile.reasoning_open_tag
        # Drop explicit open tag if model emits it.
        if open_tag and self._pending.startswith(open_tag):
            self._pending = self._pending[len(open_tag):]
            return None, True

        close_tag = self.profile.reasoning_close_tag
        if not close_tag:
            raise RuntimeError('Invariant violated: MODE_REASONING requires a reasoning_close_tag.')

        idx = self._pending.find(close_tag)
        # No close tag found, treat the whole pending text as reasoning content.
        if idx < 0:
            if not self._pending:
                return None, False
            out = self._pending
            self._pending = ''
            return out, True

        reasoning_chunk = self._pending[:idx]
        self._pending = self._pending[idx + len(close_tag):]
        # reasoning part is done, switch to plain mode
        self._mode = self.MODE_PLAIN
        return (reasoning_chunk if reasoning_chunk else None), True

    def _consume_tool(self) -> tuple[list[DeltaToolCall], bool]:
        """Consume buffered text while in tool mode.

        Behavior:
        - Treats ``self._pending`` as tool payload bytes until ``tool_close_tag``
          is found.
        - For non-final payload chunks, forwards text to
          ``tool_parser.decode_tool_incremental(..., final=False)``.
        - For the final payload chunk (before close tag), forwards text with
          ``final=True``, then calls ``tool_parser.finish_tool_call()`` and
          switches mode back to ``MODE_PLAIN``.
        - This method is format-agnostic: JSON/XML/other details are handled
          entirely by the concrete tool parser implementation.

        Returns:
            ``(tool_call_deltas, progressed)`` where ``tool_call_deltas`` is the
            list emitted by the tool parser for this step (possibly empty), and
            ``progressed`` indicates whether parser state/input was consumed.
        """
        if self.tool_parser is None:
            raise RuntimeError('Invariant violated: MODE_TOOL requires a tool_parser.')

        close_tag = self.profile.tool_close_tag
        if not close_tag:
            if not self._pending:
                return [], False
            emit = self._pending
            self._pending = ''
            return self.tool_parser.decode_tool_incremental(added_text=emit, final=False), True

        idx = self._pending.find(close_tag)

        if idx < 0:
            if not self._pending:
                return [], False
            emit = self._pending
            self._pending = ''
            return self.tool_parser.decode_tool_incremental(added_text=emit, final=False), True

        # Final chunk inside tool block.
        inner = self._pending[:idx]
        self._pending = self._pending[idx + len(close_tag):]
        calls = self.tool_parser.decode_tool_incremental(added_text=inner, final=True)
        self.tool_parser.finish_tool_call()
        self._mode = self.MODE_PLAIN
        return calls, True

    def _build_profile(self) -> ProtocolProfile:
        profile = ProtocolProfile(starts_in_reasoning_mode=False)
        rparser = self.reasoning_parser
        tparser = self.tool_parser

        if rparser is not None:
            profile.reasoning_open_tag = rparser.get_reasoning_open_tag()
            profile.reasoning_close_tag = rparser.get_reasoning_close_tag()
            profile.starts_in_reasoning_mode = bool(rparser.starts_in_reasoning_mode())
            if not profile.reasoning_close_tag:
                raise ValueError(f'Reasoning parser {rparser.__class__.__name__} must provide a reasoning close tag')

        if tparser is not None and self.request.tool_choice != 'none':
            profile.tool_open_tag = tparser.get_tool_open_tag()
            profile.tool_close_tag = tparser.get_tool_close_tag()
            profile.tool_payload_format = tparser.get_tool_payload_format()
            if not profile.tool_open_tag:
                raise ValueError(f'Tool parser {tparser.__class__.__name__} must provide a tool open tag')
            if not profile.tool_close_tag:
                raise ValueError(f'Tool parser {tparser.__class__.__name__} must provide a tool close tag')
        return profile

    def parse_complete(
        self,
        text: str,
        **kwargs,
    ) -> tuple[str, list | None, str | None]:
        """Parse the final non-streaming text output.

        Args:
            text: Full generated output text.

        Returns:
            A tuple ``(content, tool_calls, reasoning_content)``:
            - ``content``: plain assistant-visible text, or ``None``
            - ``tool_calls``: parsed tool calls, or ``None``
            - ``reasoning_content``: separated reasoning text, or ``None``
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        pos = 0
        mode = self.MODE_REASONING if (self.profile.starts_in_reasoning_mode and self.reasoning_parser is not None
                                       and self.enable_thinking is not False) else self.MODE_PLAIN
        n = len(text)

        while pos < n:
            if mode == self.MODE_REASONING:
                close_tag = self.profile.reasoning_close_tag
                close_idx = text.find(close_tag, pos) if close_tag else -1
                if close_idx < 0:
                    piece = text[pos:]
                    if self.enable_thinking is False:
                        content_parts.append(piece)
                    else:
                        reasoning_parts.append(piece)
                    break
                piece = text[pos:close_idx]
                if piece:
                    if self.enable_thinking is False:
                        content_parts.append(piece)
                    else:
                        reasoning_parts.append(piece)
                pos = close_idx + len(close_tag)
                mode = self.MODE_PLAIN
                continue

            open_idx, open_tag = self._find_first(
                text,
                [t for t in (self.profile.reasoning_open_tag, self.profile.tool_open_tag) if t],
                pos,
            )
            if open_idx < 0:
                content_parts.append(text[pos:])
                break

            if open_idx > pos:
                content_parts.append(text[pos:open_idx])

            if open_tag == self.profile.reasoning_open_tag:
                mode = self.MODE_REASONING
                pos = open_idx + len(open_tag)
                continue

            # tool block
            close_tag = self.profile.tool_close_tag
            close_idx = text.find(close_tag, open_idx + len(open_tag)) if close_tag else -1
            if close_idx < 0:
                # Unterminated tool block: keep as plain text.
                content_parts.append(text[open_idx:])
                break
            tool_payload = text[open_idx + len(open_tag):close_idx].strip()
            parsed_call = self.tool_parser.parse_tool_call_complete(tool_payload) if self.tool_parser else None
            if parsed_call is not None:
                tool_calls.append(parsed_call)
            pos = close_idx + len(close_tag)

        content = ''.join(content_parts)
        reasoning_content = ''.join(reasoning_parts) if reasoning_parts else None
        return content if content != '' else None, tool_calls or None, reasoning_content

    @staticmethod
    def _find_first(text: str, tags: list[str], start: int) -> tuple[int, str]:
        best_idx = -1
        best_tag = ''
        for tag in tags:
            idx = text.find(tag, start)
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx = idx
                best_tag = tag
        return best_idx, best_tag
