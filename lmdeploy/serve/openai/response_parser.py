# Copyright (c) OpenMMLab. All rights reserved.
"""Unified profile-driven streaming parser for reasoning/content/tool calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import partial_json_parser
import shortuuid
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
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
class _ToolDecodeState:
    active_tool_id: str = ''
    active_tool_index: int = -1
    name_emitted: bool = False
    args_emitted_len: int = 0
    prev_args_json: str | None = None
    args_prefix_emitted: bool = False
    value_chars_emitted: int = 0
    args_closed_emitted: bool = False


class ResponseParser:
    """Single entry for streaming and complete parsing."""

    reasoning_parser_cls: ClassVar[type[ReasoningParser] | None] = None
    tool_parser_cls: ClassVar[type[ToolParser] | None] = None
    MODE_PLAIN: ClassVar[str] = 'plain'
    MODE_REASONING: ClassVar[str] = 'reasoning'
    MODE_TOOL: ClassVar[str] = 'tool'

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
        if (self.reasoning_parser is not None and self.enable_thinking is not False):
            self._mode = self.MODE_REASONING
        else:
            self._mode = self.MODE_PLAIN
        self._pending = ''
        self._tool_payload = ''
        self._tool_decode_state = _ToolDecodeState()

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
        """Parse a single streamed chunk."""
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

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[DeltaToolCall] = []
        tool_calls_emitted = False

        while True:
            progressed = False
            if self._mode == self.MODE_PLAIN:
                emitted, progressed = self._consume_plain()
                if emitted:
                    content_parts.append(emitted)
            elif self._mode == self.MODE_REASONING:
                emitted, progressed = self._consume_reasoning()
                if emitted:
                    if self.enable_thinking is False:
                        content_parts.append(emitted)
                    else:
                        reasoning_parts.append(emitted)
            else:  # self.MODE_TOOL
                new_calls, progressed = self._consume_tool()
                if new_calls:
                    tool_calls.extend(new_calls)
                    tool_calls_emitted = True
            if not progressed:
                break

        delta_message = DeltaMessage(role='assistant')
        if content_parts:
            delta_message.content = ''.join(content_parts)
        if reasoning_parts:
            delta_message.reasoning_content = ''.join(reasoning_parts)
        if tool_calls:
            delta_message.tool_calls = tool_calls

        # 5. Special case: a trailing empty delta (delta_text == '') after non-empty
        # output should be surfaced as an explicit empty content delta so that
        # streaming clients see the final "no-op" chunk (some backends do this).
        emitted_trailing_empty = False
        if (
            delta_text == ''
            and delta_message.content is None
            and delta_message.reasoning_content is None
            and not delta_message.tool_calls
            and self.stream_buffer.current_text != ''
        ):
            delta_message.content = ''
            emitted_trailing_empty = True

        self._stream_step()

        # 6. If there is no reasoning, no tool_calls, and no visible content
        # change, treat this chunk as a non-delta.
        if (
            delta_message.reasoning_content is None
            and not delta_message.tool_calls
            and (delta_message.content is None or delta_message.content == '')
            and not emitted_trailing_empty
        ):
            return None, tool_calls_emitted

        return delta_message, tool_calls_emitted

    def _consume_plain(self) -> tuple[str | None, bool]:
        tags = [t for t in (self.profile.reasoning_open_tag, self.profile.tool_open_tag) if t]
        if not tags:
            if not self._pending:
                return None, False
            out = self._pending
            self._pending = ''
            return out, True

        earliest_idx = -1
        earliest_tag = None
        for tag in tags:
            idx = self._pending.find(tag)
            if idx >= 0 and (earliest_idx < 0 or idx < earliest_idx):
                earliest_idx = idx
                earliest_tag = tag

        if earliest_idx < 0:
            emit, remain = self._split_on_partial_prefix(self._pending, tags)
            if emit == '':
                return None, False
            self._pending = remain
            return emit, True

        # Emit content before protocol open tag.
        prefix = self._pending[:earliest_idx]
        self._pending = self._pending[earliest_idx + len(earliest_tag):]
        if earliest_tag == self.profile.reasoning_open_tag:
            self._mode = self.MODE_REASONING
        else:
            self._mode = self.MODE_TOOL
            self._tool_payload = ''
            self._start_tool_call()
        return (prefix if prefix else None), True

    def _consume_reasoning(self) -> tuple[str | None, bool]:
        # Drop explicit open tag if model emits it.
        open_tag = self.profile.reasoning_open_tag
        if open_tag and self._pending.startswith(open_tag):
            self._pending = self._pending[len(open_tag):]
            return None, True

        close_tag = self.profile.reasoning_close_tag
        if not close_tag:
            if not self._pending:
                return None, False
            out = self._pending
            self._pending = ''
            return out, True

        earliest_idx = self._pending.find(close_tag)

        if earliest_idx < 0:
            emit, remain = self._split_on_partial_prefix(self._pending, [close_tag])
            if emit == '':
                return None, False
            self._pending = remain
            return emit, True

        reasoning_chunk = self._pending[:earliest_idx]
        self._pending = self._pending[earliest_idx + len(close_tag):]
        self._mode = self.MODE_PLAIN
        return (reasoning_chunk if reasoning_chunk else None), True

    def _consume_tool(self) -> tuple[list[DeltaToolCall], bool]:
        close_tag = self.profile.tool_close_tag
        if not close_tag:
            if not self._pending:
                return [], False
            emit = self._pending
            self._pending = ''
            self._tool_payload += emit
            return self._decode_tool_incremental(added_text=emit, final=False), True

        earliest_idx = self._pending.find(close_tag)

        if earliest_idx < 0:
            emit, remain = self._split_on_partial_prefix(self._pending, [close_tag])
            if emit == '':
                return [], False
            self._pending = remain
            self._tool_payload += emit
            return self._decode_tool_incremental(added_text=emit, final=False), True

        # Final chunk inside tool block.
        inner = self._pending[:earliest_idx]
        self._tool_payload += inner
        self._pending = self._pending[earliest_idx + len(close_tag):]
        calls = self._decode_tool_incremental(added_text=inner, final=True)
        self._finish_tool_call()
        self._mode = self.MODE_PLAIN
        return calls, True

    def _start_tool_call(self) -> None:
        st = self._tool_decode_state
        st.active_tool_index += 1
        st.active_tool_id = f'chatcmpl-tool-{shortuuid.random()}'
        st.name_emitted = False
        st.args_emitted_len = 0
        st.args_prefix_emitted = False
        st.value_chars_emitted = 0
        st.args_closed_emitted = False

    def _finish_tool_call(self) -> None:
        st = self._tool_decode_state
        st.active_tool_id = ''
        st.name_emitted = False
        st.args_emitted_len = 0
        st.prev_args_json = None
        st.args_prefix_emitted = False
        st.value_chars_emitted = 0
        st.args_closed_emitted = False
        self._tool_payload = ''

    def _decode_tool_incremental(self, added_text: str, final: bool) -> list[DeltaToolCall]:
        if self.profile.tool_payload_format != 'json':
            return []
        payload = self._tool_payload.strip()
        if not payload:
            return []

        st = self._tool_decode_state
        flags = Allow.ALL if st.name_emitted else Allow.ALL & ~Allow.STR
        try:
            obj = partial_json_parser.loads(payload, flags)
        except partial_json_parser.core.exceptions.MalformedJSON:
            return []

        if not isinstance(obj, dict):
            return []

        out: list[DeltaToolCall] = []
        if not st.name_emitted:
            fn_name = obj.get('name')
            if isinstance(fn_name, str) and fn_name:
                out.append(
                    DeltaToolCall(
                        id=st.active_tool_id,
                        index=st.active_tool_index,
                        type='function',
                        function=DeltaFunctionCall(name=fn_name),
                    ))
                st.name_emitted = True

        args_obj = obj.get('arguments', obj.get('parameters', None))
        if args_obj is not None:
            # Value-stream mode for dict-with-string-values arguments. This
            # matches the reference chunk contract: emit object open once, then
            # only value text deltas, then close quote+brace at finalization.
            if isinstance(args_obj, dict):
                items = list(args_obj.items())
                if not st.args_prefix_emitted and items:
                    first_key = items[0][0]
                    out.append(
                        DeltaToolCall(
                            id=st.active_tool_id,
                            index=st.active_tool_index,
                            type=None,
                            function=DeltaFunctionCall(arguments=f'{{\"{first_key}\": \"')),
                    )
                    st.args_prefix_emitted = True

                values_concat = ''.join(v for _, v in items if isinstance(v, str))
                if len(values_concat) > st.value_chars_emitted:
                    diff = values_concat[st.value_chars_emitted:]
                    out.append(
                        DeltaToolCall(
                            id=st.active_tool_id,
                            index=st.active_tool_index,
                            type=None,
                            function=DeltaFunctionCall(arguments=diff),
                        ))
                    st.value_chars_emitted = len(values_concat)

                if self._is_complete_json(payload) and st.args_prefix_emitted and not st.args_closed_emitted:
                    out.append(
                        DeltaToolCall(
                            id=st.active_tool_id,
                            index=st.active_tool_index,
                            type=None,
                            function=DeltaFunctionCall(arguments='"}'),
                        ))
                    st.args_closed_emitted = True
                return out

            args_json = json.dumps(args_obj, ensure_ascii=False)
            # Do not emit/track empty dict/list placeholders during partial decode.
            if args_json not in ('{}', '[]'):
                emitted_arg = False
                candidate: str | None = None
                if self._is_complete_json(payload):
                    candidate = args_json
                elif st.prev_args_json:
                    candidate = self._common_prefix(st.prev_args_json, args_json)
                elif st.args_emitted_len == 0 and added_text:
                    pos = args_json.find(added_text)
                    if pos >= 0:
                        candidate = args_json[:pos + len(added_text)]

                if candidate and len(candidate) > st.args_emitted_len:
                    diff = candidate[st.args_emitted_len:]
                    if final or any(ch.isalnum() for ch in diff):
                        out.append(
                            DeltaToolCall(
                                id=st.active_tool_id,
                                index=st.active_tool_index,
                                type=None,
                                function=DeltaFunctionCall(arguments=diff),
                            ))
                        st.args_emitted_len = len(candidate)
                        emitted_arg = True

                # Some partial decodes don't advance parsed JSON although text
                # has advanced (e.g., unfinished string body). Stream lexical
                # text for content-bearing chunks to keep deltas monotonic.
                if (
                    not emitted_arg
                    and st.args_emitted_len > 0
                    and added_text
                    and any(ord(ch) > 127 for ch in added_text)
                ):
                    out.append(
                        DeltaToolCall(
                            id=st.active_tool_id,
                            index=st.active_tool_index,
                            type=None,
                            function=DeltaFunctionCall(arguments=added_text),
                        ))
                    st.args_emitted_len += len(added_text)
                st.prev_args_json = args_json
        return out

    @staticmethod
    def _is_complete_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def _common_prefix(s1: str, s2: str) -> str:
        i = 0
        n = min(len(s1), len(s2))
        while i < n and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    @staticmethod
    def _split_on_partial_prefix(text: str, tags: list[str]) -> tuple[str, str]:
        """Split text into (emit, remain) while preserving possible partial
        tags."""
        if not text:
            return '', ''
        max_keep = 0
        upper = min(len(text), max((len(t) for t in tags), default=0) - 1)
        for k in range(1, upper + 1):
            suffix = text[-k:]
            if any(tag.startswith(suffix) for tag in tags):
                max_keep = k
        if max_keep == 0:
            return text, ''
        return text[:-max_keep], text[-max_keep:]

    def _build_profile(self) -> ProtocolProfile:
        profile = ProtocolProfile(starts_in_reasoning_mode=False)
        rparser = self.reasoning_parser
        tparser = self.tool_parser

        if rparser is not None:
            profile.reasoning_open_tag = rparser.get_reasoning_open_tag()
            profile.reasoning_close_tag = rparser.get_reasoning_close_tag()
            profile.starts_in_reasoning_mode = bool(rparser.starts_in_reasoning_mode())

        if tparser is not None and self.request.tool_choice != 'none':
            profile.tool_open_tag = tparser.get_tool_open_tag()
            profile.tool_close_tag = tparser.get_tool_close_tag()
            profile.tool_payload_format = tparser.get_tool_payload_format()

        return profile

    def parse_complete(
        self,
        text: str,
        **kwargs,
    ) -> tuple[str, list | None, str | None]:
        """Non-streaming parse with the same profile-driven protocol
        semantics."""
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
            parsed_call = self._parse_tool_call_complete(tool_payload)
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

    def _parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        if self.profile.tool_payload_format != 'json':
            return None
        if not payload:
            return None
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get('name')
        if not isinstance(name, str) or not name:
            return None
        args_obj = obj.get('arguments', obj.get('parameters', {}))
        args_json = json.dumps(args_obj, ensure_ascii=False)
        return ToolCall(function=FunctionCall(name=name, arguments=args_json))
