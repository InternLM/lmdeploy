# Copyright (c) OpenMMLab. All rights reserved.
"""Unified streaming parser for reasoning/content/tool calls."""
from __future__ import annotations

import json
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from mmengine import Registry

from lmdeploy.serve.openai.protocol import DeltaMessage
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaToolCall

    from .reasoning_parser import ReasoningParser
    from .tool_parser import ToolParser

logger = get_logger('lmdeploy')

ResponseParserManager = Registry('response_parser', locations=['lmdeploy.serve.parsers.response_parser'])


def validate_parser_names(
    reasoning_parser_name: str | None = None,
    tool_parser_name: str | None = None,
    *,
    warn_legacy: bool = True,
) -> tuple[str | None, str | None]:
    """Validate parser registry names before expensive engine startup."""
    from .reasoning_parser import LEGACY_REASONING_PARSER_NAMES, ReasoningParserManager
    from .tool_parser import ToolParserManager

    if reasoning_parser_name in LEGACY_REASONING_PARSER_NAMES:
        if warn_legacy:
            logger.warning(f'The reasoning parser {reasoning_parser_name} is deprecated, '
                           'please use the default reasoning parser instead.')
        reasoning_parser_name = 'default'

    if reasoning_parser_name is not None and reasoning_parser_name not in ReasoningParserManager.module_dict:
        raise ValueError(f'The reasoning parser {reasoning_parser_name} is not in the parser list: '
                         f'{ReasoningParserManager.module_dict.keys()}')

    if tool_parser_name is not None and tool_parser_name not in ToolParserManager.module_dict:
        raise ValueError(f'The tool parser {tool_parser_name} is not in the parser list: '
                         f'{ToolParserManager.module_dict.keys()}')

    return reasoning_parser_name, tool_parser_name


def _parse_tool_call_arguments_dict(arguments: Any) -> dict[str, Any] | None:
    """Return dict-like tool arguments for request message normalization.

    Raises ValueError if arguments is a string but contains invalid JSON. Returns None if arguments is not a string or
    JSON parses to non-dict.
    """
    if not isinstance(arguments, str):
        return None

    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as e:
        raise ValueError(
            f'Tool call arguments contain invalid JSON at position {e.pos} '
            f'(line {e.lineno}, column {e.colno})'
        ) from e
    except TypeError as e:
        raise ValueError('Tool call arguments contain invalid JSON') from e
    if isinstance(parsed_arguments, dict):
        return parsed_arguments
    return None


def _normalize_request_messages(messages: list[dict]) -> list[dict] | None:
    """Return a render-safe copy of request messages when needed."""
    normalized_messages = None

    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        tool_calls = message.get('tool_calls')
        if not isinstance(tool_calls, list):
            continue

        normalized_tool_calls = None
        for tool_idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get('function')
            if not isinstance(function, dict) or isinstance(function.get('arguments'), dict):
                continue

            parsed_arguments = _parse_tool_call_arguments_dict(function.get('arguments'))
            if parsed_arguments is None:
                continue

            if normalized_messages is None:
                normalized_messages = list(messages)
            if normalized_tool_calls is None:
                normalized_tool_calls = list(tool_calls)
                normalized_message = dict(message)
                normalized_message['tool_calls'] = normalized_tool_calls
                normalized_messages[msg_idx] = normalized_message

            normalized_function = dict(function)
            normalized_function['arguments'] = parsed_arguments

            normalized_tool_call = dict(tool_call)
            normalized_tool_call['function'] = normalized_function
            normalized_tool_calls[tool_idx] = normalized_tool_call

    return normalized_messages


def normalize_chat_request(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """Normalize a ChatCompletionRequest for downstream consumption.

    - ``response_format``: ``ResponseFormat → dict``, ``type='text' → None``
    - ``stop``: ``str → list[str]``
    - ``max_completion_tokens``: resolves from deprecated ``max_tokens``
    - ``messages``: tool call ``arguments`` JSON strings → dicts for template rendering
    """
    if not hasattr(request, 'model_copy'):
        return request

    updates: dict = {}

    fmt = request.response_format
    if fmt is not None and fmt.type != 'text':
        updates['response_format'] = fmt.model_dump()
    elif fmt is not None and fmt.type == 'text':
        updates['response_format'] = None

    if isinstance(request.stop, str):
        updates['stop'] = [request.stop]

    if request.max_completion_tokens is None:
        max_tokens = getattr(request, 'max_tokens', None)
        if max_tokens is not None:
            updates['max_completion_tokens'] = max_tokens

    messages = request.messages
    if isinstance(messages, list):
        normalized_messages = _normalize_request_messages(messages)
        if normalized_messages is not None:
            updates['messages'] = normalized_messages

    if updates:
        request = request.model_copy(update=updates)

    return request


class ResponseParser:
    supports_required_tool_choice: ClassVar[bool] = False
    reasoning_tokens: int | None

    @classmethod
    def set_parsers(
        cls,
        reasoning_parser_name: str | None = None,
        tool_parser_name: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        pass

    def __init__(self, request: ChatCompletionRequest):
        self.request = request
        self.reasoning_tokens = None

    @abstractmethod
    def stream_chunk(self,
                     delta_text: str,
                     delta_token_ids: list[int],
                     **kwargs) -> list[tuple[DeltaMessage, bool]]:
        """Parse one streamed chunk into delta message channels.

        Returns:
            A list of ``(delta_message, tool_calls_emitted)`` pairs. Return
            ``[]`` when this engine step produces no visible delta (for example
            while buffering protocol syntax or tool-call payload). For every
            accepted tool-call index, its first visible ``DeltaToolCall`` must
            carry the call ID, type, and function name before any argument
            fragments are exposed.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_complete(self,
                       text: str,
                       token_ids: list[int] | None = None,
                       **kwargs) -> tuple[str, list | None, str | None]:
        raise NotImplementedError


@ResponseParserManager.register_module('default')
class BaseResponseParser(ResponseParser):
    """The default response parser for streaming and complete assistant
    responses.

    It separates model output into:
    - plain assistant content
    - reasoning content
    - tool-call deltas

    Parsing is protocol-driven and supports mixed chunks where one
    ``delta_text`` may contain multiple segments (for example reasoning close
    plus plain text plus tool open tag).
    """

    reasoning_parser_cls: ClassVar[type[ReasoningParser] | None] = None
    tool_parser_cls: ClassVar[type[ToolParser] | None] = None
    tokenizer: ClassVar[PreTrainedTokenizerBase | None] = None
    supports_required_tool_choice: ClassVar[bool] = True
    MODE_PLAIN: ClassVar[str] = 'plain'
    MODE_REASONING: ClassVar[str] = 'reasoning'
    MODE_TOOL: ClassVar[str] = 'tool'

    @classmethod
    def set_parsers(
        cls,
        reasoning_parser_name: str | None = None,
        tool_parser_name: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Configure reasoning/tool parser classes by registry name."""
        from .reasoning_parser import ReasoningParserManager
        from .tool_parser import ToolParserManager

        reasoning_parser_name, tool_parser_name = validate_parser_names(reasoning_parser_name, tool_parser_name)
        cls.tokenizer = tokenizer

        if reasoning_parser_name is not None:
            cls.reasoning_parser_cls = ReasoningParserManager.get(reasoning_parser_name)
            if tokenizer is not None:
                cls.reasoning_parser_cls.validate_tokenizer(tokenizer)

        if tool_parser_name is not None:
            cls.tool_parser_cls = ToolParserManager.get(tool_parser_name)

    @classmethod
    def chat_template_kwargs_from_request(cls, request: ChatCompletionRequest) -> dict:
        """Normalize parser-related template kwargs from the request.

        ``enable_thinking`` is a deprecated top-level field. This helper maps
        it into ``chat_template_kwargs`` so downstream parser behavior can rely
        on one normalized source.
        """
        chat_template_kwargs = dict(request.chat_template_kwargs or {})
        if request.enable_thinking is not None:
            logger.warning('`enable_thinking` will be deprecated in the future, '
                           'please use `chat_template_kwargs` instead.')
            if chat_template_kwargs.get('enable_thinking') is None:
                chat_template_kwargs['enable_thinking'] = request.enable_thinking
            else:
                logger.warning(
                    '`enable_thinking` in `chat_template_kwargs` will override the value in request.')
        if request.reasoning_effort in ('high', 'max'):
            chat_template_kwargs.setdefault('reasoning_effort', request.reasoning_effort)
        return chat_template_kwargs

    def __init__(self, request: ChatCompletionRequest):
        rcls = type(self).reasoning_parser_cls
        tcls = type(self).tool_parser_cls
        self._kwargs = type(self).chat_template_kwargs_from_request(request)
        self.enable_thinking: bool | None = self._kwargs.get(
            'enable_thinking', self._kwargs.get('thinking', None))
        if self._kwargs.get('thinking') is True:
            self.enable_thinking = True
        self.reasoning_parser: ReasoningParser | None = rcls(**self._kwargs) if rcls else None
        self.tool_parser: ToolParser | None = tcls() if tcls else None
        self.reasoning_enabled = bool(
            self.reasoning_parser is not None
            and self.enable_thinking is not False
            and self.reasoning_parser.starts_in_reasoning_mode()
        )
        if self.tool_parser is not None:
            self.request = self.tool_parser.adjust_request(request)
        else:
            if request.tool_choice == 'required':
                raise ValueError('`tool_choice="required"` requires a configured tool-call parser.')
            from .tool_parser.tool_parser import dump_tools

            self.request = dump_tools(request)

        self.request = normalize_chat_request(self.request)
        if request.tool_choice == 'required':
            required_response_format = self.tool_parser.build_required_response_format(
                request.tools or [], reasoning=self.reasoning_enabled)
            # Internal structural tags override client response_format because
            # an engine can apply only one guided-decoding constraint.
            self.request = self.request.model_copy(update={'response_format': required_response_format})

        self._received_any_text = False
        self._stream_final_received = False

        self._reasoning_open_tag: str | None = None
        self._reasoning_close_tag: str | None = None
        if self.reasoning_parser is not None:
            self._reasoning_open_tag = self.reasoning_parser.get_reasoning_open_tag()
            self._reasoning_close_tag = self.reasoning_parser.get_reasoning_close_tag()
            if not self._reasoning_close_tag:
                name = self.reasoning_parser.__class__.__name__
                raise RuntimeError(f'Reasoning parser {name} must provide a reasoning end tag')

        self._tool_open_tag: str | None = None
        if self.tool_parser is not None and self.request.tool_choice != 'none':
            self._tool_open_tag = self.tool_parser.get_tool_open_tag()
            if not self._tool_open_tag:
                name = self.tool_parser.__class__.__name__
                raise RuntimeError(f'Tool parser {name} must provide a tool start tag')

        if self.reasoning_enabled:
            self._mode = self.MODE_REASONING
        else:
            self._mode = self.MODE_PLAIN
        self._pending = ''
        self._after_tool_block = False

        self._initialize_reasoning_token_counter()

    def stream_chunk(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        **kwargs,
    ) -> list[tuple[DeltaMessage, bool]]:
        """Parse one streamed chunk into delta message channels.

        Args:
            delta_text: New text fragment produced in this stream step.
            delta_token_ids: Token ids corresponding to ``delta_text``.

        Returns:
            A list of ``(delta_message, tool_calls_emitted)`` pairs produced
            from this stream step. Multiple entries may be returned when one
            engine chunk contains reasoning, content, and tool-call segments.
            Return ``[]`` when this engine step produces no visible delta (for
            example while buffering protocol syntax or tool-call payload).
            Tool calls follow the identity-first contract documented by
            :class:`ResponseParser`.
        """
        self._update_reasoning_tokens(delta_token_ids)
        self._stream_final_received = bool(kwargs.get('final', False))

        # Special-case: some backends emit a leading empty delta (no text, no
        # tokens) before any actual content. Tests treat this as a visible empty
        # content delta.
        if (
            not delta_text
            and not delta_token_ids
            and not self._received_any_text
        ):
            return [(DeltaMessage(role='assistant', content=''), False)]

        if self.tool_parser is None and self.reasoning_parser is None:
            if not delta_text:
                return []
            return [(DeltaMessage(role='assistant', content=delta_text), False)]

        if delta_text:
            self._received_any_text = True
        self._pending += delta_text
        deltas: list[tuple[DeltaMessage, bool]] = []

        while True:
            progressed = False
            if self._mode == self.MODE_PLAIN:
                emitted, progressed = self._consume_plain()
                if emitted:
                    deltas.append((DeltaMessage(role='assistant', content=emitted), False))
            elif self._mode == self.MODE_REASONING:
                emitted, progressed = self._consume_reasoning()
                if emitted:
                    if self.enable_thinking is False:
                        deltas.append((DeltaMessage(role='assistant', content=emitted), False))
                    else:
                        deltas.append((DeltaMessage(role='assistant', reasoning_content=emitted), False))
            if self._mode == self.MODE_TOOL:
                # self._consume_plain() might change the mode to MODE_TOOL
                # so we need to check the mode again
                new_calls, progressed = self._consume_tool()
                if new_calls:
                    deltas.append((DeltaMessage(role='assistant', tool_calls=new_calls), True))
            # A consumed chunk normally leaves tool mode waiting for the next
            # engine delta.  Re-entering the state machine with an empty
            # buffer only repeats parser dispatch on every streamed token.
            if not progressed or not self._pending:
                break
        return deltas

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
        if self._after_tool_block:
            newline_end = 0
            pending_size = len(self._pending)
            while newline_end < pending_size and self._pending[newline_end] == '\n':
                newline_end += 1

            if newline_end:
                tool_tag = self._tool_open_tag
                remaining = pending_size - newline_end
                if not remaining:
                    if not self._stream_final_received:
                        return None, False
                elif tool_tag is not None:
                    if self._pending.startswith(tool_tag, newline_end):
                        # Drop only the confirmed inter-block newlines. The
                        # normal tag path below starts the next tool block.
                        self._pending = self._pending[newline_end:]
                    elif (
                        remaining < len(tool_tag)
                        and not self._stream_final_received
                        and self._pending.startswith(tool_tag[:remaining], newline_end)
                    ):
                        # Keep both the newlines and a split tool-opening tag
                        # until the next chunk resolves their role.
                        return None, False
            self._after_tool_block = False

        tags = [t for t in (self._reasoning_open_tag, self._tool_open_tag) if t]
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

        # No protocol open tag found. Keep possible open-tag prefix suffix in
        # buffer so split tags across chunks can still be recognized later.
        # For example, intern-s1 tool parser's open tag <|action_start|><|plugin|>
        # is generated in two chunks, so we need to keep the prefix suffix in
        # buffer so split tags across chunks can still be recognized later.
        if earliest_idx < 0:
            if not self._pending:
                return None, False
            if self._stream_final_received:
                out = self._pending
                self._pending = ''
                return out, True
            keep = self._longest_open_tag_prefix_suffix(self._pending, tags)
            if keep > 0:
                if keep >= len(self._pending):
                    return None, False
                out = self._pending[:-keep]
                self._pending = self._pending[-keep:]
                return (out if out else None), bool(out)
            out = self._pending
            self._pending = ''
            return out, True

        # Emit content before protocol open tag.
        prefix = self._pending[:earliest_idx]
        self._pending = self._pending[earliest_idx + len(earliest_tag):]
        if earliest_tag == self._reasoning_open_tag:
            self._mode = self.MODE_REASONING
        else:
            self._mode = self.MODE_TOOL
            if self.tool_parser is not None:
                self.tool_parser.begin_tool_block()
        return (prefix if prefix else None), True

    def _consume_reasoning(self) -> tuple[str | None, bool]:
        """Consume buffered text while in reasoning mode.

        Behavior:
        - Drops the explicit open tag if model emits it.
        - If no close tag is present, emits only the safe reasoning-text prefix and
          preserves possible partial-tag suffix for the next chunk.
        - If a close tag or tool-open tag is found, emits text before it as
          reasoning content and switches to the next protocol mode.

        Returns:
            ``(emitted_text, progressed)`` where ``emitted_text`` is the reasoning
            content produced in this step (or ``None``), and ``progressed``
            indicates whether parser state/input was consumed.
        """

        open_tag = self._reasoning_open_tag
        # Drop explicit open tag if model emits it.
        if open_tag and self._pending.startswith(open_tag):
            self._pending = self._pending[len(open_tag):]
            return None, True

        close_tag = self._reasoning_close_tag
        if not close_tag:
            raise RuntimeError('Invariant violated: MODE_REASONING requires a reasoning_close_tag.')

        # GLM-style outputs may start a tool call directly from reasoning.
        tool_tag = self._tool_open_tag if self.tool_parser is not None else None
        boundary_tags = [tag for tag in (close_tag, tool_tag) if tag]

        idx = -1
        matched_tag = ''
        for tag in boundary_tags:
            tag_idx = self._pending.find(tag)
            if tag_idx >= 0 and (idx < 0 or tag_idx < idx):
                idx = tag_idx
                matched_tag = tag

        if idx < 0:
            if not self._pending:
                return None, False
            if self._stream_final_received:
                out = self._pending
                self._pending = ''
                return out, True
            keep = self._longest_open_tag_prefix_suffix(self._pending, boundary_tags)
            if keep > 0:
                if keep >= len(self._pending):
                    return None, False
                out = self._pending[:-keep]
                self._pending = self._pending[-keep:]
                return (out if out else None), bool(out)
            out = self._pending
            self._pending = ''
            return out, True

        reasoning_chunk = self._pending[:idx]
        self._pending = self._pending[idx + len(matched_tag):]
        if matched_tag == close_tag:
            self._mode = self.MODE_PLAIN
        else:
            self._mode = self.MODE_TOOL
            if self.tool_parser is not None:
                self.tool_parser.begin_tool_block()
        return (reasoning_chunk if reasoning_chunk else None), True

    def _consume_tool(self) -> tuple[list[DeltaToolCall], bool]:
        """Delegate a tool block and drop only its consumed input prefix."""
        if self.tool_parser is None:
            raise RuntimeError('Invariant violated: MODE_TOOL requires a tool_parser.')

        calls: list[DeltaToolCall] = []
        consumed = self.tool_parser.feed_tool_block(
            self._pending,
            calls,
            final=self._stream_final_received,
        )
        if consumed == len(self._pending):
            self._pending = ''
        elif consumed:
            self._pending = self._pending[consumed:]
        if self.tool_parser.block_closed:
            self._mode = self.MODE_PLAIN
            self._after_tool_block = True
        return calls, consumed > 0 or self.tool_parser.block_closed

    def _initialize_reasoning_token_counter(self) -> None:
        """Initialize token-based reasoning usage accounting."""
        self.reasoning_tokens = None
        self._counting_reasoning_tokens = self._mode == self.MODE_REASONING
        self._reasoning_start_token_id: int | None = None
        self._reasoning_end_token_id: int | None = None

        tokenizer = type(self).tokenizer
        if self.reasoning_parser is None or tokenizer is None:
            return

        self.reasoning_tokens = 0
        vocab = tokenizer.get_vocab()
        if self._reasoning_open_tag:
            self._reasoning_start_token_id = vocab[self._reasoning_open_tag]
        self._reasoning_end_token_id = vocab[self._reasoning_close_tag]

    def _update_reasoning_tokens(self, token_ids: list[int]) -> None:
        """Count tokens inside the logical reasoning-tag interval."""
        if self.reasoning_tokens is None or self.enable_thinking is False:
            return

        for token_id in token_ids:
            if token_id == self._reasoning_start_token_id:
                self._counting_reasoning_tokens = True
            elif token_id == self._reasoning_end_token_id:
                self._counting_reasoning_tokens = False
            elif self._counting_reasoning_tokens:
                self.reasoning_tokens += 1

    def parse_complete(
        self,
        text: str,
        token_ids: list[int] | None = None,
        **kwargs,
    ) -> tuple[str, list | None, str | None]:
        """Parse a complete response through the streaming state machine.

        Args:
            text: Full generated output text.

        Returns:
            A tuple ``(content, tool_calls, reasoning_content)``:
            - ``content``: plain assistant-visible text, or ``None``
            - ``tool_calls``: parsed tool calls, or ``None``
            - ``reasoning_content``: separated reasoning text, or ``None``
        """
        if self.reasoning_tokens is not None:
            self.reasoning_tokens = 0
            self._counting_reasoning_tokens = self.reasoning_enabled
        messages = self.stream_chunk(text, token_ids or [], final=True)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []
        for message, _ in messages:
            if message.content:
                content_parts.append(message.content)
            if message.reasoning_content:
                reasoning_parts.append(message.reasoning_content)
            if message.tool_calls:
                tool_deltas.extend(message.tool_calls)

        content = ''.join(content_parts)
        reasoning_content = ''.join(reasoning_parts) if reasoning_parts else None
        tool_calls = self.tool_parser.build_tool_calls(tool_deltas) if self.tool_parser is not None else []
        return content if content != '' else None, tool_calls or None, reasoning_content

    @staticmethod
    def _longest_open_tag_prefix_suffix(text: str, tags: list[str]) -> int:
        """Return length of longest suffix of ``text`` that is a prefix of any
        tag."""
        best = 0
        for tag in tags:
            max_k = min(len(text), len(tag) - 1)
            for k in range(max_k, 0, -1):
                if text.endswith(tag[:k]):
                    if k > best:
                        best = k
                    break
        return best
