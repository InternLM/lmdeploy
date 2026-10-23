# Copyright (c) OpenMMLab. All rights reserved.
"""Unified streaming parser for reasoning/content/tool calls."""
from __future__ import annotations

import json
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from mmengine import Registry

from lmdeploy.serve.openai.protocol import DeltaMessage
from lmdeploy.utils import get_logger

from .token_id_buffer import ControlSpan, LocatedText, ParseBuffer
from .token_id_scanner import TokenIDProfile, TokenIDScanner, build_token_profile

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.messages import GenerationConfig
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


def _resolve_token_id_terminals(
    tokenizer: PreTrainedTokenizerBase | None,
    reasoning_parser_cls: type[ReasoningParser] | None,
    tool_parser_cls: type[ToolParser] | None,
) -> tuple[str, ...]:
    """Merge declarations and preserve the existing GLM/default entry point."""
    from .reasoning_parser import ReasoningParser
    from .tool_parser.glm47_tool_parser import Glm47ToolParser

    if tokenizer is None:
        return ()
    reasoning_terminals = getattr(reasoning_parser_cls, 'token_id_terminals', ())
    # Preserve legacy mixed configurations unless reasoning explicitly opts in.
    if (tool_parser_cls is Glm47ToolParser and reasoning_parser_cls not in (None, ReasoningParser)
            and not reasoning_terminals):
        return ()
    terminals = reasoning_terminals + getattr(tool_parser_cls, 'token_id_terminals', ())
    if reasoning_parser_cls is ReasoningParser and tool_parser_cls in (None, Glm47ToolParser):
        reasoning_tags = (reasoning_parser_cls.get_reasoning_open_tag(),
                          reasoning_parser_cls.get_reasoning_close_tag())
        is_glm = tool_parser_cls is Glm47ToolParser
        if not is_glm:
            get_added_vocab = getattr(tokenizer, 'get_added_vocab', None)
            added = get_added_vocab() if get_added_vocab is not None else {}
            # Identify the protocol before validating required tags, so a
            # missing control token cannot silently disable the fix.
            glm_markers = {'[gMASK]', '<sop>', '<|observation|>'}
            glm_tags = reasoning_tags + Glm47ToolParser.token_id_terminals
            is_glm = glm_markers.issubset(added) or set(glm_tags).issubset(added)
        if is_glm:
            terminals = reasoning_tags + terminals
    return tuple(dict.fromkeys(terminals))


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
    requires_output_chunks: bool = False
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
            fragments are exposed. A tool block nested in reasoning emits only
            structured tool deltas, then resumes the reasoning channel.
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
    plus plain text plus tool open tag). A tool block temporarily suspends its
    containing channel: a block opened from reasoning resumes reasoning after
    it closes, while a block opened from plain content resumes plain content.
    Tool markup and payload are emitted only as structured tool-call deltas.
    """

    reasoning_parser_cls: ClassVar[type[ReasoningParser] | None] = None
    tool_parser_cls: ClassVar[type[ToolParser] | None] = None
    tokenizer: ClassVar[PreTrainedTokenizerBase | None] = None
    supports_required_tool_choice: ClassVar[bool] = True
    _token_profile: ClassVar[TokenIDProfile | None] = None
    _token_profile_key: ClassVar[tuple | None] = None
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
        from .reasoning_parser import ReasoningParser, ReasoningParserManager
        from .tool_parser import ToolParserManager

        reasoning_parser_name, tool_parser_name = validate_parser_names(reasoning_parser_name, tool_parser_name)
        cls.tokenizer = tokenizer
        cls.reasoning_parser_cls = None
        cls.tool_parser_cls = None
        cls._token_profile = None
        cls._token_profile_key = None

        if reasoning_parser_name is not None:
            cls.reasoning_parser_cls = ReasoningParserManager.get(reasoning_parser_name)

        if tool_parser_name is not None:
            cls.tool_parser_cls = ToolParserManager.get(tool_parser_name)

        terminals = _resolve_token_id_terminals(tokenizer, cls.reasoning_parser_cls, cls.tool_parser_cls)
        if terminals:
            assert tokenizer is not None
            cls._token_profile = build_token_profile(tokenizer, terminals)
        if cls.reasoning_parser_cls is not None and tokenizer is not None:
            reasoning_tags = (cls.reasoning_parser_cls.get_reasoning_open_tag(),
                              cls.reasoning_parser_cls.get_reasoning_close_tag())
            # Only the shared default validator is fully covered by the
            # profile. Preserve additional checks owned by custom parsers.
            if cls.reasoning_parser_cls is not ReasoningParser or not set(reasoning_tags).issubset(terminals):
                cls.reasoning_parser_cls.validate_tokenizer(tokenizer)
        cls._token_profile_key = (tokenizer, cls.reasoning_parser_cls, cls.tool_parser_cls)

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
        response_format = request.response_format
        format_type = (response_format.get('type') if isinstance(response_format, dict)
                       else getattr(response_format, 'type', None))
        # These grammars constrain the entire generated sequence, including
        # its first token. Required tools instead use a tagged grammar below.
        self._literal_output = (format_type in ('json_object', 'json_schema', 'regex_schema')
                                and request.tool_choice != 'required')
        if self._literal_output:
            template_kwargs = dict(request.chat_template_kwargs or {})
            template_kwargs['enable_thinking'] = False
            if 'thinking' in template_kwargs:
                template_kwargs['thinking'] = False
            request = request.model_copy(update={'chat_template_kwargs': template_kwargs})
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
        if self.reasoning_enabled and request.messages:
            last_message = request.messages[-1]
            prefill = last_message.get('content')
            if last_message.get('role') == 'assistant' and isinstance(prefill, str) and prefill:
                # A nonempty assistant message continues its existing channel;
                # the template does not add a fresh reasoning-generation prefix.
                open_tag = self.reasoning_parser.get_reasoning_open_tag()
                close_tag = self.reasoning_parser.get_reasoning_close_tag()
                self.reasoning_enabled = bool(open_tag and prefill.rfind(open_tag) > prefill.rfind(close_tag))
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
        self._tool_return_mode: str = self._mode
        # A profile is valid only for the tokenizer and parser classes checked
        # together at configuration time (including reasoning-only GLM).
        cls = type(self)
        profile = cls._token_profile if cls._token_profile_key == (cls.tokenizer, rcls, tcls) else None
        self._scanner = TokenIDScanner(cls.tokenizer, profile) if profile is not None else None
        self.requires_output_chunks = self._scanner is not None
        terminals = tuple(profile.controls.values()) if profile is not None else ()
        self._pending = ParseBuffer(token_id_terminals=terminals)
        self._plain_text_tags = [tag for tag in (self._reasoning_open_tag, self._tool_open_tag)
                                 if tag and tag not in self._pending.token_id_terminals]
        self._reasoning_text_tags = [tag for tag in (self._reasoning_close_tag, self._tool_open_tag)
                                     if tag and tag not in self._pending.token_id_terminals]
        self._tool_uses_token_ids = bool(
            profile is not None and self._pending.token_id_terminals.intersection(
                getattr(tcls, 'token_id_terminals', ())))
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
            delta_text: Native decoded text from this engine result.
            delta_token_ids: New output IDs from the same result. Preserve
                them even when ``delta_text`` is empty: incomplete UTF-8 bytes
                may become visible in a later batch.

        Keyword Args:
            final (bool): Flush the semantic parser because no more text will
                follow. This alone does not mark an independently decoded
                engine result. Defaults to False.
            engine_final (bool): Mark AsyncEngine's independent final result,
                as indicated by its finish reason. For token-ID parsing, this
                discards unrendered pending IDs and retains special tokens
                when aligning any included stop text. Do not set it merely
                to flush the parser. Defaults to False.

        Note:
            With token-ID parsing enabled, pass every native batch in order,
            including empty-text batches. Call ``bind_generation_config()``
            first with the effective configuration returned by preprocessing.

        Returns:
            A list of ``(delta_message, tool_calls_emitted)`` pairs produced
            from this stream step. Multiple entries may be returned when one
            engine chunk contains reasoning, content, and tool-call segments.
            Return ``[]`` when this engine step produces no visible delta (for
            example while buffering protocol syntax or tool-call payload).
            Tool calls follow the identity-first contract documented by
            :class:`ResponseParser`. Deltas preserve channel order, including
            reasoning before and after a nested tool block.
        """
        located = (self._scanner.scan(delta_text, delta_token_ids, engine_final=kwargs.get('engine_final', False))
                   if self._scanner is not None else LocatedText(delta_text))
        return self._consume_located(located, delta_token_ids, final=kwargs.get('final', False))

    def bind_generation_config(self, gen_config: GenerationConfig) -> None:
        """Bind detokenization options before processing any output batches.

        Args:
            gen_config: Effective configuration returned by engine
                preprocessing. Its ``skip_special_tokens`` value must match
                native decoding; the original request configuration may have
                been adjusted during preprocessing.
        """
        if self._scanner is not None:
            self._scanner.skip_special_tokens = gen_config.skip_special_tokens

    def _consume_located(
        self,
        located: LocatedText,
        delta_token_ids: list[int],
        *,
        final: bool,
    ) -> list[tuple[DeltaMessage, bool]]:
        delta_text = located.text
        self._update_reasoning_tokens(delta_token_ids)
        self._stream_final_received = final

        if self._literal_output:
            # Protocol-looking strings inside constrained output are data.
            # Do not strip them or parse them as reasoning/tool-call markup.
            self._received_any_text |= bool(delta_text)
            return [(DeltaMessage(role='assistant', content=delta_text), False)] if delta_text else []

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
        self._pending.append(located)
        deltas: list[tuple[DeltaMessage, bool]] = []

        while True:
            if self._after_tool_block and not self._resolve_tool_separator():
                break

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
                # The plain/reasoning consumer may enter tool mode, so dispatch
                # the buffered payload without waiting for another engine chunk.
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
        tags = [t for t in (self._reasoning_open_tag, self._tool_open_tag) if t]
        if not tags:
            if not self._pending:
                return None, False
            out = self._pending.take(len(self._pending))
            return out, True

        # Find the earliest protocol open tag.
        earliest_idx = -1
        earliest_tag = None
        for tag in tags:
            idx = self._pending.find_structure(tag)
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
            if self._stream_final_received or not self._plain_text_tags:
                out = self._pending.take(len(self._pending))
                return out, True
            keep = self._longest_open_tag_prefix_suffix(self._pending.text, self._plain_text_tags)
            if keep > 0:
                if keep >= len(self._pending):
                    return None, False
                out = self._pending.take(len(self._pending) - keep)
                return (out if out else None), bool(out)
            out = self._pending.take(len(self._pending))
            return out, True

        # Emit content before protocol open tag.
        prefix = self._pending.take(earliest_idx)
        self._pending.take(len(earliest_tag))
        if earliest_tag == self._reasoning_open_tag:
            self._mode = self.MODE_REASONING
        else:
            self._enter_tool_mode()
        return (prefix if prefix else None), True

    def _consume_reasoning(self) -> tuple[str | None, bool]:
        """Consume buffered text while in reasoning mode.

        Behavior:
        - Drops the explicit open tag if model emits it.
        - If no close tag is present, emits only the safe reasoning-text prefix and
          preserves possible partial-tag suffix for the next chunk.
        - If a close tag is found, emits text before it and switches to plain
          mode.
        - If a tool-open tag is found, emits text before it and temporarily
          suspends reasoning. Reasoning resumes after that tool block closes.

        Returns:
            ``(emitted_text, progressed)`` where ``emitted_text`` is the reasoning
            content produced in this step (or ``None``), and ``progressed``
            indicates whether parser state/input was consumed.
        """

        open_tag = self._reasoning_open_tag
        # Drop explicit open tag if model emits it.
        if open_tag and self._pending.startswith_structure(open_tag):
            self._pending.take(len(open_tag))
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
            tag_idx = self._pending.find_structure(tag)
            if tag_idx >= 0 and (idx < 0 or tag_idx < idx):
                idx = tag_idx
                matched_tag = tag

        if idx < 0:
            if not self._pending:
                return None, False
            if self._stream_final_received or not self._reasoning_text_tags:
                out = self._pending.take(len(self._pending))
                return out, True
            keep = self._longest_open_tag_prefix_suffix(self._pending.text, self._reasoning_text_tags)
            if keep > 0:
                if keep >= len(self._pending):
                    return None, False
                out = self._pending.take(len(self._pending) - keep)
                return (out if out else None), bool(out)
            out = self._pending.take(len(self._pending))
            return out, True

        reasoning_chunk = self._pending.take(idx)
        self._pending.take(len(matched_tag))
        if matched_tag == close_tag:
            self._mode = self.MODE_PLAIN
        else:
            self._enter_tool_mode()
        return (reasoning_chunk if reasoning_chunk else None), True

    def _enter_tool_mode(self) -> None:
        """Start a tool block and remember which response mode it suspended."""
        self._tool_return_mode = self._mode
        self._mode = self.MODE_TOOL
        self.tool_parser.begin_tool_block()

    def _resolve_tool_separator(self) -> bool:
        """Resolve possible newlines between consecutive tool blocks.

        Newlines are discarded only after a following tool opening tag is
        confirmed. Until then they remain buffered so they can be routed to
        the mode that the preceding tool block suspended.

        Returns:
            Whether the pending input is ready for normal mode dispatch.
        """
        newline_end = 0
        pending_size = len(self._pending)
        pending_text = self._pending.text
        while newline_end < pending_size and pending_text[newline_end] == '\n':
            newline_end += 1

        if newline_end:
            tool_tag = self._tool_open_tag
            remaining = pending_size - newline_end
            if not remaining:
                if not self._stream_final_received:
                    return False
            elif self._pending.startswith_structure(tool_tag, newline_end):
                self._pending.take(newline_end)
            elif (
                tool_tag not in self._pending.token_id_terminals
                and remaining < len(tool_tag)
                and not self._stream_final_received
                and pending_text.startswith(tool_tag[:remaining], newline_end)
            ):
                return False

        self._after_tool_block = False
        return True

    def _consume_tool(self) -> tuple[list[DeltaToolCall], bool]:
        """Delegate a tool block and drop only its consumed input prefix."""
        if self.tool_parser is None:
            raise RuntimeError('Invariant violated: MODE_TOOL requires a tool_parser.')

        calls: list[DeltaToolCall] = []
        # Custom/legacy parsers retain their existing call signature.
        kwargs = {'boundaries': self._pending.boundaries} if self._tool_uses_token_ids else {}
        consumed = self.tool_parser.feed_tool_block(
            self._pending.text,
            calls,
            final=self._stream_final_received,
            **kwargs,
        )
        self._pending.take(consumed)
        if self.tool_parser.block_closed:
            self._mode = self._tool_return_mode
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
        if self._scanner is not None:
            vocab = {tag: token_id for token_id, tag in self._scanner.profile.controls.items()}
            # A partial declaration may cover only tool tags. Reasoning usage
            # still needs its own IDs; avoid exporting the whole vocabulary.
            for tag in (self._reasoning_open_tag, self._reasoning_close_tag):
                if tag and tag not in vocab:
                    token_id = tokenizer.backend_tokenizer.token_to_id(tag)
                    if token_id is None:
                        raise ValueError(f'Missing reasoning token: {tag!r}')
                    vocab[tag] = token_id
        else:
            vocab = tokenizer.get_vocab()
        if self._reasoning_open_tag:
            self._reasoning_start_token_id = vocab[self._reasoning_open_tag]
        assert self._reasoning_close_tag is not None
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
            token_ids: All generated output IDs in order. Required for
                token-ID parsing and used for reasoning token counts.

        Keyword Args:
            output_chunks: Original native batches as
                ``(chunk_text, chunk_ids, engine_final)`` tuples. Preserve
                their order, empty-text batches and the independent final
                result; supply ``text`` and ``token_ids`` concatenated from
                these same batches. Token-ID parsing locates boundaries per
                batch before running the semantic parser over the full text.
                This preserves the native UTF-8 and final-result context
                that would be lost by merging or re-decoding all IDs.
                Ignored when token-ID parsing is disabled.

        Note:
            Call ``bind_generation_config()`` before token-ID parsing.
            Without ``output_chunks``, the inputs are treated as one ordinary
            decode batch. Callers collecting native engine output must supply
            the original batches, as ChatRunner does.

        Returns:
            A tuple ``(content, tool_calls, reasoning_content)``:
            - ``content``: plain assistant-visible text, or ``None``
            - ``tool_calls``: parsed tool calls, or ``None``
            - ``reasoning_content``: separated reasoning text, or ``None``
        """
        if self.reasoning_tokens is not None:
            self.reasoning_tokens = 0
            self._counting_reasoning_tokens = self.reasoning_enabled
        output_chunks = kwargs.get('output_chunks')
        if self._scanner is not None and output_chunks is not None:
            # Preserve native decode boundaries, then run the semantic parser
            # once. Re-decoding concatenated IDs would lose final/UTF-8 context.
            spans: list[ControlSpan] = []
            offset = 0
            for chunk_text, chunk_ids, engine_final in output_chunks:
                located = self._scanner.scan(chunk_text, chunk_ids, engine_final=engine_final)
                assert located.spans is not None
                if not text.startswith(chunk_text, offset):
                    raise ValueError('Output chunks do not match complete text')
                spans.extend(ControlSpan(s.start + offset, s.end + offset, s.tag) for s in located.spans)
                offset += len(chunk_text)
            if offset != len(text):
                raise ValueError('Output chunks do not cover complete text')
            messages = self._consume_located(LocatedText(text, tuple(spans)), token_ids or [], final=True)
        else:
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
