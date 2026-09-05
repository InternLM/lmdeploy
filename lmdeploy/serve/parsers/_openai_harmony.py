# Copyright (c) OpenMMLab. All rights reserved.
"""GPT-OSS Harmony response parser; only imported when openai_harmony is
available."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import shortuuid
from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ResponseFormat,
    ToolCall,
)
from lmdeploy.utils import get_logger

from .response_parser import ResponseParser, ResponseParserManager, normalize_chat_request

if TYPE_CHECKING:

    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

logger = get_logger('lmdeploy')

_harmony_encoding = None


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


@ResponseParserManager.register_module('gpt-oss')
class GptOssResponseParser(ResponseParser):
    """Harmony stream parser for GPT-OSS (assistant role)."""
    tool_parser_cls = object()  # API server checks `is not None` for tool support.

    def __init__(self, request: ChatCompletionRequest):
        if hasattr(request, 'tools') and hasattr(request, 'tool_choice'):
            # GPT-OSS templates expect full tool wrappers.
            if request.tools is None or request.tool_choice == 'none':
                rendered_tools = None
            elif getattr(request.tool_choice, 'type', None) == 'function':
                # ToolChoice (type='function'): keep only the selected tool.
                rendered_tools = [
                    item.model_dump() for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                # auto/required/allowed_tools: keep all tools.
                rendered_tools = [item.model_dump() for item in request.tools]
            self.request = request.model_copy(update={'tools': rendered_tools})
        else:
            # Unit tests may inject a lightweight sentinel request object.
            self.request = request
        self._maybe_inject_tool_grammar()
        self.request = normalize_chat_request(self.request)
        self.parser = StreamableParser(get_encoding(), role=Role.ASSISTANT)
        self._seen_any = False
        self._next_tool_index = 0
        self._active_tool_id: str | None = None
        self._active_tool_index: int | None = None
        self._active_tool_name: str | None = None
        self.tool_parser = object()  # API server checks `is not None` for tool support.
        self.reasoning_tokens = 0

    def _maybe_inject_tool_grammar(self) -> None:
        """Convert any non-text ``response_format`` into a Harmony-compatible
        xgrammar structural_tag and keep it on the request so the engine can
        enforce it during generation.

        Two cases:

        - **Tool calling** (``tools`` present, ``tool_choice != 'none'``):
          delegates to :meth:`_build_tool_grammar` which uses
          ``xgrammar.get_model_structural_tag("harmony", ...)``.
        - **Plain structured output** (``json_schema``, ``regex_schema``,
          ``json_object``): builds a structural_tag that wraps the schema in
          the Harmony final channel (``<|channel|>final<|message|> ...
          <|end|>``), optionally preceded by an analysis block.

        If grammar construction fails the original ``response_format`` is
        injected into the system prompt as a ``# Response Formats`` section
        and then cleared, matching the previous Harmony-native fallback.
        """
        fmt = getattr(self.request, 'response_format', None)
        tools = getattr(self.request, 'tools', None)
        tool_choice = getattr(self.request, 'tool_choice', 'auto')
        has_tools = tools and tool_choice != 'none'

        if has_tools:
            grammar = self._build_tool_grammar(tools, tool_choice)
            if grammar is not None:
                self._set_response_format(grammar)
            else:
                # tool grammar failed — fall back to prompt injection so
                # the original response_format (if any) is not left intact
                # to conflict with Harmony tool-call constraints downstream.
                if fmt is not None and getattr(fmt, 'type', 'text') != 'text':
                    self._convert_response_format_to_harmony()
            return
        if fmt is not None and getattr(fmt, 'type', 'text') != 'text':
            grammar = self._build_response_format_grammar(fmt)
            if grammar is not None:
                self._set_response_format(grammar)
                return
            # grammar construction failed — fall back to prompt injection
            self._convert_response_format_to_harmony()

    @staticmethod
    def _build_tool_grammar(tools: list, tool_choice: Any) -> dict | None:
        """Construct a Harmony-compatible structural_tag for tool calling.

        Returns a dict suitable for ``response_format`` or ``None`` on
        failure.
        """
        try:
            from xgrammar.builtin_structural_tag import get_model_structural_tag
        except ImportError:
            logger.warning('xgrammar builtin_structural_tag not available; '
                           'falling back to prompt-only tool calling for GPT-OSS.')
            return None

        # Normalize tool_choice to xgrammar's expected format.
        xg_tool_choice = tool_choice
        if hasattr(tool_choice, 'model_dump'):
            xg_tool_choice = tool_choice.model_dump(mode='json')
        elif not isinstance(tool_choice, str):  # duck-typed without model_dump
            xg_tool_choice = {'type': 'function',
                              'function': {'name': tool_choice.function.name}}

        # Tools are usually model_dump'd dicts, but sentinel requests may
        # pass Tool objects.
        dumped_tools = [t if isinstance(t, dict) else t.model_dump() for t in tools]

        try:
            st = get_model_structural_tag(
                'harmony',
                tools=dumped_tools,
                tool_choice=xg_tool_choice,
                reasoning=True,
            )
            return {
                'type': 'structural_tag',
                'structural_tag': json.loads(st.model_dump_json()),
            }
        except Exception as e:  # xgrammar may raise ValueError/ValidationError
            logger.warning(f'Failed to build harmony structural tag for tool '
                           f'calling: {e}; falling back to prompt-only.')
            return None

    @staticmethod
    def _build_response_format_grammar(fmt: ResponseFormat) -> dict | None:
        """Convert a plain ``response_format`` (json_schema / regex_schema /
        json_object) into a Harmony-compatible structural_tag.

        The schema is wrapped in the Harmony final channel::

            [<|channel|>analysis<|message|> ... <|end|><|start|>assistant]?
            <|channel|>final<|message|> <schema> <|end|>

        Returns a dict or ``None`` on failure.
        """
        try:
            from xgrammar.structural_tag import (
                AnyTextFormat,
                ConstStringFormat,
                JSONSchemaFormat,
                OptionalFormat,
                RegexFormat,
                SequenceFormat,
                StructuralTag,
                TagFormat,
            )
        except ImportError:
            logger.warning('xgrammar structural_tag not available; '
                           'clearing response_format for GPT-OSS.')
            return None

        fmt_type = getattr(fmt, 'type', 'text')
        analysis_end = ['<|end|>', '<|return|>']
        final_begin = '<|channel|>final<|message|>'
        final_end = ['<|end|>', '<|return|>']

        if fmt_type == 'json_schema':
            schema = fmt.json_schema
            if schema is not None and schema.json_schema is not None:
                raw = schema.json_schema
            else:
                raw = {'type': 'object'}
            content = JSONSchemaFormat(json_schema=raw)
        elif fmt_type == 'regex_schema':
            content = RegexFormat(pattern=fmt.regex_schema or '.*')
        elif fmt_type == 'json_object':
            content = JSONSchemaFormat(json_schema={'type': 'object'})
        else:
            return None

        analysis_tag = OptionalFormat(
            content=SequenceFormat(elements=[
                TagFormat(begin='<|channel|>analysis<|message|>',
                          content=AnyTextFormat(), end=analysis_end),
                ConstStringFormat(value='<|start|>assistant'),
            ]))
        final_tag = TagFormat(begin=final_begin, content=content, end=final_end)
        st = StructuralTag(format=SequenceFormat(elements=[analysis_tag, final_tag]))
        return {
            'type': 'structural_tag',
            'structural_tag': json.loads(st.model_dump_json()),
        }

    def _set_response_format(self, grammar: dict) -> None:
        """Set response_format on the request, handling both Pydantic and plain
        objects."""
        if hasattr(self.request, 'model_copy'):
            self.request = self.request.model_copy(
                update={'response_format': ResponseFormat(**grammar)})
        else:
            self.request.response_format = ResponseFormat(**grammar)

    def _convert_response_format_to_harmony(self) -> None:
        """Fall back to Harmony-native prompt injection when grammar
        construction is unavailable.

        Injects the ``response_format`` schema into the system prompt as a
        ``# Response Formats`` section and clears ``response_format`` so only
        the Harmony-native instructions are used. This is the legacy path
        used when xgrammar structural_tag construction fails.
        """
        fmt = getattr(self.request, 'response_format', None)
        if fmt is None or getattr(fmt, 'type', 'text') == 'text':
            return

        try:
            format_json = json.dumps(fmt.model_dump())
            format_body = f'# Response Formats\n{format_json}'
            messages = self.request.messages

            if not isinstance(messages, list):
                logger.warning('Cannot inject response_format schema into '
                               'non-list messages for GPT-OSS; clearing response_format only.')
                self._clear_response_format()
                return

            new_messages = list(messages)
            system_idx = next(
                (i for i, msg in enumerate(new_messages)
                 if isinstance(msg, dict) and msg.get('role') == 'system'),
                None,
            )

            if system_idx is not None:
                content = new_messages[system_idx].get('content')
                if isinstance(content, list):
                    new_messages[system_idx] = {
                        **new_messages[system_idx],
                        'content': content + [{'type': 'text', 'text': format_body}],
                    }
                elif isinstance(content, str):
                    new_messages[system_idx] = {
                        **new_messages[system_idx],
                        'content': (content + '\n\n' + format_body) if content else format_body,
                    }
                else:
                    # content is None or unexpected type — insert a separate
                    # system message so the schema is still available.
                    new_messages.insert(0, {'role': 'system', 'content': format_body})
            else:
                new_messages.insert(0, {'role': 'system', 'content': format_body})

            self._clear_response_format(messages=new_messages)
        except Exception:  # fmt.model_dump() or message manipulation may fail
            logger.exception('Failed to convert response_format to Harmony-native mode for GPT-OSS')
            self._clear_response_format()

    def _clear_response_format(self, messages: list | str | None = None) -> None:
        """Clear response_format on the request, handling both Pydantic and
        plain objects."""
        if hasattr(self.request, 'model_copy'):
            update = {'response_format': None}
            if messages is not None:
                update['messages'] = messages
            self.request = self.request.model_copy(update=update)
        else:
            self.request.response_format = None
            if messages is not None:
                self.request.messages = messages

    def stream_chunk(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        **kwargs,
    ) -> list[tuple[DeltaMessage, bool]]:
        if (
            not delta_text
            and not delta_token_ids
            and not self._seen_any
        ):
            return [(DeltaMessage(role='assistant', content=''), False)]

        self._seen_any = True

        # Harmony parsing is token-based. If a backend emits text without ids,
        # degrade gracefully as plain content.
        if not delta_token_ids:
            if not delta_text:
                return []
            return [(DeltaMessage(role='assistant', content=delta_text), False)]

        content = ''
        reasoning = ''
        tool_deltas: list[DeltaToolCall] = []

        for event_kind, event_value in self._iter_harmony_events(delta_token_ids):
            if event_kind == 'tool_start':
                self._active_tool_id = f'chatcmpl-tool-{shortuuid.random()}'
                self._active_tool_index = self._next_tool_index
                self._active_tool_name = event_value
                self._next_tool_index += 1
                tool_deltas.append(
                    DeltaToolCall(
                        id=self._active_tool_id,
                        index=self._active_tool_index,
                        type='function',
                        function=DeltaFunctionCall(name=event_value),
                    ))
                continue
            if event_kind == 'tool_arguments':
                if self._active_tool_id is not None and self._active_tool_index is not None:
                    tool_deltas.append(
                        DeltaToolCall(
                            id=None,
                            index=self._active_tool_index,
                            type=None,
                            function=DeltaFunctionCall(arguments=event_value),
                        ))
                continue
            if event_kind == 'content':
                content += event_value
            elif event_kind == 'reasoning':
                reasoning += event_value

        if not content and not reasoning and not tool_deltas:
            return []

        return [(
            DeltaMessage(
                role='assistant',
                content=content or None,
                reasoning_content=reasoning or None,
                tool_calls=tool_deltas or None,
            ),
            bool(tool_deltas),
        )]

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs) -> tuple:
        if not token_ids:
            # Keep non-streaming behavior consistent with other parsers:
            # when token ids are unavailable, return raw text as assistant content.
            return text or None, None, None

        self.reasoning_tokens = 0
        content = ''
        reasoning = ''

        calls: list[dict] = []
        active: dict | None = None

        for event_kind, event_value in self._iter_harmony_events(token_ids or []):
            if event_kind == 'tool_start':
                if active is not None:
                    calls.append(active)
                active = {
                    'id': f'chatcmpl-tool-{shortuuid.random()}',
                    'name': event_value,
                    'arguments': '',
                }
                continue

            if event_kind == 'tool_arguments':
                if active is not None:
                    active['arguments'] += event_value
                continue

            if active is not None:
                calls.append(active)
                active = None
            if event_kind == 'content':
                content += event_value
            elif event_kind == 'reasoning':
                reasoning += event_value

        if active is not None:
            calls.append(active)

        tool_calls = [
            ToolCall(
                id=call['id'],
                type='function',
                function=FunctionCall(name=call['name'], arguments=call['arguments']),
            ) for call in calls
        ] or None

        return content or None, tool_calls, reasoning or None

    def _iter_harmony_events(self, token_ids: list[int]):
        """Yield parsed harmony events from token ids.

        Event kinds:
        - ``tool_start``: tool-call channel switched to a new function.
        - ``tool_arguments``: incremental tool-arguments fragment.
        - ``content``: assistant final-channel content fragment.
        - ``reasoning``: assistant analysis-channel reasoning fragment.
        """
        for token in token_ids:
            prev_recipient = self.parser.current_recipient
            self.parser.process(token)
            cur_channel = self.parser.current_channel
            cur_recipient = self.parser.current_recipient
            token_delta = self.parser.last_content_delta or ''

            tool_name = self._extract_tool_name(cur_recipient)
            prev_tool_name = self._extract_tool_name(prev_recipient)
            is_tool_channel = cur_channel in ('commentary', 'analysis')

            if is_tool_channel and tool_name:
                if tool_name != prev_tool_name:
                    yield 'tool_start', tool_name
                if token_delta:
                    yield 'tool_arguments', token_delta
                continue

            if cur_channel == 'final' and token_delta:
                yield 'content', token_delta
            elif cur_channel == 'analysis' and token_delta:
                self.reasoning_tokens += 1
                yield 'reasoning', token_delta

    @staticmethod
    def _extract_tool_name(recipient: str | None) -> str | None:
        """Extract function name from recipient string.

        Handles malformed sequences like
        ``functions.bash<|channel|>commentary`` by stripping harmony tags.
        """
        if not recipient:
            return None
        idx = recipient.find('functions.')
        if idx < 0:
            return None
        clean = recipient[idx:]
        clean = clean.split('<|channel|>', 1)[0]
        clean = re.split(r'[\s<|]', clean, maxsplit=1)[0]
        if not clean.startswith('functions.') or len(clean) <= len('functions.'):
            return None
        return clean.split('functions.', 1)[1]
