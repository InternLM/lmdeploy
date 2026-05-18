# Copyright (c) OpenMMLab. All rights reserved.
"""GPT-OSS Harmony response parser; only imported when openai_harmony is
available."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import shortuuid
from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from lmdeploy.utils import get_logger

from .response_parser import ResponseParser, ResponseParserManager, normalize_chat_request

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

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

    def __init__(self, request: ChatCompletionRequest, tokenizer: PreTrainedTokenizerBase):
        if hasattr(request, 'tools') and hasattr(request, 'tool_choice'):
            # GPT-OSS templates expect full tool wrappers.
            if request.tools is None or request.tool_choice == 'none':
                rendered_tools = None
            elif not isinstance(request.tool_choice, str):
                rendered_tools = [
                    item.model_dump() for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                rendered_tools = [item.model_dump() for item in request.tools]
            self.request = request.model_copy(update={'tools': rendered_tools})
        else:
            # Unit tests may inject a lightweight sentinel request object.
            self.request = request
        self._convert_response_format_to_harmony()
        self.request = normalize_chat_request(self.request)
        self.model_tokenizer = tokenizer
        self.parser = StreamableParser(get_encoding(), role=Role.ASSISTANT)
        self._seen_any = False
        self._next_tool_index = 0
        self._active_tool_id: str | None = None
        self._active_tool_index: int | None = None
        self._active_tool_name: str | None = None
        self.tool_parser = object()  # API server checks `is not None` for tool support.

    def _convert_response_format_to_harmony(self):
        """Convert response_format to Harmony-native mode for GPT-OSS.

        GPT-OSS uses Harmony mode for structured output, which conflicts with
        the engine's built-in JSON/response-format mode. This method injects
        the response_format schema into the system prompt as a
        ``# Response Formats`` section and clears ``response_format`` on the
        request so that only the Harmony-native instructions are used.
        """
        fmt = getattr(self.request, 'response_format', None)
        if fmt is None or getattr(fmt, 'type', 'text') == 'text':
            return

        try:
            format_json = json.dumps(fmt.model_dump())
            format_body = f'# Response Formats\n{format_json}'
            messages = self.request.messages

            if isinstance(messages, str):
                messages = messages + '\n\n' + format_body
                self._clear_response_format(messages=messages)
                return

            if not isinstance(messages, list):
                logger.warning('Cannot inject response_format schema into '
                               'non-list messages for GPT-OSS; clearing response_format only.')
                self._clear_response_format()
                return

            new_messages = list(messages)
            system_idx = next(
                (i for i, msg in enumerate(new_messages) if isinstance(msg, dict) and msg.get('role') == 'system'),
                None,
            )

            if system_idx is not None:
                content = new_messages[system_idx].get('content')
                if isinstance(content, list):
                    # Multimodal content blocks — append a text block.
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
        except Exception:
            logger.exception('Failed to convert response_format to Harmony-native mode for GPT-OSS')
            # Still clear response_format to avoid the Harmony/JSON mode conflict
            self._clear_response_format()

    def _clear_response_format(self, messages=None):
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

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs) -> tuple[DeltaMessage | None, bool]:
        if (
            not delta_text
            and not delta_token_ids
            and not self._seen_any
        ):
            return DeltaMessage(role='assistant', content=''), False

        self._seen_any = True

        # Harmony parsing is token-based. If a backend emits text without ids,
        # degrade gracefully as plain content.
        if not delta_token_ids:
            if not delta_text:
                return None, False
            return DeltaMessage(role='assistant', content=delta_text), False

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
                            id=self._active_tool_id,
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
            return None, False

        return DeltaMessage(
            role='assistant',
            content=content or None,
            reasoning_content=reasoning or None,
            tool_calls=tool_deltas or None,
        ), bool(tool_deltas)

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs) -> tuple:
        if not token_ids:
            # Keep non-streaming behavior consistent with other parsers:
            # when token ids are unavailable, return raw text as assistant content.
            return text or None, None, None

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
