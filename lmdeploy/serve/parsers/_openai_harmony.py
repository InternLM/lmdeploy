# Copyright (c) OpenMMLab. All rights reserved.
"""GPT-OSS Harmony response parser; only imported when openai_harmony is
available."""
from __future__ import annotations

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

from .response_parser import ResponseParser, ResponseParserManager

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

_harmony_encoding = None


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


@ResponseParserManager.register_module('gpt-oss')
class GptOssResponseParser(ResponseParser):
    """Harmony stream parser for GPT-OSS (assistant role)."""

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
        self.model_tokenizer = tokenizer
        self.parser = StreamableParser(get_encoding(), role=Role.ASSISTANT)
        self._seen_any = False
        self._next_tool_index = 0
        self._active_tool_id: str | None = None
        self._active_tool_index: int | None = None
        self._active_tool_name: str | None = None
        self.tool_parser = object()  # API server checks `is not None` for tool support.

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

        for token in delta_token_ids:
            prev_recipient = self.parser.current_recipient
            self.parser.process(token)
            cur_channel = self.parser.current_channel
            cur_recipient = self.parser.current_recipient
            token_delta = self.parser.last_content_delta or ''

            tool_name = self._extract_tool_name(cur_recipient)
            prev_tool_name = self._extract_tool_name(prev_recipient)
            is_tool_channel = cur_channel in ('commentary', 'analysis')

            if is_tool_channel and tool_name:
                # Start of a new tool call.
                if tool_name != prev_tool_name:
                    self._active_tool_id = f'chatcmpl-tool-{shortuuid.random()}'
                    self._active_tool_index = self._next_tool_index
                    self._active_tool_name = tool_name
                    self._next_tool_index += 1
                    tool_deltas.append(
                        DeltaToolCall(
                            id=self._active_tool_id,
                            index=self._active_tool_index,
                            type='function',
                            function=DeltaFunctionCall(name=tool_name),
                        ))

                if token_delta and self._active_tool_id is not None and self._active_tool_index is not None:
                    tool_deltas.append(
                        DeltaToolCall(
                            id=self._active_tool_id,
                            index=self._active_tool_index,
                            type=None,
                            function=DeltaFunctionCall(arguments=token_delta),
                        ))
                continue

            # Normal textual channels.
            if cur_channel == 'final':
                content += token_delta
            elif cur_channel == 'analysis':
                reasoning += token_delta

        if not content and not reasoning and not tool_deltas:
            return None, False

        return DeltaMessage(
            role='assistant',
            content=content or None,
            reasoning_content=reasoning or None,
            tool_calls=tool_deltas or None,
        ), bool(tool_deltas)

    def parse_complete(self, text: str, **kwargs) -> tuple[str, list | None, str | None]:
        token_ids = kwargs.get('token_ids') or []
        if not token_ids:
            # Non-streaming path may not always pass token ids yet.
            return text if text else None, None, None

        content = ''
        reasoning = ''

        calls: list[dict] = []
        active: dict | None = None

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
                    if active is not None:
                        calls.append(active)
                    active = {
                        'id': f'chatcmpl-tool-{shortuuid.random()}',
                        'name': tool_name,
                        'arguments': '',
                    }
                if token_delta and active is not None:
                    active['arguments'] += token_delta
                continue

            if active is not None:
                calls.append(active)
                active = None

            if cur_channel == 'final':
                content += token_delta
            elif cur_channel == 'analysis':
                reasoning += token_delta

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
