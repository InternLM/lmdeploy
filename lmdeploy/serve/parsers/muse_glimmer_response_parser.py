# Copyright (c) OpenMMLab. All rights reserved.
"""ATEM response parser for Muse-Glimmer."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import shortuuid

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

from .response_parser import (
    BaseResponseParser,
    ResponseParser,
    ResponseParserManager,
    normalize_chat_request,
)

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


_CHANNEL_RE = re.compile(r'(?:<\|start\|>assistant)?\s*to=([^\s<]+)<\|message\|>')
_INVOKE_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)">(.*?)</atem:invoke>', re.DOTALL)
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:parameter>',
    re.DOTALL,
)
_CLOSE_TAGS = ('<|eom|>', '<|eot|>')


@dataclass
class _Parsed:
    content: str
    reasoning: str
    calls: list[tuple[str, str]]
    valid: bool


@ResponseParserManager.register_module('muse-glimmer')
class MuseGlimmerResponseParser(ResponseParser):
    """Parse Muse-Glimmer assistant channels and ATEM function calls."""

    tool_parser_cls = object()

    @classmethod
    def set_parsers(cls, reasoning_parser_name=None, tool_parser_name=None, tokenizer=None):
        # Muse owns both channels as one protocol; legacy parser flags do not alter it.
        return None

    def __init__(self, request: ChatCompletionRequest):
        if hasattr(request, 'model_copy'):
            chat_kwargs = dict(request.chat_template_kwargs or {})
            if 'reasoning_strength' not in chat_kwargs and request.reasoning_effort:
                chat_kwargs['reasoning_strength'] = (
                    'xhigh' if request.reasoning_effort == 'max' else request.reasoning_effort)
            request = request.model_copy(update={
                'chat_template_kwargs': chat_kwargs or None,
                'skip_special_tokens': False,
                'spaces_between_special_tokens': False,
            })
            request = BaseResponseParser.dump_tools(request)
        self.request = normalize_chat_request(request)
        self.tool_parser = object()
        self._text = ''
        self._sent_content = 0
        self._sent_reasoning = 0
        self._sent_calls = 0
        self._tool_ids: list[str] = []

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        if not delta_text and not delta_token_ids and not self._text:
            return [(DeltaMessage(role='assistant', content=''), False)]

        self._text += delta_text
        parsed = self._parse(self._text, final=False)
        content = parsed.content[self._sent_content:]
        reasoning = parsed.reasoning[self._sent_reasoning:]

        call_deltas = []
        for index, (name, arguments) in enumerate(parsed.calls[self._sent_calls:], self._sent_calls):
            while len(self._tool_ids) <= index:
                self._tool_ids.append(f'chatcmpl-tool-{shortuuid.random()}')
            call_deltas.append(DeltaToolCall(
                id=self._tool_ids[index],
                index=index,
                type='function',
                function=DeltaFunctionCall(name=name, arguments=arguments),
            ))

        self._sent_content = len(parsed.content)
        self._sent_reasoning = len(parsed.reasoning)
        self._sent_calls = len(parsed.calls)
        if not content and not reasoning and not call_deltas:
            return []
        return [(
            DeltaMessage(
                role='assistant',
                content=content or None,
                reasoning_content=reasoning or None,
                tool_calls=call_deltas or None,
            ),
            bool(call_deltas),
        )]

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
        parsed = self._parse(text, final=True)
        calls = [
            ToolCall(
                type='function',
                function=FunctionCall(name=name, arguments=arguments),
            )
            for name, arguments in parsed.calls
        ] or None
        return parsed.content or None, calls, parsed.reasoning or None

    def validate_complete(self, text: str | None = None) -> bool:
        return self._parse(self._text if text is None else text, final=True).valid

    def _parse(self, text: str, *, final: bool) -> _Parsed:
        content: list[str] = []
        reasoning: list[str] = []
        calls: list[tuple[str, str]] = []
        valid = True
        pos = 0
        found_channel = False

        while True:
            match = _CHANNEL_RE.search(text, pos)
            if match is None:
                trailing = text[pos:]
                if trailing and not self._is_protocol_only(trailing):
                    if not final and self._looks_like_channel_prefix(trailing):
                        trailing = ''
                    content.append(trailing)
                break

            found_channel = True
            prefix = text[pos:match.start()]
            if prefix and not self._is_protocol_only(prefix):
                content.append(prefix)

            recipient = match.group(1)
            body_start = match.end()
            close_pos, close_tag = self._find_close(text, body_start)
            closed = close_pos >= 0
            body = text[body_start:close_pos if closed else len(text)]
            if not closed and not final:
                body = self._safe_body(body)

            if recipient == 'self':
                reasoning.append(body)
                if not closed and final:
                    valid = False
            elif recipient == 'user':
                content.append(body)
            else:
                tool_calls, tools_valid = self._parse_atem(body)
                if getattr(self.request, 'tool_choice', 'auto') == 'none':
                    content.append(body)
                else:
                    calls.extend(tool_calls)
                    valid = valid and tools_valid
                if not closed and final and not tools_valid:
                    valid = False

            if not closed:
                break
            pos = close_pos + len(close_tag)

        if not found_channel and '<atem:invoke' in text:
            tool_calls, tools_valid = self._parse_atem(text)
            if tool_calls and getattr(self.request, 'tool_choice', 'auto') != 'none':
                calls = tool_calls
                content = []
            valid = valid and tools_valid
        return _Parsed(''.join(content), ''.join(reasoning), calls, valid)

    @staticmethod
    def _find_close(text: str, start: int) -> tuple[int, str]:
        candidates = [(text.find(tag, start), tag) for tag in _CLOSE_TAGS]
        candidates = [(idx, tag) for idx, tag in candidates if idx >= 0]
        return min(candidates, default=(-1, ''), key=lambda item: item[0])

    @staticmethod
    def _safe_body(body: str) -> str:
        tags = (*_CLOSE_TAGS, '<|start|>assistant')
        hold = 0
        for tag in tags:
            for size in range(1, min(len(tag), len(body)) + 1):
                if body.endswith(tag[:size]):
                    hold = max(hold, size)
        return body[:-hold] if hold else body

    @staticmethod
    def _looks_like_channel_prefix(text: str) -> bool:
        stripped = text.lstrip()
        prefixes = ('to=', '<|start|>assistant')
        return any(prefix.startswith(stripped) or stripped.startswith(prefix) for prefix in prefixes)

    @staticmethod
    def _is_protocol_only(text: str) -> bool:
        stripped = text.strip()
        return not stripped or stripped in {
            '<|start|>assistant',
            '<atem:function_calls>',
            '</atem:function_calls>',
        }

    @staticmethod
    def _parse_atem(body: str) -> tuple[list[tuple[str, str]], bool]:
        calls: list[tuple[str, str]] = []
        for match in _INVOKE_RE.finditer(body):
            name = html.unescape(match.group(1))
            arguments: dict[str, Any] = {}
            for parameter in _PARAM_RE.finditer(match.group(2)):
                key = html.unescape(parameter.group(1))
                value_text = html.unescape(parameter.group(2))
                try:
                    value = json.loads(value_text)
                except (json.JSONDecodeError, TypeError):
                    value = value_text
                arguments[key] = value
            calls.append((name, json.dumps(arguments, ensure_ascii=False)))

        opens = len(re.findall(r'<atem:invoke\b', body))
        closes = body.count('</atem:invoke>')
        params_open = len(re.findall(r'<atem:parameter\b', body))
        params_close = body.count('</atem:parameter>')
        return calls, opens == closes == len(calls) and params_open == params_close
