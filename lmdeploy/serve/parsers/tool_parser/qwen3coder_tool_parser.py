# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser, XmlToolSnapshot


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(XmlToolParser):
    """Tool parser for Qwen3Coder XML tool-call payloads."""

    _complete_payload_pattern = re.compile(
        r'^\s*<function=[^\s>\n]+>\s*(?:<parameter=[^\s>\n]+>.*?</parameter>\s*)*</function>\s*$',
        re.DOTALL,
    )

    def _reset_incremental_state(self) -> None:
        self._func_name: str | None = None
        self._args: dict[str, str] = {}
        self._func_closed = False
        self._arg_name: str | None = None
        self._value_parts: list[str] = []
        self._phase = 'function'
        self._stream_started = False
        self._stream_pending_ws = ''
        self._stream_blocked = False

    # Qwen3Coder closes tool argument JSON only when the model emits the
    # explicit function end marker (</function>). We intentionally avoid
    # auto-closing on stream final to prevent producing a syntactically
    # complete but semantically incomplete arguments object.
    def _close_json_on_final(self) -> bool:
        return False

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'xml'

    def _reset_value_stream_state(self) -> None:
        self._stream_started = False
        self._stream_pending_ws = ''
        self._stream_blocked = False

    def _stream_arg_delta(self, raw: str) -> str:
        if self._stream_blocked:
            return ''

        schema_type = self._get_param_schema_type(self._func_name, self._arg_name or '')
        if schema_type not in (None, 'string'):
            self._stream_blocked = True
            return ''

        text = self._stream_pending_ws + raw
        self._stream_pending_ws = ''

        if not self._stream_started:
            text = text.lstrip()
            if not text:
                return ''
            if text.startswith('"'):
                self._stream_blocked = True
                return ''

        stable = text.rstrip()
        self._stream_pending_ws = text[len(stable):]
        if not stable:
            return ''

        self._stream_started = True
        return stable

    def _consume_payload(self, payload: str, *, final: bool) -> tuple[XmlToolSnapshot, int]:
        pos = 0
        arg_delta_parts: list[str] = []
        n = len(payload)

        while pos < n:
            remaining = payload[pos:]

            if self._phase == 'function':
                if self._func_name is not None:
                    self._phase = 'arg_start'
                    continue

                token = '<function='
                if not remaining.startswith(token):
                    idx = remaining.find(token)
                    if idx < 0:
                        if len(remaining) < len(token) and token.startswith(remaining):
                            break
                        break
                    pos += idx
                    continue

                gt = remaining.find('>', len(token))
                if gt < 0:
                    break
                self._func_name = remaining[len(token):gt].strip()
                pos += gt + 1
                self._phase = 'arg_start'
                continue

            if self._phase == 'arg_start':
                param_token = '<parameter='
                close_token = '</function>'

                param_idx = remaining.find(param_token)
                close_idx = remaining.find(close_token)

                if close_idx >= 0 and (param_idx < 0 or close_idx < param_idx):
                    pos += close_idx + len(close_token)
                    self._func_closed = True
                    self._phase = 'done'
                    break

                if param_idx < 0:
                    if len(remaining) < len(param_token) and param_token.startswith(remaining):
                        break
                    break

                if param_idx > 0:
                    pos += param_idx
                    continue

                pos += len(param_token)
                self._phase = 'arg_name'
                continue

            if self._phase == 'arg_name':
                gt = remaining.find('>')
                if gt < 0:
                    break
                self._arg_name = remaining[:gt].strip()
                self._value_parts.clear()
                self._reset_value_stream_state()
                pos += gt + 1
                self._phase = 'arg_value'
                continue

            if self._phase == 'arg_value':
                close = '</parameter>'
                close_idx = remaining.find(close)
                if close_idx >= 0:
                    raw = remaining[:close_idx]
                    if raw:
                        self._value_parts.append(raw)
                    if self._arg_name:
                        self._args[self._arg_name] = ''.join(self._value_parts).strip()
                    self._arg_name = None
                    self._value_parts.clear()
                    self._reset_value_stream_state()
                    pos += close_idx + len(close)
                    self._phase = 'arg_start'
                    continue

                raw_delta = self._strip_partial_xml_close_suffix(remaining, close)
                if not raw_delta:
                    break
                pos += len(raw_delta)
                self._value_parts.append(raw_delta)
                stream_delta = self._stream_arg_delta(raw_delta)
                if stream_delta:
                    arg_delta_parts.append(stream_delta)
                break

            break

        return (
            XmlToolSnapshot(
                self._func_name,
                dict(self._args),
                self._arg_name,
                ''.join(arg_delta_parts),
                self._func_closed,
            ),
            pos,
        )

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict, _ = self._extract_params(payload)
        if not func_name:
            return None
        args_dict = self._get_coerced_args(func_name, raw_args_dict, use_cache=False)
        args_json = json.dumps(args_dict, ensure_ascii=False) if args_dict else '{}'
        return ToolCall(function=FunctionCall(name=func_name, arguments=args_json))

    def _validate_tool_payload(self, payload: str) -> bool:
        return bool(self._complete_payload_pattern.fullmatch(payload))

    def _extract_params(self, content: str) -> tuple[str | None, dict[str, str], bool]:
        """Extract function name, parameter map, and close status from XML."""
        content = content.strip()

        func_name = None
        func_start = content.find('<function=')
        if func_start != -1:
            name_start = func_start + len('<function=')
            name_end = content.find('>', name_start)
            if name_end != -1:
                func_name = content[name_start:name_end].strip()

        args_dict = {}
        search_idx = 0
        while True:
            param_start = content.find('<parameter=', search_idx)
            if param_start == -1:
                break

            name_start = param_start + len('<parameter=')
            name_end = content.find('>', name_start)
            if name_end == -1:
                break

            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find('</parameter>', val_start)
            if val_end == -1:
                break

            args_dict[param_name] = content[val_start:val_end].strip()
            search_idx = val_end + len('</parameter>')

        is_func_closed = '</function>' in content
        return func_name, args_dict, is_func_closed
