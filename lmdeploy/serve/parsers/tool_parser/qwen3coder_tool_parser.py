# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser


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

    def _consume_function(self, payload: str, pos: int, final: bool) -> int | None:
        start = payload.find('<function=', pos)
        if start < 0:
            return None

        name_start = start + len('<function=')
        name_end = payload.find('>', name_start)
        if name_end < 0:
            return None

        self._func_name = payload[name_start:name_end].strip()
        self._phase = 'arg_start'
        return name_end + 1

    def _consume_arg_start(self, payload: str, pos: int) -> int | None:
        param_start = payload.find('<parameter=', pos)
        func_end = payload.find('</function>', pos)

        if func_end >= 0 and (param_start < 0 or func_end < param_start):
            self._payload_closed = True
            self._phase = 'done'
            return func_end + len('</function>')

        if param_start < 0:
            return None

        self._phase = 'arg_name'
        return param_start + len('<parameter=')

    def _consume_arg_name(self, payload: str, pos: int) -> int | None:
        name_end = payload.find('>', pos)
        if name_end < 0:
            return None

        self._arg_name = payload[pos:name_end].strip()
        self._value_parts.clear()
        self._reset_value_stream_state()
        self._phase = 'arg_value'
        return name_end + 1

    def _consume_arg_value(self, payload: str, pos: int, arg_delta_parts: list[str]) -> tuple[int | None, bool]:
        """Consume a parameter value.

        Returns ``(next_pos, should_stop)``. ``should_stop`` is true after
        streaming an open value delta, because the next bytes may be the
        parameter close tag and must be checked with the next chunk.
        """
        value_end = payload.find('</parameter>', pos)

        if value_end >= 0:
            raw = payload[pos:value_end]
            if raw:
                self._value_parts.append(raw)
            if self._arg_name:
                self._args[self._arg_name] = ''.join(self._value_parts).strip()
            self._arg_name = None
            self._value_parts.clear()
            self._reset_value_stream_state()
            self._phase = 'arg_start'
            return value_end + len('</parameter>'), False

        # Open value: keep any partial "</parameter>" suffix buffered instead
        # of emitting it as argument text.
        raw_end = self._trim_partial_close_tag_suffix(payload, pos, '</parameter>')
        if raw_end == pos:
            return None, True

        raw_delta = payload[pos:raw_end]
        self._value_parts.append(raw_delta)
        stream_delta = self._stream_arg_delta(raw_delta)
        if stream_delta:
            arg_delta_parts.append(stream_delta)
        return raw_end, True

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
