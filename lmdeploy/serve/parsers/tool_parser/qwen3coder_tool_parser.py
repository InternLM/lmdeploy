# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlParseResult, XmlToolParser


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(XmlToolParser):
    """Tool parser for Qwen3Coder XML tool-call payloads."""

    _complete_payload_pattern = re.compile(
        r'^\s*<function=[^\s>\n]+>\s*(?:<parameter=[^\s>\n]+>.*?</parameter>\s*)*</function>\s*$',
        re.DOTALL,
    )

    def _reset_incremental_state(self) -> None:
        self._value_parts: list[str] = []
        self._reset_value_stream_state()

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

    def _block_stream_arg_delta(self) -> None:
        self._stream_blocked = True

    def _normalize_stream_arg_delta(self, raw: str, schema_type: str | None) -> str:
        if self._stream_blocked:
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

    def _consume_function(self, payload: str, pos: int, final: bool) -> XmlParseResult:
        """Read a Qwen ``<function=name>`` opener and publish the function
        name.

        The parser waits when either the opener or its closing ``>`` is split
        across chunks.
        """
        start = payload.find('<function=', pos)
        if start < 0:
            return XmlParseResult(next_pos=None)

        name_start = start + len('<function=')
        name_end = payload.find('>', name_start)
        if name_end < 0:
            return XmlParseResult(next_pos=None)

        return XmlParseResult(
            next_pos=name_end + 1,
            next_phase='arg_start',
            func_name=payload[name_start:name_end].strip(),
        )

    def _consume_arg_start(self, payload: str, pos: int) -> XmlParseResult:
        """Find the next Qwen parameter opener or the function close marker.

        ``</function>`` closes the XML payload only when it appears before the next
        ``<parameter=`` marker.
        """
        param_start = payload.find('<parameter=', pos)
        func_end = payload.find('</function>', pos)

        if func_end >= 0 and (param_start < 0 or func_end < param_start):
            return XmlParseResult(
                next_pos=func_end + len('</function>'),
                next_phase='done',
                payload_closed=True,
            )

        if param_start < 0:
            return XmlParseResult(next_pos=None)

        return XmlParseResult(
            next_pos=param_start + len('<parameter='),
            next_phase='arg_name',
        )

    def _consume_arg_name(self, payload: str, pos: int) -> XmlParseResult:
        """Read the Qwen parameter name and advance to its value body."""
        name_end = payload.find('>', pos)
        if name_end < 0:
            return XmlParseResult(next_pos=None)

        self._value_parts.clear()
        self._reset_value_stream_state()
        return XmlParseResult(
            next_pos=name_end + 1,
            next_phase='arg_value',
            arg_name=payload[pos:name_end].strip(),
        )

    def _consume_arg_value(self, payload: str, pos: int) -> XmlParseResult:
        """Consume Qwen parameter value text.

        Closed values are stripped to preserve Qwen's current XML formatting
        behavior. Open values leave a possible partial ``</parameter>`` suffix
        buffered so split close tags are not emitted as argument text.
        """
        value_end = payload.find('</parameter>', pos)

        if value_end >= 0:
            raw = payload[pos:value_end]
            if raw:
                self._value_parts.append(raw)
            value = ''.join(self._value_parts).strip()
            self._value_parts.clear()
            self._reset_value_stream_state()
            return XmlParseResult(
                next_pos=value_end + len('</parameter>'),
                next_phase='arg_start',
                completed_arg_value=value,
            )

        raw_end = self._trim_partial_close_tag_suffix(payload, pos, '</parameter>')
        if raw_end == pos:
            return XmlParseResult(next_pos=None, should_stop=True)

        raw_delta = payload[pos:raw_end]
        self._value_parts.append(raw_delta)
        return XmlParseResult(
            next_pos=raw_end,
            raw_arg_delta=raw_delta,
            should_stop=True,
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
