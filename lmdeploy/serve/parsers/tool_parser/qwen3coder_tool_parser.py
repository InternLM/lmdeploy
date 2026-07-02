# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re
from typing import Any

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(XmlToolParser):
    """Tool parser for Qwen3Coder XML tool-call payloads."""

    func_prefix = '<function='
    func_suffix = '</function>'
    param_prefix = '<parameter='
    param_suffix = '</parameter>'
    _complete_payload_pattern = re.compile(
        r'^\s*<function=[^\s>\n]+>\s*(?:<parameter=[^\s>\n]+>.*?</parameter>\s*)*</function>\s*$',
        re.DOTALL,
    )

    def _reset_incremental_state(self) -> None:
        self._func_name: str | None = None
        self._args: dict[str, Any] = {}
        self._func_closed = False
        self._open_param_name: str | None = None
        # Offset in accumulated payload where the in-flight parameter value begins
        # (first char after ``>`` in ``<parameter=name>``); -1 when none is open.
        self._value_start = -1
        # Resume parameter scanning after the last completed ``</parameter>``.
        self._scan_pos = 0

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

    def _extract_incremental_state(self,
                                   payload: str,
                                   final: bool = False) -> tuple[str | None, dict[str, Any], bool]:
        """Update streaming parse state from accumulated inner tool payload.

        ``payload`` is the text inside ``<tool_call>...</tool_call>`` (outer tags
        are stripped by :class:`BaseResponseParser` before tool mode). This
        method mutates incremental parse state across chunks and returns the current
        snapshot for :meth:`XmlToolParser.decode_tool_incremental`.

        Returns:
            ``(func_name, args_dict, is_func_closed)`` where:

            - ``func_name``: callee parsed from ``<function=...>``, or ``None``
            - ``args_dict``: parameters whose ``</parameter>`` has been seen
            - ``is_func_closed``: whether ``</function>`` is present; used to
              emit the closing ``}`` of streamed OpenAI arguments JSON
        """
        content = payload.strip()
        if not content:
            return self._func_name, dict(self._args), self._func_closed

        if self._func_name is None:
            func_start = content.find(self.func_prefix)
            if func_start != -1:
                name_start = func_start + len(self.func_prefix)
                name_end = content.find('>', name_start)
                if name_end != -1:
                    self._func_name = content[name_start:name_end].strip()

        self._parse_params_incremental(content)
        self._func_closed = self.func_suffix in content
        return self._func_name, dict(self._args), self._func_closed

    def _complete_open_param_if_ready(self, content: str) -> bool:
        """Finalize the in-flight parameter once ``</parameter>`` is available.

        Uses ``_value_start`` so we can locate the closing tag without re-parsing
        the ``<parameter=name>`` header on every non-fast-path chunk.
        """
        if self._value_start < 0 or not self._open_param_name:
            return False
        val_end = content.find(self.param_suffix, self._value_start)
        if val_end == -1:
            self._in_progress_value = True
            return False
        param_val_str = content[self._value_start:val_end].strip()
        self._args[self._open_param_name] = self._parse_param_value(param_val_str)
        self._open_param_name = None
        self._value_start = -1
        self._in_progress_value = False
        self._scan_pos = val_end + len(self.param_suffix)
        return True

    def _parse_params_incremental(self, content: str) -> None:
        """Scan ``<parameter=name>value</parameter>`` blocks and update
        ``_args``.

        Incomplete parameter headers or values are left open in ``_open_param_name``
        / ``_value_start`` until the closing tag arrives in a later stream chunk.

        ``_scan_pos`` only advances past completed ``</parameter>`` tags; while a
        value is streaming, it stays before the open tag. The block below is an
        optimization (not required for correctness): skip the while-loop header
        re-scan and try to close the current value directly. If ``</parameter>``
        is still missing, return early because the while loop would reach the
        same open-tag state anyway.
        """
        if self._value_start >= 0:
            if not self._complete_open_param_if_ready(content):
                return

        while True:
            param_start = content.find(self.param_prefix, self._scan_pos)
            if param_start == -1:
                self._in_progress_value = False
                return

            name_start = param_start + len(self.param_prefix)
            name_end = content.find('>', name_start)
            if name_end == -1:
                self._in_progress_value = True
                return

            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_suffix, val_start)
            if val_end == -1:
                self._open_param_name = param_name
                self._value_start = val_start
                self._in_progress_value = True
                return

            next_pos = val_end + len(self.param_suffix)
            if param_name in self._args:
                self._scan_pos = next_pos
                continue

            param_val_str = content[val_start:val_end].strip()
            self._args[param_name] = self._parse_param_value(param_val_str)
            self._scan_pos = next_pos
            self._in_progress_value = False

    @staticmethod
    def _parse_param_value(param_val_str: str) -> Any:
        try:
            parsed_val = json.loads(param_val_str)
            return parsed_val if isinstance(parsed_val, str) else param_val_str
        except json.JSONDecodeError:
            return param_val_str

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict, _ = self._extract_params(payload)
        if not func_name:
            return None
        args_dict = self._get_coerced_args(func_name, raw_args_dict, use_cache=False)
        args_json = json.dumps(args_dict, ensure_ascii=False) if args_dict else '{}'
        return ToolCall(function=FunctionCall(name=func_name, arguments=args_json))

    def _validate_tool_payload(self, payload: str) -> bool:
        return bool(self._complete_payload_pattern.fullmatch(payload))

    def _extract_params(self, content: str) -> tuple[str | None, dict[str, Any], bool]:
        """Extract function name, parameter map, and close status from XML."""
        content = content.strip()

        func_name = None
        func_start = content.find(self.func_prefix)
        if func_start != -1:
            name_start = func_start + len(self.func_prefix)
            name_end = content.find('>', name_start)
            if name_end != -1:
                func_name = content[name_start:name_end].strip()

        args_dict = {}
        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                break

            name_start = param_start + len(self.param_prefix)
            name_end = content.find('>', name_start)
            if name_end == -1:
                break

            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_suffix, val_start)
            if val_end == -1:
                break

            param_val_str = content[val_start:val_end].strip()
            args_dict[param_name] = self._parse_param_value(param_val_str)
            search_idx = val_end + len(self.param_suffix)

        is_func_closed = self.func_suffix in content
        return func_name, args_dict, is_func_closed
