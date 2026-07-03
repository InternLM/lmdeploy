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
        self._args: dict[str, str] = {}
        self._func_closed = False
        self._arg_name: str | None = None
        self._value_start = -1
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

    def _parse_payload(self, payload: str, *, final: bool) -> XmlToolSnapshot:
        content = payload.strip()
        if not content:
            return XmlToolSnapshot(self._func_name, dict(self._args), None, '', self._func_closed)

        if self._func_name is None:
            func_start = content.find(self.func_prefix)
            if func_start != -1:
                name_start = func_start + len(self.func_prefix)
                name_end = content.find('>', name_start)
                if name_end != -1:
                    self._func_name = content[name_start:name_end].strip()

        self._parse_params(content)
        self._func_closed = self.func_suffix in content
        arg_name, arg_value = self._open_arg(content)
        return XmlToolSnapshot(self._func_name, dict(self._args), arg_name, arg_value, self._func_closed)

    def _complete_open_arg(self, content: str) -> bool:
        if self._value_start < 0 or not self._arg_name:
            return False
        value_end = content.find(self.param_suffix, self._value_start)
        if value_end == -1:
            self._in_progress_value = True
            return False
        self._args[self._arg_name] = content[self._value_start:value_end].strip()
        self._arg_name = None
        self._value_start = -1
        self._in_progress_value = False
        self._scan_pos = value_end + len(self.param_suffix)
        return True

    def _parse_params(self, content: str) -> None:
        if self._value_start >= 0 and not self._complete_open_arg(content):
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

            arg_name = content[name_start:name_end].strip()
            value_start = name_end + 1
            value_end = content.find(self.param_suffix, value_start)
            if value_end == -1:
                self._arg_name = arg_name
                self._value_start = value_start
                self._in_progress_value = True
                return

            next_pos = value_end + len(self.param_suffix)
            if arg_name not in self._args:
                self._args[arg_name] = content[value_start:value_end].strip()
            self._scan_pos = next_pos
            self._in_progress_value = False

    def _open_arg(self, content: str) -> tuple[str | None, str]:
        if self._arg_name is None or self._value_start < 0:
            return None, ''
        raw_value = self._strip_partial_xml_close_suffix(content[self._value_start:], self.param_suffix)
        return self._arg_name, raw_value.strip()

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

            args_dict[param_name] = content[val_start:val_end].strip()
            search_idx = val_end + len(self.param_suffix)

        is_func_closed = self.func_suffix in content
        return func_name, args_dict, is_func_closed
