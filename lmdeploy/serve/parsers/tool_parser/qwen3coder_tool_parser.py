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
    func_end_token = '</function>'
    param_prefix = '<parameter='
    param_end_token = '</parameter>'
    _complete_payload_pattern = re.compile(
        r'^\s*<function=[^\s>\n]+>\s*(?:<parameter=[^\s>\n]+>.*?</parameter>\s*)*</function>\s*$',
        re.DOTALL,
    )

    def _value_close_token(self) -> str:
        return self.param_end_token

    def _reset_incremental_state(self) -> None:
        self._qwen_func_name: str | None = None
        self._qwen_args: dict[str, Any] = {}
        self._qwen_param_scan_pos = 0
        self._qwen_func_closed = False
        self._qwen_open_param_name: str | None = None
        self._qwen_value_start = -1

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
        content = payload.strip()
        if not content:
            return self._qwen_func_name, dict(self._qwen_args), self._qwen_func_closed

        if self._qwen_func_name is None:
            func_start = content.find(self.func_prefix)
            if func_start != -1:
                name_start = func_start + len(self.func_prefix)
                terminators = [
                    idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1
                ]
                if terminators:
                    self._qwen_func_name = content[name_start:min(terminators)].strip()

        self._parse_params_incremental(content)
        self._qwen_func_closed = self.func_end_token in content
        return self._qwen_func_name, dict(self._qwen_args), self._qwen_func_closed

    def _complete_open_param_if_ready(self, content: str) -> bool:
        if self._qwen_value_start < 0 or not self._qwen_open_param_name:
            return False
        val_end = content.find(self.param_end_token, self._qwen_value_start)
        if val_end == -1:
            self._in_progress_value = True
            return False
        param_val_str = content[self._qwen_value_start:val_end].strip()
        self._qwen_args[self._qwen_open_param_name] = self._parse_param_value(param_val_str)
        self._qwen_param_scan_pos = val_end + len(self.param_end_token)
        self._qwen_open_param_name = None
        self._qwen_value_start = -1
        self._in_progress_value = False
        return True

    def _parse_params_incremental(self, content: str) -> None:
        if self._qwen_value_start >= 0:
            if not self._complete_open_param_if_ready(content):
                return

        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                self._qwen_param_scan_pos = len(content)
                self._in_progress_value = False
                return

            name_start = param_start + len(self.param_prefix)
            terminators = [
                idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1
            ]
            if not terminators:
                self._qwen_param_scan_pos = param_start
                self._in_progress_value = True
                return

            name_end = min(terminators)
            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_end_token, val_start)
            if val_end == -1:
                self._qwen_open_param_name = param_name
                self._qwen_value_start = val_start
                self._qwen_param_scan_pos = val_start
                self._in_progress_value = True
                return

            if param_name in self._qwen_args:
                search_idx = val_end + len(self.param_end_token)
                self._qwen_param_scan_pos = search_idx
                continue

            param_val_str = content[val_start:val_end].strip()
            self._qwen_args[param_name] = self._parse_param_value(param_val_str)
            search_idx = val_end + len(self.param_end_token)
            self._qwen_param_scan_pos = search_idx
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
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
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
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if terminators:
                func_name = content[name_start:min(terminators)].strip()

        args_dict = {}
        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                break

            name_start = param_start + len(self.param_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if not terminators:
                break

            name_end = min(terminators)
            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_end_token, val_start)
            if val_end == -1:
                break

            param_val_str = content[val_start:val_end].strip()
            args_dict[param_name] = self._parse_param_value(param_val_str)
            search_idx = val_end + len(self.param_end_token)

        is_func_closed = self.func_end_token in content
        return func_name, args_dict, is_func_closed
