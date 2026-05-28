# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
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

    def _extract_incremental_state(self, payload: str, final: bool = False) -> tuple[str | None, dict[str, Any], bool]:
        return self._extract_params(payload)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict, _ = self._extract_params(payload)
        if not func_name:
            return None
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
        args_json = json.dumps(args_dict, ensure_ascii=False) if args_dict else '{}'
        return ToolCall(function=FunctionCall(name=func_name, arguments=args_json))

    def _extract_params(self, content: str) -> tuple[str | None, dict[str, Any], bool]:
        """Extract function name, parameter map, and close status from XML."""
        content = content.replace('<tool_call>', '').replace('</tool_call>', '').strip()

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

            # Qwen3Coder XML payloads do not carry explicit type metadata.
            # Keep parameter values as strings to avoid implicit type coercion
            # (e.g., zip codes like 77004 being parsed into integers).
            try:
                parsed_val = json.loads(param_val_str)
                val = parsed_val if isinstance(parsed_val, str) else param_val_str
            except json.JSONDecodeError:
                val = param_val_str
            args_dict[param_name] = val
            search_idx = val_end + len(self.param_end_token)

        is_func_closed = self.func_end_token in content
        return func_name, args_dict, is_func_closed
