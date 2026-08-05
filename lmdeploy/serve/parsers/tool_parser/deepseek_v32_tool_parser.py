# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import re

import shortuuid

from lmdeploy.deepseek_v32_encoding import dsml_token, parse_tool_calls
from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParser, ToolParserManager

TOOL_CALLS_BLOCK_NAME = 'function_calls'


@ToolParserManager.register_module(['deepseek-v32', 'deepseek-v3.2'])
class DeepSeekV32ToolParser(ToolParser):
    """Tool parser for DeepSeek-V3.2 DSML function-call blocks."""

    dsml_token = dsml_token
    tool_calls_block_name = TOOL_CALLS_BLOCK_NAME
    parse_tool_calls_func = staticmethod(parse_tool_calls)

    def __init__(self):
        super().__init__()
        self._buffer = ''
        self._phase = 'invoke_start'
        self._invoke_count = 0
        self._current_tool_index = -1
        self._current_param_is_string = False
        self._emitted_param_names: set[str] = set()

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return f'\n\n<{cls.dsml_token}{cls.tool_calls_block_name}>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return f'</{cls.dsml_token}{cls.tool_calls_block_name}>'

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'dsml'

    def start_tool_call(self) -> None:
        super().start_tool_call()
        self._reset_stream_state()

    def finish_tool_call(self) -> None:
        if self._invoke_count > 0:
            self._active_tool_index += self._invoke_count - 1
        super().finish_tool_call()
        self._reset_stream_state()

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Emit each DSML function name and parameter fragment immediately."""
        self._buffer += added_text
        out: list[DeltaToolCall] = []
        pos = 0
        invoke_tag = f'<{self.dsml_token}invoke'
        parameter_tag = f'<{self.dsml_token}parameter'
        invoke_close_tag = f'</{self.dsml_token}invoke>'
        parameter_close_tag = f'</{self.dsml_token}parameter>'

        while pos < len(self._buffer):
            if self._phase == 'invoke_start':
                start = self._buffer.find(invoke_tag, pos)
                if start < 0:
                    pos = self._trim_partial_marker_suffix(self._buffer, pos, (invoke_tag, ))
                    break
                pos = start + len(invoke_tag)
                self._phase = 'invoke_header'
                continue

            if self._phase == 'invoke_header':
                header_end = self._buffer.find('>\n', pos)
                if header_end < 0:
                    break
                header = self._buffer[pos:header_end]
                match = re.fullmatch(r'\s*name="(.*?)"', header, flags=re.DOTALL)
                if match is None:
                    break
                self._current_tool_index = self._active_tool_index + self._invoke_count
                tool_id = (
                    self._active_tool_call_id
                    if self._invoke_count == 0 else f'chatcmpl-tool-{shortuuid.random()}'
                )
                out.append(
                    DeltaToolCall(
                        id=tool_id,
                        index=self._current_tool_index,
                        type='function',
                        function=DeltaFunctionCall(name=match.group(1)),
                    ))
                self._emitted_param_names.clear()
                pos = header_end + 2
                self._phase = 'parameter_or_invoke_end'
                continue

            if self._phase == 'parameter_or_invoke_end':
                parameter_start = self._buffer.find(parameter_tag, pos)
                invoke_end = self._buffer.find(invoke_close_tag, pos)
                if invoke_end >= 0 and (parameter_start < 0 or invoke_end < parameter_start):
                    self._append_arguments_delta(out, '}' if self._emitted_param_names else '{}')
                    self._invoke_count += 1
                    self._current_tool_index = -1
                    pos = invoke_end + len(invoke_close_tag)
                    self._phase = 'invoke_start'
                    continue
                if parameter_start < 0:
                    pos = self._trim_partial_marker_suffix(
                        self._buffer,
                        pos,
                        (parameter_tag, invoke_close_tag),
                    )
                    break
                pos = parameter_start + len(parameter_tag)
                self._phase = 'parameter_header'
                continue

            if self._phase == 'parameter_header':
                header_end = self._buffer.find('>', pos)
                if header_end < 0:
                    break
                header = self._buffer[pos:header_end]
                match = re.fullmatch(r'\s*name="(.*?)"\s+string="(true|false)"', header, flags=re.DOTALL)
                if match is None:
                    break
                param_name, string_flag = match.groups()
                prefix = '{' if not self._emitted_param_names else ', '
                key = json.dumps(param_name, ensure_ascii=False)
                quote = '"' if string_flag == 'true' else ''
                self._append_arguments_delta(out, f'{prefix}{key}: {quote}')
                self._emitted_param_names.add(param_name)
                self._current_param_is_string = string_flag == 'true'
                pos = header_end + 1
                self._phase = 'parameter_value'
                continue

            if self._phase == 'parameter_value':
                value_end = self._buffer.find(parameter_close_tag, pos)
                if value_end >= 0:
                    value_delta = self._encode_param_delta(self._buffer[pos:value_end])
                    if self._current_param_is_string:
                        value_delta += '"'
                    self._append_arguments_delta(out, value_delta)
                    pos = value_end + len(parameter_close_tag)
                    self._phase = 'parameter_or_invoke_end'
                    continue

                raw_end = self._trim_partial_marker_suffix(self._buffer, pos, (parameter_close_tag, ))
                if raw_end == pos:
                    break
                self._append_arguments_delta(out, self._encode_param_delta(self._buffer[pos:raw_end]))
                pos = raw_end
                break

            break

        if pos > 0:
            self._buffer = self._buffer[pos:]
        return out

    def _append_arguments_delta(self, out: list[DeltaToolCall], arguments: str) -> None:
        if not arguments:
            return
        out.append(
            DeltaToolCall(
                id=None,
                index=self._current_tool_index,
                type=None,
                function=DeltaFunctionCall(arguments=arguments),
            ))

    def _encode_param_delta(self, raw: str) -> str:
        if not self._current_param_is_string:
            return raw
        return json.dumps(raw, ensure_ascii=False)[1:-1]

    @staticmethod
    def _trim_partial_marker_suffix(payload: str, start: int, markers: tuple[str, ...]) -> int:
        """Keep only a suffix that can grow into one of ``markers``."""
        keep_from = len(payload)
        for marker in markers:
            max_len = min(len(payload) - start, len(marker) - 1)
            for suffix_len in range(max_len, 0, -1):
                suffix_start = len(payload) - suffix_len
                if marker.startswith(payload[suffix_start:]):
                    keep_from = min(keep_from, suffix_start)
                    break
        return keep_from

    def _reset_stream_state(self) -> None:
        self._buffer = ''
        self._phase = 'invoke_start'
        self._invoke_count = 0
        self._current_tool_index = -1
        self._current_param_is_string = False
        self._emitted_param_names.clear()

    def parse_tool_call_complete(self, payload: str) -> list[ToolCall] | None:
        payload = payload.strip()
        if not payload:
            return None

        wrapped = f'{self.get_tool_open_tag()}\n{payload}\n{self.get_tool_close_tag()}'
        start = len(self.get_tool_open_tag()) - 1
        try:
            _, stop_token, raw_tool_calls = self.parse_tool_calls_func(start, wrapped)
        except Exception:
            return None
        if stop_token != self.get_tool_close_tag() or not raw_tool_calls:
            return None

        return [
            ToolCall(function=FunctionCall(name=tool_call['name'], arguments=tool_call['arguments']))
            for tool_call in raw_tool_calls
        ]

    def validate_complete(self, text: str) -> bool:
        open_tag = self.get_tool_open_tag()
        close_tag = self.get_tool_close_tag()

        pos = 0
        while True:
            open_idx = text.find(open_tag, pos)
            close_idx = text.find(close_tag, pos)
            if open_idx < 0:
                return close_idx < 0

            payload_start = open_idx + len(open_tag)
            if close_idx < payload_start:
                return False
            if self.parse_tool_call_complete(text[payload_start:close_idx]) is None:
                return False

            pos = close_idx + len(close_tag)
            if pos >= len(text):
                return True
