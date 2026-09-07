# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json

from lmdeploy.deepseek_v32_encoding import dsml_token
from lmdeploy.serve.openai.protocol import DeltaToolCall

from .json_value_scanner import JsonValueScanner
from .tool_parser import ToolParser, ToolParserManager

TOOL_CALLS_BLOCK_NAME = 'function_calls'


@ToolParserManager.register_module(['deepseek-v32', 'deepseek-v3.2'])
class DeepSeekV32ToolParser(ToolParser):
    """Incrementally parse DeepSeek-V3.2 DSML function-call blocks.

    Complete responses use this same consumer through the response parser.
    """

    structural_tag_model = 'deepseek_v3_2'
    dsml_token = dsml_token
    tool_calls_block_name = TOOL_CALLS_BLOCK_NAME
    tool_close_prefixes = (
        f'</{dsml_token}{TOOL_CALLS_BLOCK_NAME}',
        f'</{dsml_token}function_c',
        f'</{dsml_token}function',
        f'</{dsml_token}',
        '</',
    )
    _invoke_prefixes = (f'<{dsml_token}inv', f'<{dsml_token}', '<')
    _parameter_prefixes = (f'<{dsml_token}', '<')
    _invoke_close_prefixes = (f'</{dsml_token}invoke', f'</{dsml_token}inv', f'</{dsml_token}', '</')
    _parameter_close_prefixes = (f'</{dsml_token}parameter', f'</{dsml_token}', '</')

    def __init__(self) -> None:
        super().__init__()
        self._phase = 'invoke_start'
        self._current_param_is_string = False
        self._emitted_param_count = 0
        self._value_scanner = JsonValueScanner()

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return f'\n\n<{cls.dsml_token}{cls.tool_calls_block_name}>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return f'</{cls.dsml_token}{cls.tool_calls_block_name}>'

    def begin_tool_block(self) -> None:
        super().begin_tool_block()
        self._phase = 'invoke_start'
        self._current_param_is_string = False
        self._emitted_param_count = 0
        self._value_scanner.reset()

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        del final
        pos = 0
        invoke_tag = f'<{self.dsml_token}invoke'
        parameter_tag = f'<{self.dsml_token}parameter'
        invoke_close_tag = f'</{self.dsml_token}invoke>'
        parameter_close_tag = f'</{self.dsml_token}parameter>'
        section_close_tag = self.get_tool_close_tag()

        while pos < len(text):
            if self._phase == 'invoke_start':
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos == len(text):
                    break
                if text.startswith(section_close_tag, pos):
                    self._payload_closed = True
                    break
                if text.startswith(invoke_tag, pos):
                    pos += len(invoke_tag)
                    self._phase = 'invoke_header'
                    continue
                next_pos = self._next_marker(
                    text,
                    pos,
                    (
                        (invoke_tag, self._invoke_prefixes),
                        (section_close_tag, self.tool_close_prefixes),
                    ),
                )
                if next_pos == pos:
                    break
                pos = next_pos
                continue

            if self._phase == 'invoke_header':
                header_end = text.find('>', pos)
                if header_end < 0:
                    break
                name = self._attribute_value(text[pos:header_end], 'name')
                self._begin_call()
                self._emit_delta(
                    deltas,
                    name=name,
                )
                self._emitted_param_count = 0
                pos = header_end + 1
                self._phase = 'parameter_or_invoke_end'
                continue

            if self._phase == 'parameter_or_invoke_end':
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos == len(text):
                    break
                if text.startswith(invoke_close_tag, pos):
                    self._emit_arguments(deltas, '}' if self._emitted_param_count else '{}')
                    pos += len(invoke_close_tag)
                    self._phase = 'invoke_start'
                    continue
                if text.startswith(parameter_tag, pos):
                    pos += len(parameter_tag)
                    self._phase = 'parameter_header'
                    continue
                next_pos = self._next_marker(
                    text,
                    pos,
                    (
                        (parameter_tag, self._parameter_prefixes),
                        (invoke_close_tag, self._invoke_close_prefixes),
                    ),
                )
                if next_pos == pos:
                    break
                pos = next_pos
                continue

            if self._phase == 'parameter_header':
                header_end = text.find('>', pos)
                if header_end < 0:
                    break
                header = text[pos:header_end]
                param_name = self._attribute_value(header, 'name')
                self._current_param_is_string = self._attribute_value(header, 'string') == 'true'
                prefix = '{' if self._emitted_param_count == 0 else ', '
                quote = '"' if self._current_param_is_string else ''
                self._emit_arguments(
                    deltas,
                    f'{prefix}{json.dumps(param_name, ensure_ascii=False)}: {quote}',
                )
                self._emitted_param_count += 1
                self._value_scanner.reset()
                pos = header_end + 1
                self._phase = 'string_value' if self._current_param_is_string else 'json_value'
                continue

            if self._phase == 'string_value':
                value_end = text.find(parameter_close_tag, pos)
                if value_end >= 0:
                    self._emit_arguments(deltas, self._escape_string(text[pos:value_end]) + '"')
                    pos = value_end + len(parameter_close_tag)
                    self._phase = 'parameter_or_invoke_end'
                    continue
                stable_end = self._stable_prefix_end(text, self._parameter_close_prefixes, pos)
                if stable_end == pos:
                    break
                self._emit_arguments(deltas, self._escape_string(text[pos:stable_end]))
                pos = stable_end
                break

            if self._phase == 'json_value':
                if self._value_scanner.in_string and text.find('"', pos) < 0:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if pos > start:
                        self._emit_arguments(deltas, text[start:pos])
                    break

                marker_at = text.find(parameter_close_tag, pos)
                scan_limit = marker_at if marker_at >= 0 else self._stable_prefix_end(
                    text, self._parameter_close_prefixes, pos)
                start = pos
                pos = self._value_scanner.feed(text, pos, scan_limit)
                if pos > start:
                    self._emit_arguments(deltas, text[start:pos])
                if marker_at >= 0 and pos == marker_at and self._value_scanner.finish_scalar():
                    self._phase = 'parameter_end'
                    continue
                if self._value_scanner.complete:
                    self._phase = 'parameter_end'
                    continue
                if marker_at >= 0:
                    if not self._value_scanner.in_string:
                        self._phase = 'parameter_end'
                        continue
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if pos > start:
                        self._emit_arguments(deltas, text[start:pos])
                    if self._value_scanner.complete:
                        self._phase = 'parameter_end'
                        continue
                elif scan_limit < len(text) and self._value_scanner.in_string:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if pos > start:
                        self._emit_arguments(deltas, text[start:pos])
                    if self._value_scanner.complete:
                        self._phase = 'parameter_end'
                        continue
                break

            if self._phase == 'parameter_end':
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos == len(text):
                    break
                if not text.startswith(parameter_close_tag, pos):
                    next_pos = self._next_marker(
                        text,
                        pos,
                        ((parameter_close_tag, self._parameter_close_prefixes), ),
                    )
                    if next_pos == pos:
                        break
                    pos = next_pos
                    continue
                pos += len(parameter_close_tag)
                self._phase = 'parameter_or_invoke_end'
                continue

        return pos

    def _emit_arguments(self, deltas: list[DeltaToolCall], arguments: str) -> None:
        if arguments:
            self._emit_delta(
                deltas,
                arguments=arguments,
            )

    @staticmethod
    def _escape_string(raw: str) -> str:
        return json.dumps(raw, ensure_ascii=False)[1:-1]

    @staticmethod
    def _attribute_value(header: str, name: str) -> str:
        prefix = f'{name}="'
        start = header.find(prefix)
        if start < 0:
            return ''
        start += len(prefix)
        end = header.find('"', start)
        return header[start:] if end < 0 else header[start:end]

    @staticmethod
    def _next_marker(
        text: str,
        pos: int,
        markers: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> int:
        """Return the next full marker or token-aligned partial marker."""
        next_pos = len(text)
        for marker, marker_prefixes in markers:
            marker_at = text.find(marker, pos)
            if marker_at >= 0:
                if marker_at < next_pos:
                    next_pos = marker_at
            else:
                marker_at = ToolParser._stable_prefix_end(text, marker_prefixes, pos)
                if marker_at < next_pos:
                    next_pos = marker_at
        return next_pos
