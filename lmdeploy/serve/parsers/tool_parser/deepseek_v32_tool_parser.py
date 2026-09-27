# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json

from lmdeploy.deepseek_v32_encoding import dsml_token as DSML_TOKEN
from lmdeploy.serve.openai.protocol import DeltaToolCall

from .json_value_scanner import JsonValueScanner
from .tool_parser import ToolParser, ToolParserManager


@ToolParserManager.register_module('deepseek-v32')
class DeepSeekV32ToolParser(ToolParser):
    """Incrementally parse DeepSeek-V3.2 DSML function-call blocks.

    Complete responses use this same consumer through the response parser.
    A function name is available in the ``invoke`` header, so its identity is
    emitted before any parameter fragments. String parameters contain raw text
    and end at their closing tag; non-string parameters contain JSON and use a
    lexical scanner to find the value boundary.

    The parser moves through these phases::

        invoke_start -- <invoke ...> --> invoke_header
        invoke_header -- header end --> parameter_or_invoke_end
        parameter_or_invoke_end -- <parameter ...> --> parameter_header
        parameter_header -- string="true" --> string_value
        parameter_header -- otherwise --> json_value
        string_value -- </parameter> --> parameter_or_invoke_end
        json_value -- value end --> parameter_end
        parameter_end -- </parameter> --> parameter_or_invoke_end
        parameter_or_invoke_end -- </invoke> --> invoke_start
        invoke_start -- </function_calls> --> payload closed

    ``ToolParser`` owns the surrounding block lifecycle and consumes the outer
    ``function_calls`` closing tag after this parser marks the payload closed.
    """

    structural_tag_model = 'deepseek_v3_2'
    dsml_token = DSML_TOKEN
    tool_calls_block_name = 'function_calls'

    # Proper marker prefixes possible at decoded-token boundaries.
    tool_close_prefixes = (
        f'</{dsml_token}{tool_calls_block_name}',
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
        self._emitted_param_count = 0
        self._value_scanner.reset()

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        """Consume the stable prefix of a DSML payload and append tool
        deltas."""
        del final
        pos = 0
        invoke_tag = f'<{self.dsml_token}invoke'
        parameter_tag = f'<{self.dsml_token}parameter'
        invoke_close_tag = f'</{self.dsml_token}invoke>'
        parameter_close_tag = f'</{self.dsml_token}parameter>'
        section_close_tag = self.get_tool_close_tag()
        size = len(text)

        while pos < size:
            if self._phase == 'invoke_start':
                while pos < size and text[pos].isspace():
                    pos += 1
                if pos == size:
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
                while pos < size and text[pos].isspace():
                    pos += 1
                if pos == size:
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
                is_string = self._attribute_value(header, 'string') == 'true'
                prefix = '{' if self._emitted_param_count == 0 else ', '
                quote = '"' if is_string else ''
                self._emit_arguments(
                    deltas,
                    f'{prefix}{json.dumps(param_name, ensure_ascii=False)}: {quote}',
                )
                self._emitted_param_count += 1
                self._value_scanner.reset()
                pos = header_end + 1
                self._phase = 'string_value' if is_string else 'json_value'
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
                if self._value_scanner.complete:
                    self._phase = 'parameter_end'
                    continue
                if marker_at >= 0 and not self._value_scanner.in_string:
                    # The DSML boundary also terminates malformed JSON values.
                    self._phase = 'parameter_end'
                    continue
                if scan_limit < size and self._value_scanner.in_string:
                    start = pos
                    pos = self._value_scanner.feed(text, pos)
                    if pos > start:
                        self._emit_arguments(deltas, text[start:pos])
                    if self._value_scanner.complete:
                        self._phase = 'parameter_end'
                        continue
                break

            if self._phase == 'parameter_end':
                while pos < size and text[pos].isspace():
                    pos += 1
                if pos == size:
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
            if marker_at < 0:
                marker_at = ToolParser._stable_prefix_end(text, marker_prefixes, pos)
            next_pos = min(next_pos, marker_at)
        return next_pos
