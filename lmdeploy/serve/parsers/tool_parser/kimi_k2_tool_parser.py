# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParser, ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


@ToolParserManager.register_module(['kimi_k2', 'kimi-k2'])
class KimiK2ToolParser(ToolParser):
    """Tool parser for the Kimi K2 tool-call section protocol."""

    structural_tag_model = 'kimi'

    section_begin = '<|tool_calls_section_begin|>'
    section_end = '<|tool_calls_section_end|>'
    call_begin = '<|tool_call_begin|>'
    argument_begin = '<|tool_call_argument_begin|>'
    call_end = '<|tool_call_end|>'

    def __init__(self):
        super().__init__()
        self._section_start_index = 0
        self._stream_scan_pos = 0
        self._stream_emitted_calls = 0

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        # Kimi's protocol delimiters must be retained without spaces even for
        # requests that do not provide tools: the same incremental detokenizer
        # also produces the reasoning delimiters.
        request.skip_special_tokens = False
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)

    def start_tool_call(self) -> None:
        super().start_tool_call()
        self._section_start_index = self._active_tool_index
        self._reset_stream_state()

    def finish_tool_call(self) -> None:
        super().finish_tool_call()
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._stream_scan_pos = 0
        self._stream_emitted_calls = 0

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return cls.section_begin

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return cls.section_end

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'kimi-k2'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._tool_payload += added_text
        out: list[DeltaToolCall] = []

        while True:
            parsed = self._parse_next_stream_call(final=final)
            if parsed is None:
                break
            tool_call, end_pos = parsed
            self._stream_scan_pos = end_pos
            if tool_call is None:
                continue
            index = self._section_start_index + self._stream_emitted_calls
            out.extend([
                DeltaToolCall(
                    id=tool_call.id,
                    index=index,
                    type='function',
                    function=DeltaFunctionCall(name=tool_call.function.name),
                ),
                DeltaToolCall(
                    id=None,
                    index=index,
                    type=None,
                    function=DeltaFunctionCall(arguments=tool_call.function.arguments),
                ),
            ])
            self._stream_emitted_calls += 1
            self._active_tool_index = index

        return out

    def _parse_next_stream_call(self, *, final: bool) -> tuple[ToolCall | None, int] | None:
        payload = self._tool_payload
        begin = payload.find(self.call_begin, self._stream_scan_pos)
        if begin < 0:
            return None
        if payload[self._stream_scan_pos:begin].strip():
            self._stream_scan_pos = begin

        id_start = begin + len(self.call_begin)
        arg_begin = payload.find(self.argument_begin, id_start)
        nested_begin = payload.find(self.call_begin, id_start)
        if nested_begin >= 0 and (arg_begin < 0 or nested_begin < arg_begin):
            return None, nested_begin
        if arg_begin < 0:
            return None

        args_start = arg_begin + len(self.argument_begin)
        end = payload.find(self.call_end, args_start)
        if end < 0:
            if not final:
                return None
            end = len(payload)
            end_pos = end
        else:
            end_pos = end + len(self.call_end)

        raw_id = payload[id_start:arg_begin].strip()
        raw_args = payload[args_start:end].strip()
        tool_call = self._build_tool_call(raw_id, raw_args)
        return tool_call, end_pos

    def parse_tool_call_complete(self, payload: str) -> list[ToolCall] | None:
        calls: list[ToolCall] = []
        pos = 0
        while True:
            begin = payload.find(self.call_begin, pos)
            if begin < 0:
                break
            if payload[pos:begin].strip():
                pos = begin

            id_start = begin + len(self.call_begin)
            arg_begin = payload.find(self.argument_begin, id_start)
            nested_begin = payload.find(self.call_begin, id_start)
            if arg_begin < 0 or (nested_begin >= 0 and nested_begin < arg_begin):
                if nested_begin >= 0:
                    pos = nested_begin
                    continue
                break

            args_start = arg_begin + len(self.argument_begin)
            end = payload.find(self.call_end, args_start)
            if end < 0:
                end = len(payload)
                end_pos = end
            else:
                end_pos = end + len(self.call_end)

            raw_id = payload[id_start:arg_begin].strip()
            raw_args = payload[args_start:end].strip()
            tool_call = self._build_tool_call(raw_id, raw_args)
            if tool_call is not None:
                calls.append(tool_call)
            pos = end_pos

        return calls or None

    def _build_tool_call(self, raw_id: str, raw_args: str) -> ToolCall | None:
        name = self._resolve_function_name(raw_id)
        if not name:
            return None
        arguments = raw_args or '{}'
        return ToolCall(id=raw_id, function=FunctionCall(name=name, arguments=arguments))

    @staticmethod
    def _resolve_function_name(raw_id: str) -> str | None:
        prefix, separator, counter = raw_id.rpartition(':')
        if separator and prefix and counter.isdigit():
            if prefix.startswith('functions.'):
                prefix = prefix[len('functions.'):]
            return prefix or None
        return None

    def _validate_tool_payload(self, payload: str) -> bool:
        pos = 0
        while True:
            begin = payload.find(self.call_begin, pos)
            if begin < 0:
                return not payload[pos:].strip()
            if payload[pos:begin].strip():
                return False

            id_start = begin + len(self.call_begin)
            arg_begin = payload.find(self.argument_begin, id_start)
            nested_begin = payload.find(self.call_begin, id_start)
            if arg_begin < 0 or (nested_begin >= 0 and nested_begin < arg_begin):
                return False
            if self._resolve_function_name(payload[id_start:arg_begin].strip()) is None:
                return False

            args_start = arg_begin + len(self.argument_begin)
            end = payload.find(self.call_end, args_start)
            if end < 0:
                return True
            pos = end + len(self.call_end)
