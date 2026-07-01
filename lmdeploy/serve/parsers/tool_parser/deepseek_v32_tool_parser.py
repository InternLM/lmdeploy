# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

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

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return f'\n\n<{cls.dsml_token}{cls.tool_calls_block_name}>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return f'</{cls.dsml_token}{cls.tool_calls_block_name}>'

    @classmethod
    def get_tool_payload_format(cls) -> str:
        return 'dsml'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._tool_payload += added_text
        if not final:
            return []

        tool_calls = self.parse_tool_call_complete(self._tool_payload)
        if not tool_calls:
            return []

        out: list[DeltaToolCall] = []
        for offset, tool_call in enumerate(tool_calls):
            index = self._active_tool_index + offset
            out.append(
                DeltaToolCall(
                    id=f'chatcmpl-tool-{shortuuid.random()}',
                    index=index,
                    type='function',
                    function=DeltaFunctionCall(name=tool_call.function.name),
                ))
            out.append(
                DeltaToolCall(
                    id=None,
                    index=index,
                    type=None,
                    function=DeltaFunctionCall(arguments=tool_call.function.arguments),
                ))

        self._active_tool_index += len(tool_calls) - 1
        return out

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
