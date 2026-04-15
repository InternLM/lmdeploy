# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import partial_json_parser
from partial_json_parser.core.options import Allow

from lmdeploy.serve.openai.protocol import DeltaFunctionCall, DeltaToolCall

from .tool_parser import ToolParser, ToolParserManager

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import (
        ChatCompletionRequest,
        DeltaToolCall,
        ToolCall,
    )

@ToolParserManager.register_module(['internlm', 'intern-s1'])
class Internlm2ToolParser(ToolParser):
    """Tool parser for InternLM JSON tool-call payloads."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.plugin_tag = '<|plugin|>'

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

    def get_tool_open_tag(self) -> str | None:
        # Streaming chunks may split ``<|action_start|>`` and ``<|plugin|>``
        # across multiple deltas, so enter tool mode right after action_start.
        return '<|action_start|>'

    def get_tool_close_tag(self) -> str | None:
        return '<|action_end|>'

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental JSON tool payload.

        InternLM streams often emit ``<|plugin|>`` as a separate delta before
        JSON starts. Strip that marker from the accumulated payload before JSON
        incremental parsing.
        """
        self._tool_payload += added_text
        payload = self._tool_payload.lstrip()
        if payload.startswith(self.plugin_tag):
            payload = payload[len(self.plugin_tag):].lstrip('\n')
        # Wait until the full plugin marker arrives when it is chunk-split.
        elif self.plugin_tag.startswith(payload) or payload.startswith(self.plugin_tag[:3]):
            return []

        if not payload:
            return []

        flags = Allow.ALL if self._name_emitted else Allow.ALL & ~Allow.STR
        try:
            obj = partial_json_parser.loads(payload, flags)
        except partial_json_parser.core.exceptions.MalformedJSON:
            return []
        if not isinstance(obj, dict):
            return []

        out: list[DeltaToolCall] = []
        if not self._name_emitted:
            fn_name = obj.get('name')
            if isinstance(fn_name, str) and fn_name:
                out.append(
                    DeltaToolCall(
                        id=self._active_tool_call_id,
                        index=self._active_tool_index,
                        type='function',
                        function=DeltaFunctionCall(name=fn_name),
                    ))
                self._name_emitted = True

        args_obj = obj.get('arguments', obj.get('parameters', None))
        if args_obj is None:
            return out

        import json
        args_json = json.dumps(args_obj, ensure_ascii=False)
        if args_json in ('{}', '[]'):
            return out

        if final and len(args_json) > self._args_emitted_len:
            diff = args_json[self._args_emitted_len:]
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=diff),
                ))
            self._args_emitted_len = len(args_json)
        return out

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
