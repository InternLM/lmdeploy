# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/tool_parsers
import json
from functools import cached_property

import partial_json_parser
import shortuuid
from mmengine import Registry
from partial_json_parser.core.options import Allow

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.openai.tool_parser'])


class ToolParser:
    """Abstract ToolParser class that should not be used directly.

    Provided properties and methods should be used in derived classes.
    """

    def __init__(self, tokenizer: object):
        self.model_tokenizer = tokenizer
        self._tool_payload: str = ''
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False
        self._args_prefix_emitted: bool = False
        self._value_chars_emitted: int = 0
        self._args_closed_emitted: bool = False
        self._args_emitted_len: int = 0
        self._prev_args_json: str | None = None

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Static method that used to adjust the request parameters."""
        if request.tools is not None and request.tool_choice != 'none':
            if not isinstance(request.tool_choice, str):
                request.tools = [
                    item.function.model_dump() for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                request.tools = [item.function.model_dump() for item in request.tools]
        return request

    def get_tool_open_tag(self) -> str | None:
        """Return tool opening tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_open_tag has not been implemented!')

    def get_tool_close_tag(self) -> str | None:
        """Return tool closing tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_close_tag has not been implemented!')

    def get_tool_payload_format(self) -> str:
        """Return payload format for tool call body."""
        raise NotImplementedError('ToolParser.get_tool_payload_format has not been implemented!')

    def start_tool_call(self) -> None:
        """Mark start of a tool-call block."""
        self._active_tool_index += 1
        self._active_tool_call_id = f'chatcmpl-tool-{shortuuid.random()}'
        self._name_emitted = False
        self._args_prefix_emitted = False
        self._value_chars_emitted = 0
        self._args_closed_emitted = False
        self._args_emitted_len = 0
        self._prev_args_json = None
        self._tool_payload = ''

    def finish_tool_call(self) -> None:
        """Mark end of a tool-call block."""
        self._active_tool_call_id = ''
        self._name_emitted = False
        self._args_prefix_emitted = False
        self._value_chars_emitted = 0
        self._args_closed_emitted = False
        self._args_emitted_len = 0
        self._prev_args_json = None
        self._tool_payload = ''

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental tool payload emitted between tool tags."""
        raise NotImplementedError('ToolParser.decode_tool_incremental has not been implemented!')

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        """Parse one complete tool payload into OpenAI tool call object."""
        raise NotImplementedError('ToolParser.parse_tool_call_complete has not been implemented!')

    def _decode_tool_incremental_json(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._tool_payload += added_text
        payload = self._tool_payload.strip()
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

        if isinstance(args_obj, dict):
            items = list(args_obj.items())
            if not self._args_prefix_emitted and items:
                first_key = items[0][0]
                out.append(
                    DeltaToolCall(
                        id=self._active_tool_call_id,
                        index=self._active_tool_index,
                        type=None,
                        function=DeltaFunctionCall(arguments=f'{{\"{first_key}\": \"')),
                )
                self._args_prefix_emitted = True

            values_concat = ''.join(v for _, v in items if isinstance(v, str))
            if len(values_concat) > self._value_chars_emitted:
                diff = values_concat[self._value_chars_emitted:]
                out.append(
                    DeltaToolCall(
                        id=self._active_tool_call_id,
                        index=self._active_tool_index,
                        type=None,
                        function=DeltaFunctionCall(arguments=diff),
                    ))
                self._value_chars_emitted = len(values_concat)

            if self._is_complete_json(payload) and self._args_prefix_emitted and not self._args_closed_emitted:
                out.append(
                    DeltaToolCall(
                        id=self._active_tool_call_id,
                        index=self._active_tool_index,
                        type=None,
                        function=DeltaFunctionCall(arguments='"}'),
                    ))
                self._args_closed_emitted = True
            return out

        args_json = json.dumps(args_obj, ensure_ascii=False)
        if args_json in ('{}', '[]'):
            return out

        emitted_arg = False
        candidate: str | None = None
        if self._is_complete_json(payload):
            candidate = args_json
        elif self._prev_args_json:
            candidate = self._common_prefix(self._prev_args_json, args_json)
        elif self._args_emitted_len == 0 and added_text:
            pos = args_json.find(added_text)
            if pos >= 0:
                candidate = args_json[:pos + len(added_text)]

        if candidate and len(candidate) > self._args_emitted_len:
            diff = candidate[self._args_emitted_len:]
            if final or any(ch.isalnum() for ch in diff):
                out.append(
                    DeltaToolCall(
                        id=self._active_tool_call_id,
                        index=self._active_tool_index,
                        type=None,
                        function=DeltaFunctionCall(arguments=diff),
                    ))
                self._args_emitted_len = len(candidate)
                emitted_arg = True

        if (
            not emitted_arg
            and self._args_emitted_len > 0
            and added_text
            and any(ord(ch) > 127 for ch in added_text)
        ):
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=added_text),
                ))
            self._args_emitted_len += len(added_text)
        self._prev_args_json = args_json
        return out

    @staticmethod
    def _is_complete_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def _common_prefix(s1: str, s2: str) -> str:
        i = 0
        n = min(len(s1), len(s2))
        while i < n and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    @staticmethod
    def _parse_tool_call_complete_json(payload: str) -> ToolCall | None:
        if not payload:
            return None
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get('name')
        if not isinstance(name, str) or not name:
            return None
        args_obj = obj.get('arguments', obj.get('parameters', {}))
        args_json = json.dumps(args_obj, ensure_ascii=False)
        return ToolCall(function=FunctionCall(name=name, arguments=args_json))
