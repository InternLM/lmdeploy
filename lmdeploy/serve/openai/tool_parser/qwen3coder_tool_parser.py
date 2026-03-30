# Copyright (c) OpenMMLab. All rights reserved.
import json
import re
from collections.abc import Sequence
from typing import Any

import shortuuid

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from lmdeploy.serve.openai.response_parser import StreamBuffer
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')


def _parse_tool_call_arguments_dict(arguments: Any) -> dict[str, Any] | None:
    """Return dict-like tool arguments for Qwen3Coder request rendering."""
    if not isinstance(arguments, str):
        return None

    try:
        parsed_arguments = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed_arguments, dict):
        return parsed_arguments
    return None


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(ToolParser):
    """Parser for Qwen3 Coder model's tool call format.

    Handles the extraction of tool calls from Qwen3 Coder's output format, which uses purely XML tags for function names
    and parameters, e.g., <tool_call> <function=func_name> <parameter=arg_name>arg_value</parameter> </function>
    </tool_call>
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'
        self.func_prefix = '<function='
        self.func_end_token = '</function>'
        self.param_prefix = '<parameter='
        self.param_end_token = '</parameter>'

        self.tool_call_pat = re.compile(r'\n*<tool_call>(.*?)</tool_call>', re.DOTALL)
        self.parse_cursor = 0
        self.qwen_tool_serial_index = -1
        self.qwen_active_tool_call_id = ''
        self.coder_has_emitted_name = False
        self.coder_has_emitted_json_start = False
        self.coder_json_closed = False
        self.coder_emitted_param_names: set[str] = set()

    def _normalize_request_messages(self, messages: list[dict]) -> list[dict] | None:
        """Return a render-safe copy of request messages when needed."""
        normalized_messages = None

        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            tool_calls = message.get('tool_calls')
            if not isinstance(tool_calls, list):
                continue

            normalized_tool_calls = None
            for tool_idx, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get('function')
                if not isinstance(function, dict) or isinstance(function.get('arguments'), dict):
                    continue

                parsed_arguments = _parse_tool_call_arguments_dict(function.get('arguments'))
                if parsed_arguments is None:
                    continue

                if normalized_messages is None:
                    normalized_messages = list(messages)
                if normalized_tool_calls is None:
                    normalized_tool_calls = list(tool_calls)
                    normalized_message = dict(message)
                    normalized_message['tool_calls'] = normalized_tool_calls
                    normalized_messages[msg_idx] = normalized_message

                normalized_function = dict(function)
                normalized_function['arguments'] = parsed_arguments

                normalized_tool_call = dict(tool_call)
                normalized_tool_call['function'] = normalized_function
                normalized_tool_calls[tool_idx] = normalized_tool_call

        return normalized_messages

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        messages = request.messages
        if not isinstance(messages, list):
            return request

        normalized_messages = self._normalize_request_messages(messages)
        if normalized_messages is None:
            return request
        return request.model_copy(update={'messages': normalized_messages})

    def _split(self, parsing_content: str) -> tuple[str, str, bool]:
        """Split content into tuple: (text_content, tool_content, has_tool_end)"""
        try:
            start_idx = parsing_content.index(self.tool_start_token)
            self.parse_cursor += start_idx
        except ValueError:
            self.parse_cursor += len(parsing_content)
            return parsing_content, '', False

        try:
            end_idx = parsing_content.index(self.tool_end_token)
        except ValueError:
            return parsing_content[:start_idx], parsing_content[start_idx:], False

        rem = end_idx - start_idx
        self.parse_cursor += rem + len(self.tool_end_token)
        return parsing_content[:start_idx], parsing_content[start_idx:end_idx + len(self.tool_end_token)], True

    def _extract_params(self, content: str) -> tuple[str | None, dict[str, Any], bool]:
        """Parse XML tool content into components."""
        content = content.replace(self.tool_start_token, '').replace(self.tool_end_token, '').strip()

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

            if param_val_str.lower() == 'null':
                val = None
            elif param_val_str.lower() == 'true':
                val = True
            elif param_val_str.lower() == 'false':
                val = False
            else:
                try:
                    val = json.loads(param_val_str)
                except json.JSONDecodeError:
                    val = param_val_str
            args_dict[param_name] = val
            search_idx = val_end + len(self.param_end_token)

        is_func_closed = self.func_end_token in content
        return func_name, args_dict, is_func_closed

    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ) -> DeltaMessage | None:
        current_text = stream_buffer.current_text

        split_result = self._split(current_text[self.parse_cursor:])
        text_content, tool_content, has_tool_end = split_result

        delta = DeltaMessage()
        if text_content:
            delta.content = text_content

        if tool_content:
            if not self.qwen_active_tool_call_id:
                self.qwen_active_tool_call_id = f'chatcmpl-tool-{shortuuid.random()}'
                self.qwen_tool_serial_index += 1
                self.coder_has_emitted_name = False
                self.coder_has_emitted_json_start = False
                self.coder_json_closed = False
                self.coder_emitted_param_names.clear()

            func_name, args_dict, is_func_closed = self._extract_params(tool_content)

            fcall_delta = DeltaFunctionCall()
            has_updates = False

            if func_name and not self.coder_has_emitted_name:
                fcall_delta.name = func_name
                self.coder_has_emitted_name = True
                has_updates = True

            json_fragments = []
            if not self.coder_has_emitted_json_start:
                if args_dict or is_func_closed:
                    json_fragments.append('{')
                    self.coder_has_emitted_json_start = True

            for k, v in args_dict.items():
                if k not in self.coder_emitted_param_names:
                    prefix = ', ' if len(self.coder_emitted_param_names) > 0 else ''
                    serialized = json.dumps(v, ensure_ascii=False)
                    json_fragments.append(f'{prefix}\"{k}\": {serialized}')
                    self.coder_emitted_param_names.add(k)

            if is_func_closed and not self.coder_json_closed:
                if self.coder_has_emitted_json_start:
                    json_fragments.append('}')
                    self.coder_json_closed = True

            joined_fragments = ''.join(json_fragments)
            if joined_fragments:
                fcall_delta.arguments = joined_fragments
                has_updates = True

            if has_updates:
                parsed_delta = DeltaToolCall(
                    id=self.qwen_active_tool_call_id,
                    index=self.qwen_tool_serial_index,
                    function=fcall_delta,
                )
                delta.tool_calls = [parsed_delta]

        if has_tool_end:
            self.qwen_active_tool_call_id = ''
            self.coder_has_emitted_name = False
            self.coder_has_emitted_json_start = False
            self.coder_json_closed = False
            self.coder_emitted_param_names.clear()

        return delta

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        buf = []
        scan_pos = 0
        tool_calls = []

        for idx, match in enumerate(self.tool_call_pat.finditer(text)):
            buf.append(text[scan_pos:match.start()])
            scan_pos = match.end()

            tool_content = match.group(1)
            func_name, args_dict, _ = self._extract_params(tool_content)

            if func_name:
                tool_calls.append(
                    ToolCall(function=FunctionCall(
                        name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False) if args_dict else '{}')))

        if scan_pos < len(text):
            buf.append(text[scan_pos:])

        text = ''.join(buf)

        return ExtractedToolCallInformation(
            content=text,
            tool_calls=tool_calls,
            tools_called=bool(tool_calls),
        )
