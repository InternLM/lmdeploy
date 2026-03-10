# Copyright (c) OpenMMLab. All rights reserved.
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import shortuuid

from lmdeploy.serve.openai.protocol import (ChatCompletionRequest, DeltaFunctionCall, DeltaMessage, DeltaToolCall,
                                            ExtractedToolCallInformation, FunctionCall, ToolCall)
from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import get_streaming_state
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')


@dataclass
class ParserState(object):
    """Maintains the state of parsing during tool call extraction."""
    position: int = 0  # Current position in the text being parsed
    current_index: int = -1  # Index of the current tool call

    id: str = ''  # ID of the current tool call

    def reset_tool_call(self):
        """Called when `</tool_call>` finish tag occurred."""
        self.id = ''


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

    def _split(self, parser_state: ParserState, parsing_content: str) -> Tuple[str, str, bool]:
        """Split content into tuple: (text_content, tool_content, has_tool_end)"""
        try:
            start_idx = parsing_content.index(self.tool_start_token)
            parser_state.position += start_idx
        except ValueError:
            parser_state.position += len(parsing_content)
            return parsing_content, '', False

        try:
            end_idx = parsing_content.index(self.tool_end_token)
        except ValueError:
            return parsing_content[:start_idx], parsing_content[start_idx:], False

        rem = end_idx - start_idx
        parser_state.position += rem + len(self.tool_end_token)
        return parsing_content[:start_idx], parsing_content[start_idx:end_idx + len(self.tool_end_token)], True

    def _extract_params(self, content: str) -> Tuple[Optional[str], Dict[str, Any], bool]:
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
    ) -> Union[DeltaMessage, None]:
        state = get_streaming_state(request)
        current_text = state.current_text

        parser_state = getattr(request, '_tool_parser_state', None)
        if parser_state is None:
            parser_state = ParserState()
            setattr(request, '_tool_parser_state', parser_state)

        split_result = self._split(parser_state, current_text[parser_state.position:])
        text_content, tool_content, has_tool_end = split_result

        delta = DeltaMessage()
        if text_content:
            delta.content = text_content

        if tool_content:
            if not parser_state.id:
                parser_state.id = f'chatcmpl-tool-{shortuuid.random()}'
                parser_state.current_index += 1
                parser_state.has_emitted_name = False
                parser_state.has_emitted_json_start = False
                parser_state.json_closed = False
                parser_state.emitted_params = set()

            func_name, args_dict, is_func_closed = self._extract_params(tool_content)

            fcall_delta = DeltaFunctionCall()
            has_updates = False

            if func_name and not getattr(parser_state, 'has_emitted_name', False):
                fcall_delta.name = func_name
                parser_state.has_emitted_name = True
                has_updates = True

            json_fragments = []
            if not getattr(parser_state, 'has_emitted_json_start', False):
                if args_dict or is_func_closed:
                    json_fragments.append('{')
                    parser_state.has_emitted_json_start = True

            for k, v in args_dict.items():
                if k not in parser_state.emitted_params:
                    prefix = ', ' if len(parser_state.emitted_params) > 0 else ''
                    serialized = json.dumps(v, ensure_ascii=False)
                    json_fragments.append(f'{prefix}\"{k}\": {serialized}')
                    parser_state.emitted_params.add(k)

            if is_func_closed and not getattr(parser_state, 'json_closed', False):
                if getattr(parser_state, 'has_emitted_json_start', False):
                    json_fragments.append('}')
                    parser_state.json_closed = True

            joined_fragments = ''.join(json_fragments)
            if joined_fragments:
                fcall_delta.arguments = joined_fragments
                has_updates = True

            if has_updates:
                parsed_delta = DeltaToolCall(
                    id=parser_state.id,
                    index=parser_state.current_index,
                    function=fcall_delta,
                )
                delta.tool_calls = [parsed_delta]

        if has_tool_end:
            parser_state.reset_tool_call()
            # Prepare for the next tool call
            if hasattr(parser_state, 'has_emitted_name'):
                delattr(parser_state, 'has_emitted_name')
                delattr(parser_state, 'has_emitted_json_start')
                delattr(parser_state, 'json_closed')
                delattr(parser_state, 'emitted_params')

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
