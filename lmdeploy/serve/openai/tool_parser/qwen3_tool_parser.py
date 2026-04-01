# Copyright (c) OpenMMLab. All rights reserved.
import json
import re
from collections.abc import Sequence

import partial_json_parser
import shortuuid
from partial_json_parser.core.options import Allow

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
from .utils import find_common_prefix, is_complete_json

logger = get_logger('lmdeploy')


@ToolParserManager.register_module(['qwen', 'qwen3'])
class Qwen3ToolParser(ToolParser):
    """Parser for Qwen3 model's tool call format.

    Handles the extraction of tool calls from Qwen3's output format, which uses XML-like tags for tool calls and
    reasoning.
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'
        self.tool_call_pattern = re.compile(r'\n*<tool_call>(.*?)</tool_call>', re.DOTALL)
        self.parse_cursor = 0
        self.qwen_tool_serial_index = -1
        self.qwen_active_tool_call_id = ''
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        # True when we are between <tool_call> and </tool_call> in the accumulated output.
        self.in_tool_block: bool = False

    def get_argments(self, obj):
        """Extract arguments from tool call object, handling different formats.

        Supports both 'parameters' and 'arguments' keys in the tool call object.
        """
        if 'parameters' in obj:
            return obj.get('parameters')
        elif 'arguments' in obj:
            return obj.get('arguments')
        return None

    def _split(self, parsing_content: str):
        """Split content into tuple: (text_content, tool_content, has_tool_end)

        This method parses the model output and separates it into regular text,
        and tool call content.
        """
        try:
            start_idx = parsing_content.index(self.tool_start_token)
            self.parse_cursor += start_idx
        except ValueError:
            # No new <tool_call> in this slice.
            self.parse_cursor += len(parsing_content)
            return parsing_content, '', False
        try:
            end_idx = parsing_content.index(self.tool_end_token)
        except ValueError:
            # Saw a start tag but not an end tag: enter tool block.
            self.in_tool_block = True
            return parsing_content[:start_idx], '', False
        # Completed a full <tool_call>...</tool_call> block in this slice.
        self.parse_cursor += (end_idx - start_idx) + len(self.tool_end_token)
        self.in_tool_block = False
        return (
            parsing_content[:start_idx],
            parsing_content[start_idx + len(self.tool_start_token):end_idx],
            True,
        )

    def detect_tool_start_tag(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        *,
        stream_buffer: StreamBuffer,
        request: ChatCompletionRequest,
    ) -> int | None:
        """Return index in delta_text where <tool_call> starts, if present.

        This is used by ResponseParser to split the chunk into reasoning vs tool-call portions without hard-coding
        protocol details there.
        """
        idx = delta_text.find(self.tool_start_token)
        return idx if idx >= 0 else None

    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output."""
        current_text = stream_buffer.current_text
        split_result = self._split(current_text[self.parse_cursor:])
        text_content, tool_content, has_tool_end = split_result
        delta = DeltaMessage()

        if text_content:
            delta.content = text_content

        if tool_content:
            strip = tool_content.strip()
            if strip:
                flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
                obj: dict | None
                try:
                    obj = partial_json_parser.loads(strip, flags)
                except partial_json_parser.core.exceptions.MalformedJSON:
                    logger.debug('cannot parse into partial JSON yet')
                    obj = None

                if obj is not None and not self.current_tool_name_sent:
                    func_name = obj.get('name')
                    if func_name:
                        if not self.qwen_active_tool_call_id:
                            self.qwen_active_tool_call_id = f'chatcmpl-tool-{shortuuid.random()}'
                            self.qwen_tool_serial_index += 1
                            self.streamed_args_for_tool.append('')
                        idx = self.qwen_tool_serial_index
                        delta.tool_calls = [
                            DeltaToolCall(
                                id=self.qwen_active_tool_call_id,
                                index=idx,
                                type='function',
                                function=DeltaFunctionCall(name=func_name).model_dump(exclude_none=True),
                            )
                        ]
                        self.current_tool_name_sent = True
                        self.prev_tool_call_arr = [dict(obj)]
                elif obj is not None:
                    idx = self.qwen_tool_serial_index
                    args = self.get_argments(obj)
                    cur_arguments = args if isinstance(args, dict) else None
                    prev_arguments = (
                        self.get_argments(self.prev_tool_call_arr[0]) if self.prev_tool_call_arr else None
                    )
                    is_comp = is_complete_json(strip)
                    argument_diff = None
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        if is_comp:
                            sent = len(self.streamed_args_for_tool[idx])
                            argument_diff = cur_args_json[sent:]
                        elif prev_arguments:
                            prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                            if cur_args_json != prev_args_json:
                                prefix = find_common_prefix(prev_args_json, cur_args_json)
                                sent = len(self.streamed_args_for_tool[idx])
                                argument_diff = prefix[sent:]
                        if argument_diff is not None:
                            delta.tool_calls = [
                                DeltaToolCall(
                                    index=idx,
                                    id=self.qwen_active_tool_call_id,
                                    function=DeltaFunctionCall(
                                        arguments=argument_diff).model_dump(exclude_none=True),
                                )
                            ]
                            self.streamed_args_for_tool[idx] += argument_diff
                    self.prev_tool_call_arr = [obj]

        if has_tool_end:
            self.qwen_active_tool_call_id = ''
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []

        return delta if delta.content is not None or delta.tool_calls else None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output.

        This method processes the full model output to extract tool calls, reasoning content, and regular text content.
        Unlike the streaming version, this processes the entire output at once.
        """
        text = model_output

        buf = []
        scan_pos = 0
        tool_calls = []
        for idx, match in enumerate(self.tool_call_pattern.finditer(text)):
            buf.append(text[scan_pos:match.start()])
            scan_pos = match.end()
            action = json.loads(match.group(1))
            name, arguments = action['name'], json.dumps(action['arguments'], ensure_ascii=False)
            tool_calls.append(ToolCall(function=FunctionCall(name=name, arguments=arguments)))
        if scan_pos < len(text):
            buf.append(text[scan_pos:])
        text = ''.join(buf)

        return ExtractedToolCallInformation(
            content=text,
            tool_calls=tool_calls,
            tools_called=bool(tool_calls),
        )
