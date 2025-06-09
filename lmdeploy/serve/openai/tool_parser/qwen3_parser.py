# Copyright (c) OpenMMLab. All rights reserved.
import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import partial_json_parser
import shortuuid
from partial_json_parser.core.exceptions import MalformedJSON, PartialJSON
from partial_json_parser.core.options import Allow

from lmdeploy.serve.openai.protocol import (ChatCompletionRequest, DeltaFunctionCall, DeltaMessage, DeltaToolCall,
                                            ExtractedToolCallInformation, FunctionCall, ToolCall)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')


@dataclass
class ParserState(object):
    """Maintains the state of parsing during tool call extraction.

    Tracks position in text, tool call index, parsing flags, and what parts of the tool call have been sent to the
    client.
    """
    position: int = 0  # Current position in the text being parsed
    current_index: int = -1  # Index of the current tool call
    parsing_reasoning: bool = False  # Whether currently parsing reasoning content

    id: str = ''  # ID of the current tool call
    parse_arg_as_object: bool = False  # Whether to parse arguments as an object
    name_sent: bool = False  # Whether the tool name has been sent
    args_sent: bool = False  # Whether the tool arguments have been sent

    def reset_tool_call(self):
        """Called when `</tool_call>` finish tag occurred."""
        self.id = ''
        self.parse_arg_as_object = False
        self.name_sent = False
        self.args_sent = False


@ToolParserManager.register_module(['qwen3'])
class Qwen3ToolParser(ToolParser):
    """Parser for Qwen3 model's tool call format.

    Handles the extraction of tool calls from Qwen3's output format, which uses XML-like tags for tool calls and
    reasoning.
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.think_start_token = '<think>'
        self.think_end_token = '</think>'
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'
        self.tool_call_pat = re.compile(r'\n*<tool_call>(.*?)</tool_call>', re.DOTALL)
        self.think_pat = re.compile(r'<think>\n*(.*?)\n*</think>', re.DOTALL)

    def get_argments(self, obj):
        """Extract arguments from tool call object, handling different formats.

        Supports both 'parameters' and 'arguments' keys in the tool call object.
        """
        if 'parameters' in obj:
            return obj.get('parameters')
        elif 'arguments' in obj:
            return obj.get('arguments')
        return None

    def _split(self, parser_state: ParserState, parsing_content: str):
        """Split content into tuple: (text_content, reasoning_content, tool_content, has_tool_end)

        This method parses the model output and separates it into regular text,
        reasoning content (inside <think> tags), and tool call content.
        """
        # reasoning content
        if parser_state.parsing_reasoning:
            if self.think_end_token in parsing_content:
                parser_state.parsing_reasoning = False
                end_pos = parsing_content.index(self.think_end_token)
                parser_state.position += end_pos + len(self.think_end_token)
                return '', parsing_content[:end_pos], '', False
            parser_state.position += len(parsing_content)
            return '', parsing_content, '', False
        if self.think_start_token in parsing_content:
            start_pos = parsing_content.index(self.think_start_token)
            if self.think_end_token not in parsing_content:
                # parse content: <think> ... (incomplete)
                parser_state.parsing_reasoning = True
                parser_state.position += len(parsing_content)
                return parsing_content[:start_pos], parsing_content[start_pos + len(self.think_start_token):], '', False
            # parse content: <think> ... </think>
            end_pos = parsing_content.index(self.think_end_token)
            parser_state.parsing_reasoning = False
            parser_state.position += end_pos + len(self.think_end_token)
            content = parsing_content[:start_pos]
            reasoning_content = parsing_content[start_pos + len(self.think_start_token):end_pos]
            return content, reasoning_content, '', False

        # tool call
        try:
            start_idx = parsing_content.index(self.tool_start_token)
        except ValueError:
            parser_state.position += len(parsing_content)
            return parsing_content, '', '', False
        try:
            end_idx = parsing_content.index(self.tool_end_token)
        except ValueError:
            parser_state.position += start_idx
            return parsing_content[:start_idx], '', parsing_content[start_idx + len(self.tool_start_token):], False
        parser_state.position += len(parsing_content)
        return parsing_content[:start_idx], '', parsing_content[start_idx + len(self.tool_start_token):end_idx], True

    def _parse_delta_tool_call(self, parser_state: ParserState, tool_content: str) -> Optional[DeltaToolCall]:
        """Parse tool content into a DeltaToolCall object.

        This method handles the incremental parsing of tool calls from partial JSON content.
        """
        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending an incomplete string since OpenAI only ever
        # allows sending the entire tool/function name at once.
        flags = Allow.ALL & ~Allow.STR & ~Allow.NUM & ~Allow.BOOL & ~Allow.NULL
        if parser_state.parse_arg_as_object:
            flags &= ~Allow.OBJ

        try:
            parsable_arr = tool_content.strip()

            # Tool calls are generated in an object format,
            # Attempt to parse the JSON content incrementally.
            # Parallel tool calls is not supported yet.
            try:
                tool_call_arr: Dict = partial_json_parser.loads(parsable_arr, flags)
            except (MalformedJSON, PartialJSON):
                logger.debug('not enough tokens to parse into JSON yet')
                return

            fcall = DeltaFunctionCall()
            # Extract and set function name if not already sent
            if not parser_state.name_sent:
                func_name = tool_call_arr.get('name')
                if func_name:
                    fcall.name = func_name
                    parser_state.name_sent = True
            # Extract and set function arguments if not already sent
            if not parser_state.args_sent:
                args = self.get_argments(tool_call_arr)
                if isinstance(args, dict) and not parser_state.parse_arg_as_object:
                    # Reparse with new flags to handle object arguments properly
                    parser_state.parse_arg_as_object = True
                    flags &= ~Allow.OBJ
                    try:
                        tool_call_arr: Dict = partial_json_parser.loads(parsable_arr, flags)
                        args = self.get_argments(tool_call_arr)
                    except (MalformedJSON, PartialJSON):
                        logger.debug('not enough tokens to parse into JSON yet')
                        args = None
                if args and isinstance(args, dict):
                    fcall.arguments = json.dumps(args, ensure_ascii=False)
                    parser_state.args_sent = True

            # Return None if no new information to send
            if not fcall.name and not fcall.arguments:
                return
            if not parser_state.id:
                # A new tool call parsed, allocate a new id & index
                parser_state.id = f'chatcmpl-tool-{shortuuid.random()}'
                parser_state.current_index += 1
            # Create and return the DeltaToolCall object
            return DeltaToolCall(
                id=parser_state.id,
                index=parser_state.current_index,
                function=fcall.model_dump(exclude_none=True),
            )
        except Exception:
            logger.exception('Error trying to handle streaming tool call.')
            logger.debug('Skipping chunk as a result of tool streaming extraction error')
            return

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """Extract tool calls from streaming model output.

        This method processes incremental model output to extract tool calls, reasoning content, and regular text
        content in a streaming fashion. It maintains parser state between calls to handle partial outputs.
        """
        parser_state = getattr(request, '_tool_parser_state', None)
        if parser_state is None:
            parser_state = ParserState()
            setattr(request, '_tool_parser_state', parser_state)

        # Split the new content into text, reasoning, and tool content
        split_result = self._split(parser_state, current_text[parser_state.position:])
        text_content, reasoning_content, tool_content, has_tool_end = split_result
        delta = DeltaMessage()

        # Add each type of content to the delta message if present
        if text_content:
            delta.content = text_content
        if reasoning_content:
            delta.reasoning_content = reasoning_content
        if tool_content:
            # Parse tool content into a DeltaToolCall object
            delta_tool_call = self._parse_delta_tool_call(parser_state, tool_content)
            if delta_tool_call is not None:
                delta.tool_calls = [delta_tool_call]
            if has_tool_end:
                parser_state.reset_tool_call()

        if delta.content or delta.reasoning_content or delta.tool_calls:
            return delta
        return

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
        reasoning_content = []

        # Extract reasoning content (content inside <think> tags)
        buf = []
        scan_pos = 0
        for match in self.think_pat.finditer(text):
            buf.append(text[scan_pos:match.start()])  # Add text before the <think> tag
            scan_pos = match.end()
            reasoning_content.append(match.group(1))  # Extract content inside <think> tags
        if scan_pos < len(text):
            buf.append(text[scan_pos:])  # Add remaining text
        text = ''.join(buf)  # Reconstruct text without <think> tags

        # Extract tool calls (content inside <tool_call> tags)
        buf = []
        scan_pos = 0
        tool_calls = []
        for idx, match in enumerate(self.tool_call_pat.finditer(text)):
            buf.append(text[scan_pos:match.start()])  # Add text before the <tool_call> tag
            scan_pos = match.end()
            action = json.loads(match.group(1))  # Parse the tool call JSON
            name, arguments = action['name'], json.dumps(action['arguments'], ensure_ascii=False)
            tool_calls.append(ToolCall(function=FunctionCall(name=name, arguments=arguments)))
        if scan_pos < len(text):
            buf.append(text[scan_pos:])  # Add remaining text
        text = ''.join(buf)  # Reconstruct text without <tool_call> tags

        return ExtractedToolCallInformation(
            content=text,
            reasoning_content=''.join(reasoning_content),
            tool_calls=tool_calls,
            tools_called=bool(tool_calls),
        )
