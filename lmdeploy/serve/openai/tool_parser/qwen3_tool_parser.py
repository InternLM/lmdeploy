# Copyright (c) OpenMMLab. All rights reserved.
import re

from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

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

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode Qwen3 JSON tool payload incrementally."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)

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
