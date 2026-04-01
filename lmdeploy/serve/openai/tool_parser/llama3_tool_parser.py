# Copyright (c) OpenMMLab. All rights reserved.
import re

from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    ToolCall,
)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')


@ToolParserManager.register_module('llama3')
class Llama3JsonToolParser(ToolParser):
    """Tool call parser for Llama 3.1 models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --tool-call-parser llama3 are all set
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool: list[str] = []
        self.prev_tool_call_arr: list[dict] = []

        self.bot_token = '<|python_tag|>'
        self.bot_token_id = tokenizer.encode(self.bot_token, add_special_tokens=False)[0]
        self.tool_call_regex = re.compile(r'\[{.*?}\]', re.DOTALL)

    def get_tool_open_tag(self) -> str | None:
        return self.bot_token

    def get_tool_close_tag(self) -> str | None:
        return None

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Llama3 tool payload is JSON; reuse shared JSON incremental
        decoder."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
