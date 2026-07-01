# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from lmdeploy.deepseek_v4_encoding import dsml_token, parse_tool_calls, tool_calls_block_name

from .deepseek_v32_tool_parser import DeepSeekV32ToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module(['deepseek-v4'])
class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """Tool parser for DeepSeek-V4 DSML tool-call blocks."""

    dsml_token = dsml_token
    tool_calls_block_name = tool_calls_block_name
    parse_tool_calls_func = staticmethod(parse_tool_calls)
