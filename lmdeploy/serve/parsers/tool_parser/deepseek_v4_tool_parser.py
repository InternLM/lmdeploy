# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from lmdeploy.deepseek_v4_encoding import dsml_token, tool_calls_block_name

from .deepseek_v32_tool_parser import DeepSeekV32ToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module(['deepseek-v4'])
class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """Tool parser for DeepSeek-V4 DSML tool-call blocks."""

    structural_tag_model = 'deepseek_v4'

    dsml_token = dsml_token
    tool_calls_block_name = tool_calls_block_name
    tool_close_prefixes = (
        f'</{dsml_token}{tool_calls_block_name}',
        f'</{dsml_token}tool_c',
        f'</{dsml_token}tool',
        f'</{dsml_token}',
        '</',
    )
