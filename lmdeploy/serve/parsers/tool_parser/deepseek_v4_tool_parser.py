# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from lmdeploy import deepseek_v4_encoding

from .deepseek_v32_tool_parser import DeepSeekV32ToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module('deepseek-v4')
class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """Tool parser for DeepSeek-V4 DSML tool-call blocks."""

    structural_tag_model = 'deepseek_v4'

    dsml_token = deepseek_v4_encoding.dsml_token
    tool_calls_block_name = deepseek_v4_encoding.tool_calls_block_name

    # Proper close-tag prefixes possible at decoded-token boundaries.
    tool_close_prefixes = (
        f'</{dsml_token}{tool_calls_block_name}',
        f'</{dsml_token}tool_c',
        f'</{dsml_token}tool',
        f'</{dsml_token}',
        '</',
    )
