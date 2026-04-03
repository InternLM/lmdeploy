# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

from .internlm2_tool_parser import Internlm2ToolParser
from .llama3_tool_parser import Llama3JsonToolParser
from .qwen2d5_tool_parser import Qwen2d5ToolParser
from .qwen3_tool_parser import Qwen3ToolParser
from .qwen3coder_tool_parser import Qwen3CoderToolParser
from .tool_parser import ToolParser

ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.parsers.tool_parser'])

__all__ = [
    'ToolParser',
    'ToolParserManager',
    'Internlm2ToolParser',
    'Llama3JsonToolParser',
    'Qwen2d5ToolParser',
    'Qwen3ToolParser',
    'Qwen3CoderToolParser',
]
