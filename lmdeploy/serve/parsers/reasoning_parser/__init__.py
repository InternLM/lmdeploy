# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

from .deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from .reasoning_parser import ReasoningParser

ReasoningParserManager = Registry('reasoning_parser', locations=['lmdeploy.serve.parsers.reasoning_parser'])

__all__ = [
    'ReasoningParser',
    'ReasoningParserManager',
    'DeepSeekV3ReasoningParser',
]
