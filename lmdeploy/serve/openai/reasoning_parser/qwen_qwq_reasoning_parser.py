# Copyright (c) OpenMMLab. All rights reserved.
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from .reasoning_parser import ReasoningParserManager


@ReasoningParserManager.register_module(name='qwen-qwq')
class QwenQwQReasoningParser(DeepSeekR1ReasoningParser):
    """Reasoning parser for Qwen QwQ model.

    The Qwen QwQ model uses <think>...</think> tokens to denote reasoning text. This parser extracts the reasoning
    content from the model output.
    """
