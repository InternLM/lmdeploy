# Copyright (c) OpenMMLab. All rights reserved.
from .qwen_reasoning_parser import QwenReasoningParser
from .reasoning_parser import ReasoningParserManager


@ReasoningParserManager.register_module(name='deepseek-r1')
class DeepSeekR1ReasoningParser(QwenReasoningParser):
    """Reasoning parser for DeepSeek R1 model.

    DeepSeek R1 always put <think> tag to user's prompt. see more details in
    https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
    Since DeepSeek-R1 and Qwen3-Thinking models have the same reasoning behavior,
    we remove its original implementation and directly use QwenReasoningParser.
    """
    pass
