# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/qwen3_reasoning_parser.py
from .reasoning_parser import ReasoningParserManager, ThinkingReasoningParser


@ReasoningParserManager.register_module(name=['qwen-qwq', 'qwen3', 'intern-s1', 'deepseeek-r1'])
class QwenReasoningParser(ThinkingReasoningParser):
    """Reasoning parser for Qwen QwQ / Qwen3 / Intern-S / Qwen3.5 models.

    Qwen3 models, such as Qwen3-8B, Qwen3-**-Instruct, generate <think> tag if enable_thinking is True.
    However, Qwen3-Thinking models and Qwen3.5 models put <think> in user's prompt, thus they don't
    generate <think> tag. Intern-S models hold the same behavior as Qwen3-Thinking models.

    This parser handles both styles: if <think> appears in the generated output
    it is stripped before extraction (non-streaming) or skipped (streaming).
    """

    start_token = '<think>'
    end_token = '</think>'
