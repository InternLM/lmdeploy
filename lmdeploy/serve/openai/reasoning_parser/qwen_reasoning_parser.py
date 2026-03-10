# Copyright (c) OpenMMLab. All rights reserved.
from .reasoning_parser import ReasoningParserManager, ThinkingReasoningParser


@ReasoningParserManager.register_module(name=['qwen-qwq', 'intern-s1'])
class QwenQwQReasoningParser(ThinkingReasoningParser):
    """Reasoning parser for Qwen QwQ / Qwen3 / InternLM-S1 models.

    Uses <think>...</think> tokens. When the end tag is missing in
    non-streaming mode, the entire output is treated as normal content
    (not reasoning). Leading/trailing newlines in reasoning content are
    stripped.
    """

    start_token = '<think>'
    end_token = '</think>'
    strip_newlines = True
    on_missing_end_tag = 'content'
