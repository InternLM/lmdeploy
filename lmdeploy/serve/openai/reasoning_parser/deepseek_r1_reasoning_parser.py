# Copyright (c) OpenMMLab. All rights reserved.
from .reasoning_parser import ReasoningParserManager, ThinkingReasoningParser


@ReasoningParserManager.register_module(name='deepseek-r1')
class DeepSeekR1ReasoningParser(ThinkingReasoningParser):
    """Reasoning parser for DeepSeek R1 model.

    Uses <think>...</think> tokens. When the end tag is missing in
    non-streaming mode, the entire output is treated as reasoning content
    (DeepSeek R1 may omit the start tag).

    Ref: https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
    """

    start_token = '<think>'
    end_token = '</think>'
    strip_newlines = False
    on_missing_end_tag = 'reasoning'

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError('DeepSeek R1 reasoning parser could not locate '
                               'think start/end tokens in the tokenizer!')
