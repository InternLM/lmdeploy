# Copyright (c) OpenMMLab. All rights reserved.
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module('deepseek-v3')
class DeepSeekV3ReasoningParser(ReasoningParser):
    """Reasoning parser for DeepSeek-V3.

    DeepSeek-V3 differs from qwen3 default behavior:
    - ``enable_thinking=True``: model can emit reasoning stream (<think>...</think>)
    - ``enable_thinking=None``: model typically emits no reasoning part
    """

    def __init__(self, tokenizer: object, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.enable_thinking = kwargs.get('enable_thinking', None)

    def starts_in_reasoning_mode(self) -> bool:
        # Enter reasoning mode only when explicitly requested.
        return self.enable_thinking is True
