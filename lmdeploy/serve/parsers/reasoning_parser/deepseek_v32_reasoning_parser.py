# Copyright (c) OpenMMLab. All rights reserved.
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(['deepseek-v32', 'deepseek-v3.2'])
class DeepSeekV32ReasoningParser(ReasoningParser):
    """Reasoning parser for DeepSeek-V3.2 thinking mode."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thinking = kwargs.get('thinking', None)
        self.enable_thinking = kwargs.get('enable_thinking', None)

    def starts_in_reasoning_mode(self) -> bool:
        return self.thinking is True or self.enable_thinking is True
