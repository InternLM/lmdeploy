# Copyright (c) OpenMMLab. All rights reserved.
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(['kimi_k2', 'kimi-k2'])
class KimiK2ReasoningParser(ReasoningParser):
    """Reasoning parser for Kimi K2 thinking models.

    Kimi K2 Thinking always starts in reasoning mode. Kimi K2.5 and K2.6
    also default to thinking, but their chat templates support disabling it
    per request with ``thinking=False``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thinking = kwargs.get('thinking', None)

    def starts_in_reasoning_mode(self) -> bool:
        return True if self.thinking is None else bool(self.thinking)
