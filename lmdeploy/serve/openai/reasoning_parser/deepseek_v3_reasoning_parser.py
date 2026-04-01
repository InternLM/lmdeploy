# Copyright (c) OpenMMLab. All rights reserved.

from typing import TYPE_CHECKING

from .identity_reasoning_parser import IdentityReasoningParser
from .reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    pass

class DeepSeekV3ReasoningParser(ReasoningParser):
    """The reasoning behavior of the DeepSeek V3.1 model varies depending on
    the `enable_thinking` parameter.

    When set to True, a <think> tag is added to the user's prompt, which corresponds to the thinking mode
    of DeepSeek R1.
    When `enable_thinking` is None, the thinking mode is disabled. In this case, the parser falls back to
    the identity parser, which treats the entire model output as content and ignores any reasoning.
    """

    def __init__(self, tokenizer: object, **kwargs):
        super().__init__(tokenizer, **kwargs)

        enable_thinking = bool(kwargs.get('enable_thinking', False))
        self._parser: ReasoningParser
        if enable_thinking:
            from .qwen_reasoning_parser import QwenReasoningParser as DeepSeekR1ReasoningParser
            self._parser = DeepSeekR1ReasoningParser(tokenizer, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, **kwargs)

    def get_reasoning_open_tag(self) -> str | None:
        return self._parser.get_reasoning_open_tag()

    def get_reasoning_close_tag(self) -> str | None:
        return self._parser.get_reasoning_close_tag()

    def starts_in_reasoning_mode(self) -> bool:
        return self._parser.starts_in_reasoning_mode()
