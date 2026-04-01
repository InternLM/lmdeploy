# Copyright (c) OpenMMLab. All rights reserved.

# modified from https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/identity_reasoning_parser.py
from typing import TYPE_CHECKING

from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    pass


class IdentityReasoningParser(ReasoningParser):
    """Identity reasoning parser.

    This parser does not attempt to parse or strip out reasoning tokens. It treats the entire model output as content
    and ignores reasoning.
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)


    def get_reasoning_open_tag(self) -> str | None:
        return None

    def get_reasoning_close_tag(self) -> str | None:
        return None

    def starts_in_reasoning_mode(self) -> bool:
        return False
