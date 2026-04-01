# Copyright (c) OpenMMLab. All rights reserved.

# modified from https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/identity_reasoning_parser.py
from typing import TYPE_CHECKING

from lmdeploy.serve.openai.protocol import DeltaMessage
from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import ReasoningParser
from lmdeploy.serve.openai.response_parser import StreamBuffer

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


class IdentityReasoningParser(ReasoningParser):
    """Identity reasoning parser.

    This parser does not attempt to parse or strip out reasoning tokens. It treats the entire model output as content
    and ignores reasoning.
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)


    def extract_reasoning_streaming(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: object,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ) -> DeltaMessage | None:
        # Just wrap delta_text as content, ignore reasoning
        if delta_text:
            return DeltaMessage(content=delta_text)
        return None

    def extract_reasoning(self, model_output: str, request: 'ChatCompletionRequest') -> tuple[str | None, str | None]:
        # No reasoning separation: return None for reasoning,
        # and full model_output as content
        return None, model_output

    def get_reasoning_open_tag(self) -> str | None:
        return None

    def get_reasoning_close_tag(self) -> str | None:
        return None

    def starts_in_reasoning_mode(self) -> bool:
        return False
