# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/qwen3_reasoning_parser.py
from typing import TYPE_CHECKING

from lmdeploy.serve.openai.protocol import DeltaMessage
from lmdeploy.serve.openai.response_parser import StreamBuffer

from .reasoning_parser import ReasoningParserManager, ThinkingReasoningParser

if TYPE_CHECKING:
    pass

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

    def extract_reasoning_streaming(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: object,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ) -> DeltaMessage | None:
        previous_token_ids = stream_buffer.previous_token_ids
        # Strip <think> from delta if present (old template / edge case where the model generates <think> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            # End token in this delta: split reasoning from content.
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning_content=reasoning if reasoning else None,
                    content=content if content else None,
                )
            # end_token_id in IDs but not in text (already stripped)
            return None

        # No end token in this delta.
        if not delta_text:
            # Nothing left after stripping start token.
            return None
        elif self.end_token_id in previous_token_ids:
            # End token already passed: everything is content now.
            return DeltaMessage(content=delta_text)
        else:
            # No end token yet: still in reasoning phase.
            return DeltaMessage(reasoning_content=delta_text)
