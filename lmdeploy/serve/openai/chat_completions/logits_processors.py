# Copyright (c) OpenMMLab. All rights reserved.
"""Logits processors for the ``/v1/chat/completions`` endpoint."""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from lmdeploy.messages import LogitsProcessor


# modified from https://github.com/vllm-project/vllm/blob/v0.5.4/vllm/entrypoints/openai/logits_processors.py#L51  # noqa
def logit_bias_logits_processor(
        logit_bias: dict[int, float] | dict[str, float],
        tokenizer: PreTrainedTokenizerBase) -> LogitsProcessor:
    try:
        # Convert token_id to integer
        # Clamp the bias between -100 and 100 per OpenAI API spec
        clamped_logit_bias: dict[int, float] = {
            int(token_id): min(100.0, max(-100.0, bias))
            for token_id, bias in logit_bias.items()
        }
    except ValueError as exc:
        raise ValueError(
            'Found token_id in logit_bias that is not '
            'an integer or string representing an integer') from exc

    # Check if token_id is within the vocab size
    for token_id, bias in clamped_logit_bias.items():
        if token_id < 0 or token_id >= tokenizer.vocab_size:
            raise ValueError(f'token_id {token_id} in logit_bias contains '
                             'out-of-vocab token id')

    def _logit_bias_processor(
        logit_bias,
        token_ids,
        logits,
    ):
        for token_id, bias in logit_bias.items():
            logits[token_id] = logits[token_id] + bias
        return logits

    return partial(_logit_bias_processor, clamped_logit_bias)
