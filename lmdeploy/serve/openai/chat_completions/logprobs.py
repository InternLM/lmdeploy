# Copyright (c) OpenMMLab. All rights reserved.
"""Logprobs construction helpers for the ``/v1/chat/completions`` endpoint."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from lmdeploy.serve.openai.protocol import ChatCompletionTokenLogprob, ChoiceLogprobs, TopLogprob


def _create_chat_completion_logprobs(tokenizer: PreTrainedTokenizerBase,
                                     token_ids: list[int] | None = None,
                                     logprobs: list[dict[int, float]]
                                     | None = None):
    """Create openai LogProbs for chat.completion.

    Args:
        tokenizer (PreTrainedTokenizerBase): tokenizer.
        token_ids (list[int]): output token ids.
        logprobs (list[dict[int, float]]): the top logprobs for each output
            position.
    Returns:
        ChoiceLogprobs: logprob result.
    """
    if token_ids is None or logprobs is None:
        return None

    content: list[ChatCompletionTokenLogprob] = []
    for token_id, tops in zip(token_ids, logprobs):
        item = ChatCompletionTokenLogprob(token='',
                                          bytes=[],
                                          logprob=0.0,
                                          top_logprobs=[])
        for top_id, prob in tops.items():
            token = tokenizer.convert_ids_to_tokens(top_id)
            if isinstance(token, bytes):
                _bytes = list(token)
                token = token.decode('utf-8', errors='backslashreplace')
            else:
                _bytes = list(token.encode())  # token is str
            if top_id == token_id:
                item.token = token
                item.bytes = _bytes
                item.logprob = prob
            else:
                item.top_logprobs.append(
                    TopLogprob(token=token, bytes=_bytes, logprob=prob))
        content.append(item)
    return ChoiceLogprobs(content=content)


def _create_output_token_logprobs(token_ids: list[int] | None = None,
                                  logprobs: list[dict[int, float]]
                                  | None = None):
    """Create raw (logprob, token_id) pairs for output tokens."""
    if token_ids is None or logprobs is None:
        return None

    output_token_logprobs = []
    for tok, tok_logprobs in zip(token_ids, logprobs):
        output_token_logprobs.append((tok_logprobs[tok], tok))
    return output_token_logprobs or None
