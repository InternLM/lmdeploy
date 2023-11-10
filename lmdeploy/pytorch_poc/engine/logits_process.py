# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers.generation.logits_process import LogitsWarper

from lmdeploy.pytorch_poc.messages import SamplingParam


class FusedLogitsProcessor(LogitsWarper):
    """Custom logits processor."""

    def __init__(self, sampling_param: SamplingParam):
        self.sampling_param = sampling_param

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (torch.LongTensor):
                Indices of input sequence tokens in the vocabulary.
            scores (torch.FloatTensor):
                Prediction scores of a language modeling head.
                These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token
                when using beam search

        Return:
            torch.FloatTensor: The processed prediction scores.

        """
        new_scores = scores
        filter_value = -float('inf')

        # top_k
        top_k_indices = None
        top_k = self.sampling_param.top_k
        if top_k is not None and top_k > 0:
            top_k = min(top_k, scores.size(-1))
            new_scores, top_k_indices = torch.topk(scores, top_k)

        # top_p
        top_p = self.sampling_param.top_p
        if top_p is not None and top_p >= 0 and top_p <= 1:
            sorted_logits, sorted_indices = torch.sort(new_scores,
                                                       descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[..., -1:] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            new_scores = new_scores.masked_fill(indices_to_remove,
                                                filter_value)

        # temperature
        temperature = self.sampling_param.temperature
        if temperature is not None:
            new_scores /= temperature

        # recovery top_k
        if top_k_indices is not None:
            output = torch.full_like(scores, filter_value)
            new_scores = torch.scatter(output, 1, top_k_indices, new_scores)

        return new_scores
