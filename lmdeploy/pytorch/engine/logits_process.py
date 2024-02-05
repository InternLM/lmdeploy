# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
from transformers.generation.logits_process import LogitsWarper

from ..messages import SamplingParam


class SeedManager:
    """random seed manager."""

    def __init__(self):
        self._generators: Dict[int, torch.Generator] = dict()

    def new_generator(self, seed: int, device: str = 'cuda'):
        """new generator."""
        return torch.Generator(device=device).manual_seed(seed)

    def get(self, seed: int, device: str = 'cuda'):
        """get generator."""
        if seed not in self._generators:
            generator = self.new_generator(seed, device)
            self._generators[seed] = generator
        return self._generators[seed]


SEED_MANAGER = SeedManager()


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

        # temperature
        temperature = self.sampling_param.temperature
        if temperature is not None and temperature > 0:
            new_scores /= temperature

        # bad words
        bad_words = self.sampling_param.bad_words
        if bad_words:
            bad_words = list(set(bad_words))
            bad_words_bias = new_scores.new_zeros(new_scores.size(1))
            bad_words_bias[bad_words] = filter_value
            new_scores += bad_words_bias[None]

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

        # recovery top_k
        if top_k_indices is not None:
            output = torch.full_like(scores, filter_value)
            new_scores = torch.scatter(output, 1, top_k_indices, new_scores)

        return new_scores

    def sampling(self, logits: torch.Tensor):
        """sampling."""

        def __random_sampling(seed, logits: torch.Tensor):
            """random sampling."""
            generator = SEED_MANAGER.get(seed, logits.device)
            logits = logits.softmax(-1)
            return torch.multinomial(logits, 1, generator=generator)[:, 0]

        seed = self.sampling_param.random_seed
        if seed is None or self.sampling_param.top_k == 1:
            return logits.argmax(-1)
        else:
            return __random_sampling(seed, logits)
