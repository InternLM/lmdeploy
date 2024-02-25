# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
from transformers.generation.logits_process import LogitsWarper

from ..messages import SamplingParam


@dataclass
class SamplingInputs:
    temperature: torch.Tensor = None
    bad_words: torch.LongTensor = None
    repetition_penalty: torch.Tensor = None
    top_k: torch.LongTensor = None
    top_p: torch.Tensor = None
    random_seed: int = None
    max_top_k: int = 1
    min_top_p: float = 1.0

    @classmethod
    def from_sampling_params(cls, sampling_params: List[SamplingParam]):
        """from samplingg params."""
        batch_size = len(sampling_params)
        temperature = [None] * batch_size
        repetition_penalty = [None] * batch_size
        top_k = [None] * batch_size
        top_p = [None] * batch_size
        bad_words = [None] * batch_size
        random_seed = [torch.seed()] * batch_size

        def __gather_params():
            """gather params."""
            for idx, param in enumerate(sampling_params):
                temperature[idx] = param.temperature
                bad_words[idx] = param.bad_words
                repetition_penalty[idx] = param.repetition_penalty
                top_k[idx] = param.top_k
                top_p[idx] = param.top_p
                if param.random_seed is not None:
                    random_seed[idx] = param.random_seed

        def __get_topp(top_p):
            """get topp."""
            min_top_p = min(top_p)
            if min_top_p == 1.0:
                top_p = None
            else:
                top_p = torch.tensor(top_p)
            return top_p, min_top_p

        def __get_bad_words(bad_words, max_bw_len):
            """get bad words."""
            ret = torch.full((batch_size, max_bw_len), -1, dtype=torch.int64)
            for idx, bw in enumerate(bad_words):
                bw_len = len(bw)
                if bw_len == 0:
                    continue
                bw = ret.new_tensor(bw)
                ret[idx, :bw_len] = bw
            return ret

        __gather_params()

        temperature = torch.tensor(temperature)

        repetition_penalty = torch.tensor(repetition_penalty)
        if (repetition_penalty == 1.0).all():
            repetition_penalty = None

        top_k = torch.tensor(top_k)
        max_top_k = top_k.max().item()
        if max_top_k == 1:
            top_p, min_top_p = None, 1.0
        else:
            top_p, min_top_p = __get_topp(top_p)

        max_bw_len = max(len(bw) for bw in bad_words)
        if max_bw_len == 0:
            bad_words = None
        else:
            bad_words = __get_bad_words(bad_words, max_bw_len)

        random_seed = torch.tensor(random_seed)

        sampling_input = cls(
            temperature=temperature,
            bad_words=bad_words,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            random_seed=random_seed,
            max_top_k=max_top_k,
            min_top_p=min_top_p,
        )
        return sampling_input

    def to_device(self, device: str):
        """to device."""
        input_dict = asdict(self)
        out_dict = dict()
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            out_dict[k] = v

        return SamplingInputs(**out_dict)


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
