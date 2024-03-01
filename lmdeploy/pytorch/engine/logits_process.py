# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
from transformers.generation.logits_process import LogitsWarper

from ..messages import SchedulerSequence


def _process_temperature(scores: torch.Tensor,
                         temperature: torch.Tensor,
                         inplace: bool = True):
    """process temperature."""
    if not inplace:
        scores = scores / temperature[:, None]
    else:
        scores /= temperature[:, None]
    return scores


def _process_bad_words(scores: torch.Tensor,
                       bad_words: torch.LongTensor,
                       filter_value: float = -float('inf'),
                       inplace: bool = True):
    """process bad words."""
    batch_size = scores.size(0)
    batch_idx = torch.arange(batch_size, device=scores.device)
    filtered_scores = scores[batch_idx[:, None], bad_words]
    filtered_scores[bad_words >= 0] = filter_value

    if not inplace:
        scores = scores.clone()

    scores[batch_idx[:, None], bad_words] = filtered_scores
    return scores


def _process_repetition_penalty(scores: torch.Tensor,
                                input_ids: torch.LongTensor,
                                penalty: torch.Tensor,
                                inplace: bool = True):
    """process repetition penalty."""
    score = torch.gather(scores, 1, input_ids)
    score = torch.where(score < 0, score * penalty[:, None],
                        score / penalty[:, None])
    if not inplace:
        scores = scores.clone()
    scores.scatter_(1, input_ids, score)
    return scores


def _filter_topk_sorted(scores: torch.Tensor,
                        topk: torch.LongTensor,
                        filter_value: float = -float('inf'),
                        inplace: bool = True):
    """filter topk on sorted scores."""
    filter_value = -float('inf')
    num_tokens = scores.size(1)
    token_idx = torch.arange(num_tokens, device=scores.device)
    mask = token_idx[None, :] >= topk[:, None]
    if inplace:
        scores.masked_fill_(mask, filter_value)
    else:
        scores = scores.masked_fill(mask, filter_value)
    return scores


def _filter_topp_sorted(scores: torch.Tensor,
                        topp: torch.Tensor,
                        filter_value: float = -float('inf'),
                        inplace: bool = True):
    """filter topp on sorted scores."""
    softmax_scores = scores.softmax(-1)
    cum_scores = softmax_scores.cumsum(1) - softmax_scores
    mask = cum_scores > topp[:, None]
    mask[:, 0] = False  # keep at least one
    if inplace:
        scores.masked_fill_(mask, filter_value)
    else:
        scores = scores.masked_fill(mask, filter_value)
    return scores


def _multinomial_sampling(scores: torch.Tensor,
                          seeds: torch.LongTensor,
                          offsets: torch.LongTensor,
                          indices: torch.LongTensor = None):
    """sampling."""
    from lmdeploy.pytorch.kernels import multinomial_sampling
    return multinomial_sampling(scores, seeds, offsets, indices)


@dataclass
class SamplingInputs:
    temperature: torch.Tensor = None
    bad_words: torch.LongTensor = None
    repetition_penalty: torch.Tensor = None
    top_k: torch.LongTensor = None
    top_p: torch.Tensor = None
    random_seeds: int = None
    random_offsets: int = None
    max_top_k: int = 1
    min_top_p: float = 1.0

    @classmethod
    def from_sampling_params(cls, seqs: List[SchedulerSequence]):
        """from samplingg params."""
        batch_size = len(seqs)
        temperature = [None] * batch_size
        repetition_penalty = [None] * batch_size
        top_k = [None] * batch_size
        top_p = [None] * batch_size
        bad_words = [None] * batch_size
        random_seeds = [torch.seed() & 0xffffffff] * batch_size
        random_offsets = [None] * batch_size

        def __gather_params():
            """gather params."""
            for idx, seq in enumerate(seqs):
                param = seq.sampling_param
                temperature[idx] = param.temperature
                repetition_penalty[idx] = param.repetition_penalty
                top_k[idx] = param.top_k
                top_p[idx] = param.top_p
                random_offsets[idx] = seq.random_offsets
                if param.random_seed is not None:
                    random_seeds[idx] = param.random_seed & 0xffffffff

                bw = param.bad_words
                if (not param.ignore_eos
                        and seq.num_new_tokens < param.min_new_tokens):
                    bw = bw + param.stop_words
                bad_words[idx] = bw

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

        if all(rp == 1.0 for rp in repetition_penalty):
            repetition_penalty = None
        else:
            repetition_penalty = torch.tensor(repetition_penalty)

        temperature = torch.tensor(temperature)

        max_bw_len = max(len(bw) for bw in bad_words)
        if max_bw_len == 0:
            bad_words = None
        else:
            if all(len(bw) == max_bw_len for bw in bad_words):
                bad_words = torch.tensor(bad_words)
            else:
                bad_words = __get_bad_words(bad_words, max_bw_len)

        max_top_k = max(top_k)
        if max_top_k == 1:
            top_k = None
            top_p, min_top_p = None, 1.0
            random_seeds = None
            random_offsets = None
        else:
            top_k = torch.tensor(top_k)
            top_p, min_top_p = __get_topp(top_p)
            random_seeds = torch.tensor(random_seeds)
            random_offsets = torch.tensor(random_offsets)

        sampling_input = cls(
            temperature=temperature,
            bad_words=bad_words,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            random_seeds=random_seeds,
            random_offsets=random_offsets,
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

    def __init__(self, sampling_inputs: SamplingInputs):
        self.sampling_inputs: SamplingInputs = sampling_inputs

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
        sampling_inputs = self.sampling_inputs
        scores = scores.clone()

        repetition_penalty = sampling_inputs.repetition_penalty
        if repetition_penalty is not None:
            scores = _process_repetition_penalty(scores, input_ids,
                                                 repetition_penalty)

        temperature = sampling_inputs.temperature
        if temperature is not None:
            scores = _process_temperature(scores, temperature)

        bad_words = sampling_inputs.bad_words
        if bad_words is not None:
            scores = _process_bad_words(scores, bad_words)

        return scores

    def sampling(self, logits: torch.Tensor):
        """sampling."""
        sampling_inputs = self.sampling_inputs

        def __random_sampling(scores: torch.Tensor, indices: torch.LongTensor):
            """random sampling."""
            top_k = sampling_inputs.top_k
            if top_k is not None:
                scores = _filter_topk_sorted(scores, top_k)

            top_p = sampling_inputs.top_p
            if top_p is not None:
                scores = _filter_topp_sorted(scores, top_p)

            softmax_scores = scores.softmax(1)
            seeds = sampling_inputs.random_seeds
            offsets = sampling_inputs.random_offsets
            return _multinomial_sampling(softmax_scores, seeds, offsets,
                                         indices)

        if sampling_inputs.max_top_k == 1:
            return logits.argmax(-1)
        else:
            scores, indices = logits.sort(1, descending=True)
            return __random_sampling(scores, indices)
