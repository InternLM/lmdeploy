# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from lmdeploy.messages import LogitsProcessor

from ..messages import SchedulerSequence
from .guided_process import GuidedDecodingManager


def _process_temperature_(scores: torch.Tensor, temperature: torch.Tensor):
    """Process temperature."""
    temperature = temperature.to(scores.dtype)
    scores.div_(temperature[:, None])
    return scores


def _process_bad_words_(scores: torch.Tensor,
                        bad_words: torch.LongTensor,
                        mask: torch.BoolTensor,
                        filter_value: float = -float('inf')):
    """Process bad words."""
    filtered_scores = scores.gather(1, bad_words)
    filtered_scores[mask] = filter_value
    scores.scatter_(1, bad_words, filtered_scores)
    return scores


def _process_repetition_penalty_(scores: torch.Tensor, input_ids: torch.Tensor, penalty: torch.Tensor):
    """Process repetition penalty."""
    score = torch.gather(scores, 1, input_ids)
    penalty = penalty.to(score.dtype)
    score = torch.where(score < 0, score * penalty[:, None], score / penalty[:, None])
    scores.scatter_(1, input_ids, score)
    return scores


def _filter_topk_sorted_(scores: torch.Tensor, topk: torch.LongTensor, filter_value: float = -float('inf')):
    """Filter topk on sorted scores."""
    filter_value = -float('inf')
    num_tokens = scores.size(1)
    token_idx = torch.arange(num_tokens, device=scores.device)
    mask = token_idx[None, :] >= topk[:, None]
    scores.masked_fill_(mask, filter_value)
    return scores


def _filter_topp_sorted_(scores: torch.Tensor, topp: torch.Tensor, filter_value: float = -float('inf')):
    """Filter topp on sorted scores."""
    softmax_scores = scores.softmax(-1)
    cum_scores = softmax_scores.cumsum(1) - softmax_scores
    mask = cum_scores > topp[:, None]
    mask[:, 0] = False  # keep at least one
    scores.masked_fill_(mask, filter_value)
    return scores


def _filter_minp_sorted_(scores: torch.Tensor, minp: torch.Tensor, filter_value: float = -float('inf')):
    """Filter minp on sorted scores."""
    softmax_scores = scores.softmax(-1)
    top_probs, _ = softmax_scores.max(dim=-1, keepdim=True)
    scaled_min_p = minp.unsqueeze(dim=1) * top_probs
    mask = softmax_scores < scaled_min_p
    scores.masked_fill_(mask, filter_value)
    return scores


@lru_cache(maxsize=1)
def _ngram_one(dtype: torch.dtype, device: torch.device):
    return torch.ones(1, dtype=dtype, device=device)


def ngram(token_ids: torch.Tensor, n: torch.Tensor, threshold: torch.Tensor, max_n: int, same_n: bool = False):
    """Compute n-gram matches between sliding windows and a target sequence.

    For each batch, performs cosine similarity checking between:
      - All sliding windows of length `max_n` from the full sequence
      - The last `max_n` tokens of the sequence (target window)

    A match is counted when both:
      1. Cosine similarity ≈ 1 (normalized vectors match)
      2. Vector lengths match (preventing zero/normalization artifacts)

    Parameters
    ----------
    token_ids : torch.Tensor
        Input token IDs of shape (batch_size, seq_len).
        Values are typically ≥0 (0 may represent padding/special tokens).
    n : torch.Tensor
        Effective n-gram length for each batch element, shape (batch_size,).
        When `same_n=False`, positions beyond `n` in the last `max_n` tokens are masked.
    threshold : torch.Tensor
        Minimum number of matching windows required for validity, shape (batch_size,).
    max_n : int
        Maximum n-gram length (window size for matching).
    same_n : bool, default False
        If True, use full `max_n`-length windows regardless of `n`.
        If False, mask positions where index < (max_n - n) in the target window.

    Returns
    -------
    matched_mask : torch.Tensor
        Boolean mask of shape (batch_size, seq_len - max_n + 1) indicating
        which sliding windows match the target n-gram.
    found : torch.Tensor
        Boolean tensor of shape (batch_size,) indicating whether each batch
        element has at least `threshold` matches.
    """

    batch_size, seq_len = token_ids.size()
    if seq_len < max_n:
        # Not enough tokens to form a single n-gram
        matched_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=token_ids.device)
        found = torch.zeros((batch_size, ), dtype=torch.bool, device=token_ids.device)
        return matched_mask, found
    # token_ids could be 0, so we add 1 to avoid div 0
    token_ids = token_ids.to(torch.float32) + 1

    # normalize ids
    norm = token_ids[:, -max_n:]
    if not same_n:
        # fill 0 for n < max_n
        mask = torch.arange(max_n, device=token_ids.device).unsqueeze(0) >= (max_n - n.unsqueeze(1))
        norm = norm * mask.to(torch.float32)
    norm = norm.norm(2, dim=-1, keepdim=True)
    normed_ids = token_ids / norm

    # concate p1 and p2 so we can check distance and vector in one conv1d
    normed_n_ids = normed_ids[:, -max_n:]
    normed_ids_p2 = normed_ids * normed_ids
    ones_ids = torch.ones_like(normed_n_ids)
    if not same_n:
        # fill 0 for n < max_n
        normed_n_ids = normed_n_ids * mask.to(torch.float32)
        ones_ids = ones_ids * mask.to(torch.float32)
    normed_ids = torch.cat([normed_ids, normed_ids_p2], dim=0)
    normed_n_ids = torch.cat([normed_n_ids, ones_ids], dim=0)

    # check cos distance & check vector length
    match_norm = torch.conv1d(normed_ids.unsqueeze(0), normed_n_ids.unsqueeze(1), groups=batch_size * 2)[0]
    match_norm, match_ones = match_norm.chunk(2, dim=0)

    # both match result should be close to 1
    one_tensor = _ngram_one(dtype=match_norm.dtype, device=match_norm.device)
    matched_mask = match_norm.isclose(one_tensor) & match_ones.isclose(one_tensor)

    # threshold
    count = matched_mask.sum(-1)
    found = (count >= threshold) & (threshold > 0)

    return matched_mask, found


def _filter_ngram_(
    scores: torch.Tensor,
    stop_words: torch.Tensor,
    generated_ids: torch.Tensor,
    n: torch.Tensor,
    threshold: torch.Tensor,
    max_n: int,
    same_n: bool = False,
):
    """Filter ngram."""
    if stop_words is None or stop_words.numel() == 0:
        return scores
    # use first stop words
    _, found = ngram(generated_ids, n, threshold, max_n, same_n)
    stop_words = stop_words[:, 0]
    # fill all scores -inf
    scores.masked_fill_(found[:, None], -float('inf'))
    # set stop words to 0
    stop_scores = scores.gather(1, stop_words[:, None])
    stop_scores.masked_fill_(found[:, None], 0)
    scores.scatter_(1, stop_words[:, None], stop_scores)
    return scores


def _multinomial_sampling(scores: torch.Tensor,
                          seeds: torch.LongTensor,
                          offsets: torch.LongTensor,
                          indices: torch.LongTensor = None):
    """sampling."""
    from lmdeploy.pytorch.nn.multinomial_sampling import multinomial_sampling
    return multinomial_sampling(scores, seeds, offsets, indices)


SeqList = List[SchedulerSequence]


@dataclass
class SamplingInputsDelta:
    num_ignore_eos: torch.Tensor = None
    random_offsets: torch.Tensor = None
    all_ids: None | torch.Tensor = None


@dataclass
class SamplingInputs:
    temperature: torch.Tensor = None
    bad_words: torch.LongTensor = None
    bad_mask: torch.BoolTensor = None
    stop_words: torch.LongTensor = None
    stop_mask: torch.BoolTensor = None
    repetition_penalty: torch.Tensor = None
    top_k: torch.LongTensor = None
    top_p: torch.Tensor = None
    min_p: torch.Tensor = None
    random_seeds: torch.Tensor = None
    random_offsets: torch.Tensor = None
    max_top_k: int = 1
    min_top_p: float = 1.0
    response_formats: Tuple[str] = ()
    logits_processors: List[List[LogitsProcessor]] = None
    max_num_logprobs: None | int = None
    all_ids: None | torch.Tensor = None
    num_ignore_eos: torch.Tensor = None
    batch_size: int = 0
    session_ctx: None | List[Dict[str, Any]] = None
    session_to_cleanup: None | List[int] = None
    # for repetition_penalty and ngram
    generated_ids: torch.Tensor | None = None
    generated_ids_cpu: np.ndarray | None = None

    # n gram
    ngram_size: torch.Tensor = None
    ngram_threshold: torch.Tensor = None
    max_ngram_size: int = 0
    ngram_same_n: bool = False

    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        out_dict = dict()
        if self.generated_ids_cpu is not None:
            self.generated_ids = torch.from_numpy(self.generated_ids_cpu.copy())
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device, non_blocking=non_blocking)
            out_dict[k] = v

        return SamplingInputs(**out_dict)

    def get_delta(self) -> SamplingInputsDelta:
        """Get delta."""
        delta = SamplingInputsDelta()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                setattr(delta, k, v)
        return delta

    def update_delta(self, delta: SamplingInputsDelta):
        """Update from delta."""
        for f in fields(delta):
            k = f.name
            v = getattr(delta, k)
            if v is not None:
                setattr(self, k, v)


def _apply_custom_logits_processors(batched_logits_processors, all_ids, logits):
    """Apply custom logits processors."""
    for seq_id, processors in enumerate(batched_logits_processors):
        if processors is not None:
            for processor in processors:
                logits[seq_id] = processor(all_ids[seq_id], logits[seq_id])
    return logits


def _torch_topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    if k == 1:
        # torch.topk would not fallback to torch.max/torch.min automatically
        if largest:
            return torch.max(x, dim=dim, keepdim=True)
        else:
            return torch.min(x, dim=dim, keepdim=True)
    else:
        return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)


class FusedLogitsProcessor:
    """Custom logits processor."""

    def __init__(
        self,
        sampling_inputs: SamplingInputs,
        logprobs_mode: None | str = None,
        guided_decoding_manager: None | GuidedDecodingManager = None,
    ):
        self.sampling_inputs: SamplingInputs = sampling_inputs
        self.logprobs_mode = logprobs_mode
        self.guided_decoding_manager = guided_decoding_manager
        if sampling_inputs.session_to_cleanup:
            self.cleanup_sessions(sampling_inputs.session_to_cleanup)

        if self.guided_decoding_manager:
            self.guided_processors = self.guided_decoding_manager.get_processors(sampling_inputs.session_ctx,
                                                                                 sampling_inputs.response_formats)
        else:
            self.guided_processors = {}

    async def _wait_stream_once(self):
        """Wait stream once."""
        stream = torch.cuda.current_stream()
        if not stream.query():
            await asyncio.sleep(0)

    async def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            scores (torch.FloatTensor):
                Prediction scores of a language modeling head.
                These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token
                when using beam search


        Return:
            torch.FloatTensor: The processed prediction scores.

        """

        num_logprobs = self.sampling_inputs.max_num_logprobs
        # get raw logprobs
        if num_logprobs < 0:
            logprobs = None
        else:
            if self.logprobs_mode == 'raw_logits':
                logprobs = scores.clone()
            elif self.logprobs_mode == 'raw_logprobs':
                logprobs = scores.log_softmax(dim=-1)
            else:
                logprobs = None

        sampling_inputs = self.sampling_inputs
        all_ids = sampling_inputs.all_ids
        custom_logits_processors = self.sampling_inputs.logits_processors
        if self.guided_decoding_manager and self.guided_processors:
            if not hasattr(self, 'guided_bitmask'):
                self.guided_bitmask = self.guided_decoding_manager.allocate_batched_bitmap(len(scores))

            assert self.guided_bitmask is not None
            guided_bitmask = self.guided_bitmask

            await self._wait_stream_once()
            for i, processor in self.guided_processors.items():
                self.guided_decoding_manager.fill_bitmap(processor, guided_bitmask, i)

            self.guided_decoding_manager.apply_batched_bitmap(scores, guided_bitmask)

        if any(custom_logits_processors):
            await self._wait_stream_once()
            scores = _apply_custom_logits_processors(custom_logits_processors, all_ids, scores)

        repetition_penalty = sampling_inputs.repetition_penalty
        if repetition_penalty is not None:
            generated_ids = sampling_inputs.generated_ids
            scores = _process_repetition_penalty_(scores, generated_ids, repetition_penalty)

        if sampling_inputs.max_ngram_size > 0:
            generated_ids = sampling_inputs.generated_ids
            scores = _filter_ngram_(
                scores,
                sampling_inputs.stop_words,
                generated_ids,
                sampling_inputs.ngram_size,
                sampling_inputs.ngram_threshold,
                sampling_inputs.max_ngram_size,
                sampling_inputs.ngram_same_n,
            )

        temperature = sampling_inputs.temperature
        if temperature is not None:
            scores = _process_temperature_(scores, temperature)

        bad_words = sampling_inputs.bad_words
        if bad_words is not None:
            bad_mask = sampling_inputs.bad_mask
            scores = _process_bad_words_(scores, bad_words, bad_mask)

        stop_words = sampling_inputs.stop_words
        if stop_words is not None:
            ignore_eos = sampling_inputs.num_ignore_eos > 0
            stop_mask = sampling_inputs.stop_mask
            stop_mask = torch.where(ignore_eos[:, None], stop_mask, False)
            scores = _process_bad_words_(scores, stop_words, stop_mask)

        return scores, logprobs

    @torch.inference_mode()
    def sampling(self, logits: torch.Tensor):
        """sampling."""
        sampling_inputs = self.sampling_inputs

        def __random_sampling(scores: torch.Tensor, indices: torch.LongTensor):
            """Random sampling."""
            max_topk = sampling_inputs.max_top_k
            top_k = sampling_inputs.top_k
            if max_topk <= 0:
                max_topk = scores.size(1)
                if top_k is not None:
                    top_k = torch.masked_fill(top_k, top_k <= 0, max_topk)

            if top_k is not None:
                scores = _filter_topk_sorted_(scores, top_k)

            top_p = sampling_inputs.top_p
            if top_p is not None:
                scores = _filter_topp_sorted_(scores, top_p)

            min_p = sampling_inputs.min_p
            if min_p is not None:
                scores = _filter_minp_sorted_(scores, min_p)

            softmax_scores = scores.softmax(1)

            seeds = sampling_inputs.random_seeds
            offsets = sampling_inputs.random_offsets
            return _multinomial_sampling(softmax_scores, seeds, offsets, indices)

        if sampling_inputs.max_top_k == 1:
            result = logits.argmax(-1)
        else:
            # sort logits is too slow. and we only need topk logits
            max_topk = sampling_inputs.max_top_k
            if max_topk <= 0:
                scores, indices = logits.sort(1, descending=True)
            else:
                scores, indices = _torch_topk(logits, max_topk, dim=1)
            result = __random_sampling(scores, indices)

        if self.guided_decoding_manager and self.guided_processors:
            for i, processor in self.guided_processors.items():
                self.guided_decoding_manager.accept_token(processor, result[i])

        return result

    @torch.inference_mode()
    def compute_logprobs(self, raw_logprobs: torch.Tensor, token_ids: torch.LongTensor):
        """Compute logprobs."""
        if raw_logprobs is None:
            return None

        indices = token_ids.unsqueeze(-1)
        logprobs = raw_logprobs.gather(-1, indices)
        num_logprobs = self.sampling_inputs.max_num_logprobs
        if num_logprobs > 0:
            topk_logprobs, topk_indices = _torch_topk(raw_logprobs, num_logprobs, dim=-1)
            logprobs = torch.cat([logprobs, topk_logprobs], dim=-1)
            indices = torch.cat([indices, topk_indices], dim=-1)

        return logprobs, indices.to(torch.int32)

    def cleanup_sessions(self, session_ids: List[int]):
        if self.guided_decoding_manager:
            for session_id in session_ids:
                self.guided_decoding_manager.remove_processor(session_id)
