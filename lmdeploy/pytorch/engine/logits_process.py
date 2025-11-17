# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

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


def _process_repetition_penalty_(scores: torch.Tensor, input_ids: torch.LongTensor, penalty: torch.Tensor):
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


def _multinomial_sampling(scores: torch.Tensor,
                          seeds: torch.LongTensor,
                          offsets: torch.LongTensor,
                          indices: torch.LongTensor = None):
    """sampling."""
    from lmdeploy.pytorch.nn.multinomial_sampling import multinomial_sampling
    return multinomial_sampling(scores, seeds, offsets, indices)


SeqList = List[SchedulerSequence]


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
    max_num_logprobs: Optional[int] = None
    all_ids: Optional[torch.Tensor] = None
    num_ignore_eos: torch.Tensor = None
    batch_size: int = 0
    session_ctx: Optional[List[Dict[str, Any]]] = None
    session_to_cleanup: Optional[List[int]] = None

    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device, non_blocking=non_blocking)
            out_dict[k] = v

        return SamplingInputs(**out_dict)


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
        logprobs_mode: Optional[str] = None,
        guided_decoding_manager: Optional[GuidedDecodingManager] = None,
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
            scores = _process_repetition_penalty_(scores, all_ids, repetition_penalty)

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
