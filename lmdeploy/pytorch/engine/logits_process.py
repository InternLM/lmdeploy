# Copyright (c) OpenMMLab. All rights reserved.
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers.generation.logits_process import LogitsWarper

from lmdeploy.messages import LogitsProcessor
from lmdeploy.tokenizer import Tokenizer

from ..messages import SchedulerSequence


def _process_temperature_(scores: torch.Tensor, temperature: torch.Tensor):
    """process temperature."""
    temperature = temperature.to(scores.dtype)
    scores.div_(temperature[:, None])
    return scores


def _process_bad_words_(scores: torch.Tensor,
                        bad_words: torch.LongTensor,
                        filter_value: float = -float('inf')):
    """process bad words."""
    mask = bad_words >= 0
    bad_words = bad_words.where(mask, 0)
    filtered_scores = scores.gather(1, bad_words)
    filtered_scores[mask] = filter_value
    scores.scatter_(1, bad_words, filtered_scores)
    return scores


def _process_repetition_penalty_(scores: torch.Tensor,
                                 input_ids: torch.LongTensor,
                                 penalty: torch.Tensor):
    """process repetition penalty."""
    score = torch.gather(scores, 1, input_ids)
    penalty = penalty.to(score.dtype)
    score = torch.where(score < 0, score * penalty[:, None],
                        score / penalty[:, None])
    scores.scatter_(1, input_ids, score)
    return scores


def _filter_topk_sorted_(scores: torch.Tensor,
                         topk: torch.LongTensor,
                         filter_value: float = -float('inf')):
    """filter topk on sorted scores."""
    filter_value = -float('inf')
    num_tokens = scores.size(1)
    token_idx = torch.arange(num_tokens, device=scores.device)
    mask = token_idx[None, :] >= topk[:, None]
    scores.masked_fill_(mask, filter_value)
    return scores


def _filter_topp_sorted_(scores: torch.Tensor,
                         topp: torch.Tensor,
                         filter_value: float = -float('inf')):
    """filter topp on sorted scores."""
    softmax_scores = scores.softmax(-1)
    cum_scores = softmax_scores.cumsum(1) - softmax_scores
    mask = cum_scores > topp[:, None]
    mask[:, 0] = False  # keep at least one
    scores.masked_fill_(mask, filter_value)
    return scores


def _filter_minp_sorted_(scores: torch.Tensor,
                         minp: torch.Tensor,
                         filter_value: float = -float('inf')):
    """filter minp on sorted scores."""
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


def _guided_sampling(response_formats: Tuple[Dict], scores: torch.Tensor,
                     guided_input_ids: Optional[torch.Tensor],
                     tokenizer: object):
    if guided_input_ids is None:
        return scores
    for i in range(len(response_formats)):
        _format = response_formats[i]
        if isinstance(_format, Dict) and _format.get('type', 'text') != 'text':
            if _format['type'] == 'json_schema':
                schema = _format['json_schema']
                if isinstance(schema, Dict):
                    for key in ['json_schema', 'schema']:
                        if key in schema:
                            schema = json.dumps(schema[key])
                elif schema is None:
                    from .guided_process import JSON_GRAMMAR
                    schema = JSON_GRAMMAR
                elif isinstance(schema, str):
                    raise ValueError(
                        f'Cannot parse schema {schema}. The schema must be '
                        'either a dictionary or a string that contains the'
                        ' JSON Schema specification')
            elif _format['type'] == 'regex_schema':
                schema = _format.get('regex_schema', '')
            else:
                raise ValueError(f"unsupported format type: {_format['type']}")
            from .guided_process import _get_guided_logits_processor
            processor = _get_guided_logits_processor(schema, tokenizer,
                                                     _format['type'])
            if processor:
                scores[i] = processor(guided_input_ids[i].tolist(), scores[i])
    return scores


@dataclass
class SamplingInputs:
    temperature: torch.Tensor = None
    bad_words: torch.LongTensor = None
    stop_words: torch.LongTensor = None
    repetition_penalty: torch.Tensor = None
    top_k: torch.LongTensor = None
    top_p: torch.Tensor = None
    min_p: torch.Tensor = None
    random_seeds: int = None
    random_offsets: int = None
    max_top_k: int = 1
    min_top_p: float = 1.0
    response_formats: Tuple[str] = ()
    logits_processors: List[List[LogitsProcessor]] = None

    @classmethod
    def from_sampling_params(cls, seqs: List[SchedulerSequence]):
        """from samplingg params."""
        batch_size = len(seqs)
        temperature = [None] * batch_size
        repetition_penalty = [None] * batch_size
        top_k = [None] * batch_size
        top_p = [None] * batch_size
        min_p = [None] * batch_size
        bad_words = [None] * batch_size
        stop_words = [None] * batch_size
        random_seeds = [torch.seed() & 0xffffffff] * batch_size
        random_offsets = [None] * batch_size
        response_formats = [None] * batch_size
        logits_processors = [None] * batch_size

        def __gather_params():
            """gather params."""
            for idx, seq in enumerate(seqs):
                param = seq.sampling_param
                temperature[idx] = param.temperature
                repetition_penalty[idx] = param.repetition_penalty
                top_k[idx] = param.top_k
                top_p[idx] = param.top_p
                min_p[idx] = param.min_p
                random_offsets[idx] = seq.random_offsets
                response_formats[idx] = param.response_format
                if param.random_seed is not None:
                    random_seeds[idx] = param.random_seed & 0xffffffff

                bw = param.bad_words
                sw = param.stop_words
                if (not param.ignore_eos
                        and seq.num_new_tokens < param.min_new_tokens):
                    bw = bw + sw
                bad_words[idx] = bw
                stop_words[idx] = sw
                logits_processors[idx] = param.logits_processors

        def __get_topp(top_p):
            """get topp."""
            min_top_p = min(top_p)
            if min_top_p == 1.0:
                top_p = None
            else:
                top_p = torch.tensor(top_p)
            return top_p, min_top_p

        def __get_minp(min_p):
            """get minp."""
            max_min_p = max(min_p)
            if max_min_p == 0.0:
                min_p = None
            else:
                min_p = torch.Tensor(min_p)
            return min_p

        def __get_bad_words(bad_words):
            """get bad words."""
            max_bw_len = max(len(bw) for bw in bad_words)
            if max_bw_len == 0:
                return None
            if all(len(bw) == max_bw_len for bw in bad_words):
                return torch.tensor(bad_words)
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

        bad_words = __get_bad_words(bad_words)
        stop_words = __get_bad_words(stop_words)

        max_top_k = max(top_k)
        if min(top_k) <= 0:
            max_top_k = 0
        if max_top_k == 1:
            top_k = None
            top_p, min_top_p = None, 1.0
            min_p = None
            random_seeds = None
            random_offsets = None
        else:
            top_k = torch.tensor(top_k)
            top_p, min_top_p = __get_topp(top_p)
            min_p = __get_minp(min_p)
            random_seeds = torch.tensor(random_seeds)
            random_offsets = torch.tensor(random_offsets)

        sampling_input = cls(
            temperature=temperature,
            bad_words=bad_words,
            stop_words=stop_words,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            random_seeds=random_seeds,
            random_offsets=random_offsets,
            response_formats=tuple(response_formats),
            max_top_k=max_top_k,
            min_top_p=min_top_p,
            logits_processors=logits_processors,
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


def _apply_custom_logits_processors(batched_logits_processors, all_ids,
                                    logits):
    """Apply custom logits processors."""
    for seq_id, processors in enumerate(batched_logits_processors):
        if processors is not None:
            for processor in processors:
                logits[seq_id] = processor(all_ids[seq_id], logits[seq_id])
    return logits


class FusedLogitsProcessor(LogitsWarper):
    """Custom logits processor."""

    def __init__(self,
                 sampling_inputs: SamplingInputs,
                 ignore_eos: torch.Tensor,
                 tokenizer: Optional[Tokenizer] = None):
        self.sampling_inputs: SamplingInputs = sampling_inputs
        self.ignore_eos = ignore_eos
        self.tokenizer = tokenizer

    def __call__(self, all_ids: torch.LongTensor,
                 guided_input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            all_ids (torch.LongTensor): All the token ids.
            guided_input_ids (torch.LongTensor): Guided prompt ids.
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

        custom_logits_processors = self.sampling_inputs.logits_processors
        if any(custom_logits_processors):
            scores = _apply_custom_logits_processors(custom_logits_processors,
                                                     all_ids, scores)

        repetition_penalty = sampling_inputs.repetition_penalty
        if repetition_penalty is not None:
            scores = _process_repetition_penalty_(scores, all_ids,
                                                  repetition_penalty)

        temperature = sampling_inputs.temperature
        if temperature is not None:
            scores = _process_temperature_(scores, temperature)

        bad_words = sampling_inputs.bad_words
        if bad_words is not None:
            scores = _process_bad_words_(scores, bad_words)

        stop_words = sampling_inputs.stop_words
        if stop_words is not None:
            stop_words = torch.where(self.ignore_eos[:, None], stop_words, -1)
            scores = _process_bad_words_(scores, stop_words)

        scores = _guided_sampling(sampling_inputs.response_formats, scores,
                                  guided_input_ids, self.tokenizer)
        return scores

    def sampling(self, logits: torch.Tensor):
        """sampling."""
        sampling_inputs = self.sampling_inputs

        def __random_sampling(scores: torch.Tensor, indices: torch.LongTensor):
            """random sampling."""
            max_topk = sampling_inputs.max_top_k
            top_k = sampling_inputs.top_k
            if max_topk <= 0:
                max_topk = scores.size(1)
                if top_k is not None:
                    top_k = torch.where(top_k <= 0, top_k.new_tensor(max_topk),
                                        top_k)

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
            return _multinomial_sampling(softmax_scores, seeds, offsets,
                                         indices)

        if sampling_inputs.max_top_k == 1:
            return logits.argmax(-1)
        else:
            # sort logits is too slow. and we only need topk logits
            max_topk = sampling_inputs.max_top_k
            if max_topk <= 0:
                scores, indices = logits.sort(1, descending=True)
            else:
                scores, indices = logits.topk(max_topk, dim=1)
            return __random_sampling(scores, indices)
