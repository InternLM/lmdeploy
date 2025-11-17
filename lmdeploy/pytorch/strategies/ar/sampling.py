# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence

from ..base.sampling import SamplingStrategy

SeqList = List[SchedulerSequence]


def _gather_all_ids(pad_id: int, seqs: SeqList, sampling_inputs: SamplingInputs):
    """Gather history."""
    if sampling_inputs.repetition_penalty is None and not any(sampling_inputs.logits_processors):
        return None
    batch = len(seqs)
    max_len = max(seq.num_valid_ids for seq in seqs)
    output = torch.full((batch, max_len), pad_id, dtype=torch.int64)
    for idx, seq in enumerate(seqs):
        h_len = seq.num_valid_ids
        if h_len == 0:
            continue
        h_ids = torch.from_numpy(seq.valid_ids)
        output[idx, -h_len:] = h_ids
    return output


def _get_num_ignore_eos(seqs: SeqList):
    """Get num ignore eos."""
    ret = [seq.sampling_param.min_new_tokens - seq.num_new_tokens for seq in seqs]
    return torch.tensor(ret)


class ARSamplingStrategy(SamplingStrategy):
    """Sampling strategy for autoregressive models."""

    def __init__(self, pad_token_id: int) -> None:
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        self.pad_token_id = pad_token_id
        self.session_to_cleanup = []

    def make_sampling_inputs(self, seqs: SeqList) -> SamplingInputs:
        """Create sampling inputs from the sequences."""
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
        num_logprobs = [None] * batch_size
        session_to_cleanup = self.session_to_cleanup
        self.session_to_cleanup = []

        def __gather_params():
            """Gather params."""
            for idx, seq in enumerate(seqs):
                param = seq.sampling_param
                temperature[idx] = param.temperature
                repetition_penalty[idx] = param.repetition_penalty
                top_k[idx] = max(0, param.top_k)
                top_p[idx] = param.top_p
                min_p[idx] = param.min_p
                random_offsets[idx] = seq.num_valid_ids
                response_formats[idx] = param.response_format
                if param.random_seed is not None:
                    random_seeds[idx] = param.random_seed & 0xffffffff

                bw = param.bad_words
                sw = param.stop_words
                if (not param.ignore_eos and seq.num_new_tokens < param.min_new_tokens):
                    bw = bw + sw
                bad_words[idx] = bw
                stop_words[idx] = sw
                logits_processors[idx] = param.logits_processors
                num_logprobs[idx] = param.num_logprobs

        def __get_topp(top_p):
            """Get topp."""
            min_top_p = min(top_p)
            if min_top_p == 1.0:
                top_p = None
            else:
                top_p = torch.tensor(top_p)
            return top_p, min_top_p

        def __get_minp(min_p):
            """Get minp."""
            max_min_p = max(min_p)
            if max_min_p == 0.0:
                min_p = None
            else:
                min_p = torch.Tensor(min_p)
            return min_p

        def __get_bad_words(bad_words):
            """Get bad words."""
            max_bw_len = max(len(bw) for bw in bad_words)
            if max_bw_len == 0:
                return None, None
            if all(len(bw) == max_bw_len for bw in bad_words):
                ret = torch.tensor(bad_words)
                mask = torch.ones_like(ret, dtype=bool)
                return ret, mask
            ret = torch.full((batch_size, max_bw_len), -1, dtype=torch.int64)
            for idx, bw in enumerate(bad_words):
                bw_len = len(bw)
                if bw_len == 0:
                    continue
                bw = ret.new_tensor(bw)
                ret[idx, :bw_len] = bw

            mask = ret >= 0
            ret = ret.where(mask, 0)
            return ret, mask

        __gather_params()

        if all(rp == 1.0 for rp in repetition_penalty):
            repetition_penalty = None
        else:
            repetition_penalty = torch.tensor(repetition_penalty)

        temperature = torch.tensor(temperature)
        if (temperature == 1.0).all():
            # skip temperature processing if all temperature are 1.0
            temperature = None

        bad_words, bad_mask = __get_bad_words(bad_words)
        stop_words, stop_mask = __get_bad_words(stop_words)

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
            if (top_k == max_top_k).all():
                # we would perform max_top_k before top_k
                # if all top_k are same, we do not need to filter topk again
                top_k = None
            top_p, min_top_p = __get_topp(top_p)
            min_p = __get_minp(min_p)
            random_seeds = torch.tensor(random_seeds)
            random_offsets = torch.tensor(random_offsets)

        max_num_logprobs = max(num_logprobs)

        session_ctx = [{
            'session_id': seq.session.session_id,
            'seq_id': seq.seq_id,
        } for seq in seqs]

        sampling_input = SamplingInputs(
            temperature=temperature,
            bad_words=bad_words,
            bad_mask=bad_mask,
            stop_words=stop_words,
            stop_mask=stop_mask,
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
            max_num_logprobs=max_num_logprobs,
            batch_size=batch_size,
            session_ctx=session_ctx,
            session_to_cleanup=session_to_cleanup,
        )

        pad_token_id = self.pad_token_id
        sampling_input.all_ids = _gather_all_ids(pad_token_id, seqs, sampling_input)
        sampling_input.num_ignore_eos = _get_num_ignore_eos(seqs)
        return sampling_input

    def on_session_end(self, session_id: int):
        self.session_to_cleanup.append(session_id)
