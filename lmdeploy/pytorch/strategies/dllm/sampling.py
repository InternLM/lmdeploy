# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.engine.logits_process import SamplingInputs, SamplingInputsDelta
from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.model_inputs import ModelInputsDelta

from ..ar.sampling import ARSamplingStrategy
from .model_agent import DLLMExtraInputs

SeqList = List[SchedulerSequence]


class DLLMSamplingStrategy(ARSamplingStrategy):
    """Sampling strategy for autoregressive models."""

    def __init__(self, pad_token_id: int, dllm_block_length: int) -> None:
        super().__init__(pad_token_id)
        self.dllm_block_length = dllm_block_length

    @record_function('make_sampling_inputs')
    def make_sampling_inputs(self, seqs: SeqList) -> SamplingInputs:
        """Create sampling inputs from the sequences."""
        out = super().make_sampling_inputs(seqs)
        dllm_block_length = self.dllm_block_length

        # repeat tensor
        update_attr_names = [
            'temperature',
            'bad_words',
            'bad_mask',
            'stop_words',
            'stop_mask',
            'repetition_penalty',
            'top_k',
            'top_p',
            'min_p',
            'random_seeds',
            'random_offsets',
            'all_ids',
            'num_ignore_eos',
        ]
        for name in update_attr_names:
            attr = getattr(out, name)
            if attr is None:
                continue
            repeats = (dllm_block_length, ) + (1, ) * (attr.dim())
            attr = attr[None].repeat(*repeats).flatten(0, 1)
            setattr(out, name, attr)

        if len(out.response_formats) > 0:
            new_resp_formats = []
            for resp in out.response_formats:
                new_resp_formats += [resp] * dllm_block_length
            out.response_formats = tuple(new_resp_formats)

        out.batch_size *= dllm_block_length

        return out

    def merge_sampling_delta(
        self,
        sampling_delta: 'SamplingInputsDelta',
        other: 'SamplingInputsDelta',
    ) -> 'SamplingInputsDelta':
        """Merge two sampling deltas."""
        num_ignore_eos = torch.cat([sampling_delta.num_ignore_eos, other.num_ignore_eos], 0)
        random_offsets = torch.cat([sampling_delta.random_offsets, other.random_offsets], 0)

        return SamplingInputsDelta(
            num_ignore_eos=num_ignore_eos,
            random_offsets=random_offsets,
            all_ids=None,
        )

    def update_sampling_delta(
        self,
        sampling_delta: 'SamplingInputsDelta',
        delta: 'ModelInputsDelta',
    ) -> 'SamplingInputsDelta':
        """Update sampling delta with model inputs delta."""
        indices = delta.indices
        num_ignore_eos = sampling_delta.num_ignore_eos.view(-1, self.dllm_block_length)
        num_ignore_eos = num_ignore_eos[indices].flatten()
        if sampling_delta.random_offsets is not None:
            random_offsets = sampling_delta.random_offsets.view(-1, self.dllm_block_length)
            random_offsets = random_offsets[indices].flatten()
        else:
            random_offsets = None
        return SamplingInputsDelta(
            num_ignore_eos=num_ignore_eos,
            random_offsets=random_offsets,
            all_ids=None,
        )

    def step_sampling_delta(
        self,
        sampling_delta: 'SamplingInputsDelta',
        next_token_ids: torch.Tensor,
        extra_inputs: 'DLLMExtraInputs',
    ) -> 'SamplingInputsDelta':
        """Step next delta."""
        from lmdeploy.pytorch import consts
        dllm_mask = extra_inputs.dllm_mask
        dllm_block_size = self.dllm_block_length
        DLLM_UNMASKED = consts.DLLM_UNMASKED
        is_unmasked = (dllm_mask == DLLM_UNMASKED).view(-1, dllm_block_size).all(dim=1, keepdim=True)
        num_ignore_eos = sampling_delta.num_ignore_eos.view(-1, dllm_block_size)
        num_ignore_eos = torch.where(is_unmasked, num_ignore_eos - dllm_block_size, num_ignore_eos)
        sampling_delta.num_ignore_eos = num_ignore_eos.flatten()
        if sampling_delta.random_offsets is not None:
            # random offset is used to generate random numbers for multinomial sampling
            # so we need to increase it by 1 at each step
            sampling_delta.random_offsets += 1
        return sampling_delta
