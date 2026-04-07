# Copyright (c) OpenMMLab. All rights reserved.
# StepInputs lifecycle in _async_step:
#
# Prefill path:
#   1. reindex(delta)              — shrink batch (finished seqs removed)
#   2. model forward
#   3. merge_prefill(output)       — add prefill result into decode state
#
# Decode path:
#   1. reindex(delta)              — shrink batch
#   2. model forward
#   3. step_decode(output)         — advance to next decode step

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.engine.logits_process import SamplingInputsDelta
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

from ..base.model_agent import ExtraInputs, ExtraOutputs, StoppingCriteria
from ..base.step_inputs import StepInputs
from .model_inputs import get_model_inputs_next_decoding, index_select_model_inputs, merge_model_inputs


def step_sampling_delta(sampling_delta: SamplingInputsDelta,
                         next_token_ids: torch.Tensor) -> SamplingInputsDelta:
    """Advance sampling delta for one decode step."""
    sampling_delta.num_ignore_eos = sampling_delta.num_ignore_eos - 1
    if sampling_delta.random_offsets is not None:
        sampling_delta.random_offsets += 1
    all_ids = sampling_delta.all_ids
    if all_ids is not None:
        sampling_delta.all_ids = torch.cat([all_ids, next_token_ids[:, None]], 1)
    return sampling_delta


def merge_sampling_delta(
    sampling_delta: SamplingInputsDelta,
    other: SamplingInputsDelta,
    pad_token_id: int,
) -> SamplingInputsDelta:
    """Merge two sampling deltas by concatenation."""
    num_ignore_eos = torch.cat([sampling_delta.num_ignore_eos, other.num_ignore_eos], 0)
    random_offsets = torch.cat([sampling_delta.random_offsets, other.random_offsets], 0)

    batch_size = num_ignore_eos.size(0)
    all_ids0 = sampling_delta.all_ids
    all_ids1 = other.all_ids
    if all_ids0 is None and all_ids1 is None:
        all_ids = None
    else:
        max_len0 = 0 if all_ids0 is None else all_ids0.size(1)
        max_len1 = 0 if all_ids1 is None else all_ids1.size(1)
        max_len = max(max_len0, max_len1)
        all_ids = torch.full((batch_size, max_len),
                             pad_token_id,
                             dtype=torch.int64,
                             device=num_ignore_eos.device)
        if all_ids0 is not None:
            bs0 = all_ids0.size(0)
            all_ids[:bs0, :max_len0] = all_ids0
        if all_ids1 is not None:
            bs1 = all_ids1.size(0)
            all_ids[-bs1:, :max_len1] = all_ids1

    return SamplingInputsDelta(
        num_ignore_eos=num_ignore_eos,
        random_offsets=random_offsets,
        all_ids=all_ids,
    )


def reindex_sampling_delta(
    sampling_delta: SamplingInputsDelta,
    delta: ModelInputsDelta,
) -> SamplingInputsDelta:
    """Reindex sampling delta by delta.indices."""
    indices = delta.indices
    num_ignore_eos = sampling_delta.num_ignore_eos[indices]
    if sampling_delta.random_offsets is not None:
        random_offsets = sampling_delta.random_offsets[indices]
    else:
        random_offsets = None
    all_ids = sampling_delta.all_ids
    if all_ids is not None:
        all_ids = all_ids[indices]
    return SamplingInputsDelta(
        num_ignore_eos=num_ignore_eos,
        random_offsets=random_offsets,
        all_ids=all_ids,
    )


def _reindex_model_inputs(inputs: ModelInputs, delta: ModelInputsDelta) -> ModelInputs:
    """Reindex model inputs by delta.indices."""
    return index_select_model_inputs(
        inputs=inputs,
        indices=delta.indices,
        indice_cpu=delta.indice_cpu,
        block_offsets=delta.block_offsets,
        max_q_seqlen=delta.max_q_seqlen,
        max_kv_seqlen=delta.max_kv_seqlen,
        sum_kv_seqlen=delta.sum_kv_seqlen,
        num_ignored_history=delta.num_ignored_history,
    )


@dataclass
class ARStepInputs(StepInputs):
    """AR paradigm step inputs."""
    _pad_token_id: int = field(repr=False, default=0)

    @record_function('StepInputs.merge_prefill')
    def merge_prefill(
        self,
        inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ExtraOutputs,
    ):
        """Add prefill result into accumulated decode state."""
        # convert prefill inputs → first decode step
        inputs = get_model_inputs_next_decoding(
            inputs, next_token_ids, max_q_seqlen=1, model_metas=model_metas)

        # advance sampling state
        stopping_criteria = stopping_criteria.clone()
        sampling_delta = step_sampling_delta(sampling_delta, next_token_ids)

        if self.model_inputs is None:
            self.model_inputs = inputs
            self.extra_inputs = extra_inputs
            self.stopping_criteria = stopping_criteria
            self.sampling_delta = sampling_delta
        else:
            self.model_inputs = merge_model_inputs(self.model_inputs, inputs)
            self.extra_inputs = self.extra_inputs.merge(extra_inputs)
            self.stopping_criteria = self.stopping_criteria.merge(stopping_criteria)
            self.sampling_delta = merge_sampling_delta(
                self.sampling_delta, sampling_delta, self._pad_token_id)

    def reindex(self, delta: ModelInputsDelta):
        """Shrink batch — keep only sequences at delta.indices."""
        self.model_inputs = _reindex_model_inputs(self.model_inputs, delta)
        # AR has no extra_inputs to reindex
        self.stopping_criteria = self.stopping_criteria.update(delta)
        self.sampling_delta = reindex_sampling_delta(self.sampling_delta, delta)

    @record_function('StepInputs.step_decode')
    def step_decode(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ExtraOutputs,
    ):
        """Advance decode state for next step."""
        # advance model state
        model_inputs.is_decoding = True
        model_inputs.model_metas = model_metas
        self.model_inputs = model_inputs.step(next_token_ids, model_inputs.seq_length)
        self.extra_inputs = extra_inputs

        # advance sampling state
        self.stopping_criteria = stopping_criteria.clone()
        self.sampling_delta = step_sampling_delta(sampling_delta, next_token_ids)
