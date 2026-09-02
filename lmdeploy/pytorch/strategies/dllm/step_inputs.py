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

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.engine.logits_process import SamplingInputsDelta
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

from ..ar.model_inputs import merge_model_inputs
from ..base.model_agent import ExtraInputs, ExtraOutputs, StoppingCriteria
from ..base.step_inputs import StepInputs
from .model_agent import DLLMExtraInputs, DLLMExtraOutputs


def _get_model_inputs_next_decoding(inputs: ModelInputs, input_ids: torch.Tensor, max_q_seqlen,
                                    step_seqlens: torch.Tensor, model_metas) -> ModelInputs:
    """Next decoding step for DLLM."""
    if input_ids.dim() == 1:
        input_ids = input_ids[None, :]
    step_seqlens = torch.where(step_seqlens > 0, step_seqlens, inputs.seq_length - max_q_seqlen)
    return ModelInputs(
        input_ids=input_ids,
        seq_length=torch.full_like(inputs.seq_length, max_q_seqlen),
        history_lengths=inputs.history_lengths + step_seqlens,
        block_offsets=inputs.block_offsets,
        is_decoding=True,
        num_ignored_history=inputs.num_ignored_history,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=inputs.max_kv_seqlen + max_q_seqlen,
        sum_kv_seqlen=inputs.sum_kv_seqlen + inputs.seq_length.numel() * inputs.max_q_seqlen,
        local_adapter_ids=inputs.local_adapter_ids,
        model_metas=model_metas,
        state_offsets=inputs.state_offsets,
    )


def _update_dllm(next_token_ids: torch.Tensor, dllm_mask: torch.Tensor,
                 seqlens: torch.Tensor, block_size: int,
                 dllm_mask_token: int):
    """Update token_ids and dllm_mask."""
    # reshape to (batch, dllm_block_length)
    next_token_ids = next_token_ids.view(-1, block_size).clone()
    dllm_mask = dllm_mask.view(-1, block_size).clone()

    # flags
    is_cached = (dllm_mask == consts.DLLM_CACHED).all(dim=1)

    is_masked = (dllm_mask == consts.DLLM_MASKED)
    next_token_ids[is_cached[:, None] | is_masked] = dllm_mask_token
    dllm_mask[is_cached] = consts.DLLM_MASKED
    seqlens = torch.where(is_cached.view(-1), seqlens, seqlens.new_zeros((1, )))

    return next_token_ids.flatten(), dllm_mask.flatten(), seqlens


def _step_sampling_delta_dllm(
    sampling_delta: SamplingInputsDelta,
    extra_inputs: DLLMExtraInputs,
    dllm_block_size: int,
) -> SamplingInputsDelta:
    """Advance sampling delta for DLLM.(mask-dependent decrement)."""
    dllm_mask = extra_inputs.dllm_mask
    DLLM_UNMASKED = consts.DLLM_UNMASKED
    is_unmasked = (dllm_mask == DLLM_UNMASKED).view(-1, dllm_block_size).all(dim=1, keepdim=True)
    num_ignore_eos = sampling_delta.num_ignore_eos.view(-1, dllm_block_size)
    num_ignore_eos = torch.where(is_unmasked, num_ignore_eos - dllm_block_size, num_ignore_eos)
    sampling_delta.num_ignore_eos = num_ignore_eos.flatten()
    if sampling_delta.random_offsets is not None:
        sampling_delta.random_offsets += 1
    return sampling_delta


def _merge_sampling_delta_dllm(
    sampling_delta: SamplingInputsDelta,
    other: SamplingInputsDelta,
) -> SamplingInputsDelta:
    """Merge two DLLM sampling deltas."""
    num_ignore_eos = torch.cat([sampling_delta.num_ignore_eos, other.num_ignore_eos], 0)
    random_offsets = torch.cat([sampling_delta.random_offsets, other.random_offsets], 0)
    return SamplingInputsDelta(
        num_ignore_eos=num_ignore_eos,
        random_offsets=random_offsets,
        all_ids=None,
    )


def _reindex_sampling_delta_dllm(
    sampling_delta: SamplingInputsDelta,
    delta: ModelInputsDelta,
    dllm_block_length: int,
) -> SamplingInputsDelta:
    """Reindex DLLM sampling delta by delta.indices."""
    indices = delta.indices
    num_ignore_eos = sampling_delta.num_ignore_eos.view(-1, dllm_block_length)
    num_ignore_eos = num_ignore_eos[indices].flatten()
    if sampling_delta.random_offsets is not None:
        random_offsets = sampling_delta.random_offsets.view(-1, dllm_block_length)
        random_offsets = random_offsets[indices].flatten()
    else:
        random_offsets = None
    return SamplingInputsDelta(
        num_ignore_eos=num_ignore_eos,
        random_offsets=random_offsets,
        all_ids=None,
    )


def _reindex_model_inputs_dllm(
    inputs: ModelInputs,
    delta: ModelInputsDelta,
    block_size: int,
) -> ModelInputs:
    """Reindex DLLM model inputs by delta.indices."""
    assert inputs.is_decoding, 'Only support index_select in decoding.'
    indices = delta.indices
    indice_cpu = delta.indice_cpu
    block_offsets = delta.block_offsets
    max_q_seqlen = delta.max_q_seqlen
    max_kv_seqlen = delta.max_kv_seqlen
    sum_kv_seqlen = delta.sum_kv_seqlen
    num_ignored_history = delta.num_ignored_history

    # required inputs — reshape by block_size for DLLM
    inputs_ids = inputs.input_ids.reshape(1, -1, block_size)
    input_ids = inputs_ids[:, indices].reshape(1, -1)
    seq_length = inputs.seq_length[indices]
    history_lengths = inputs.history_lengths[indices]
    if block_offsets is None:
        block_offsets = inputs.block_offsets[indices]
    if num_ignored_history is None:
        num_ignored_history = inputs.num_ignored_history[indices]
    max_q_seqlen = max_q_seqlen or inputs.max_q_seqlen
    max_kv_seqlen = max_kv_seqlen or inputs.max_kv_seqlen
    sum_kv_seqlen = sum_kv_seqlen or inputs.sum_kv_seqlen

    # lora adapter ids
    local_adapter_ids = inputs.local_adapter_ids
    if local_adapter_ids is not None:
        local_adapter_ids = local_adapter_ids[indices]

    # model metas for vl models
    model_metas = inputs.model_metas
    if model_metas is not None and indice_cpu is not None:
        model_metas = [model_metas[i] for i in indice_cpu]

    # for ssm
    state_offsets = inputs.state_offsets
    if state_offsets is not None:
        state_offsets = state_offsets[indices]

    return ModelInputs(
        input_ids=input_ids,
        seq_length=seq_length,
        history_lengths=history_lengths,
        block_offsets=block_offsets,
        is_decoding=inputs.is_decoding,
        num_ignored_history=num_ignored_history,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_kv_seqlen,
        sum_kv_seqlen=sum_kv_seqlen,
        local_adapter_ids=local_adapter_ids,
        model_metas=model_metas,
        state_offsets=state_offsets,
    )


@dataclass
class DLLMStepInputs(StepInputs):
    """DLLM paradigm step inputs."""
    _block_size: int = field(repr=False, default=4)
    _dllm_mask_token: int = field(repr=False, default=0)

    @record_function('StepInputs.merge_prefill')
    def merge_prefill(
        self,
        inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: DLLMExtraOutputs,
    ):
        """Add prefill result into accumulated decode state."""
        block_size = self._block_size
        dllm_mask = extra_outputs.dllm_mask
        next_token_ids, dllm_mask, step_seqlens = _update_dllm(
            next_token_ids, dllm_mask, inputs.seq_length,
            block_size, self._dllm_mask_token)

        # convert prefill inputs → first decode step
        inputs = _get_model_inputs_next_decoding(
            inputs, next_token_ids, model_metas=model_metas,
            max_q_seqlen=block_size, step_seqlens=step_seqlens)
        extra_inputs = DLLMExtraInputs(dllm_mask=dllm_mask)

        # advance sampling state
        stopping_criteria = stopping_criteria.clone()
        sampling_delta = _step_sampling_delta_dllm(
            sampling_delta, extra_inputs, block_size)

        if self.model_inputs is None:
            self.model_inputs = inputs
            self.extra_inputs = extra_inputs
            self.stopping_criteria = stopping_criteria
            self.sampling_delta = sampling_delta
        else:
            self.model_inputs = merge_model_inputs(self.model_inputs, inputs)
            self.extra_inputs = self.extra_inputs.merge(extra_inputs)
            self.stopping_criteria = self.stopping_criteria.merge(stopping_criteria)
            self.sampling_delta = _merge_sampling_delta_dllm(
                self.sampling_delta, sampling_delta)

    def reindex(self, delta: ModelInputsDelta):
        """Shrink batch — keep only sequences at delta.indices."""
        block_size = self._block_size
        self.model_inputs = _reindex_model_inputs_dllm(
            self.model_inputs, delta, block_size)

        # reindex dllm_mask
        dllm_mask = self.extra_inputs.dllm_mask
        dllm_mask = dllm_mask.reshape(-1, block_size)
        dllm_mask = dllm_mask[delta.indices].flatten()
        self.extra_inputs = DLLMExtraInputs(dllm_mask=dllm_mask)

        self.stopping_criteria = self.stopping_criteria.update(delta)
        self.sampling_delta = _reindex_sampling_delta_dllm(
            self.sampling_delta, delta, block_size)

    @record_function('StepInputs.step_decode')
    def step_decode(
        self,
        model_inputs: ModelInputs,
        extra_inputs: DLLMExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ExtraOutputs,
    ):
        """Advance decode state for next step."""
        block_size = self._block_size

        # advance model state
        model_inputs.is_decoding = True
        model_inputs.model_metas = model_metas
        dllm_mask = extra_inputs.dllm_mask

        next_token_ids, dllm_mask, step_seqlens = _update_dllm(
            next_token_ids, dllm_mask, model_inputs.seq_length,
            block_size, self._dllm_mask_token)
        self.model_inputs = model_inputs.step(next_token_ids, step_seqlens)
        self.extra_inputs = DLLMExtraInputs(dllm_mask=dllm_mask)

        # advance sampling state
        self.stopping_criteria = stopping_criteria.clone()
        self.sampling_delta = _step_sampling_delta_dllm(
            sampling_delta, self.extra_inputs, block_size)
