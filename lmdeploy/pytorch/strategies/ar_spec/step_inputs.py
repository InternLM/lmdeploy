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

from ..ar.model_inputs import get_model_inputs_next_decoding, merge_model_inputs
from ..ar.step_inputs import _merge_sampling_delta, _reindex_sampling_delta, _step_sampling_delta
from ..base.model_agent import ExtraInputs, StoppingCriteria
from ..base.step_inputs import StepInputs
from .model_agent import ARSpecExtraInputs, ARSpecExtraOutputs


def _reindex_model_inputs_arspec(
    inputs: ModelInputs,
    delta: ModelInputsDelta,
    num_spec_tokens: int,
) -> ModelInputs:
    """Reindex AR Spec model inputs by delta.indices."""
    assert inputs.is_decoding, 'Only support update_delta in decoding.'
    indices = delta.indices
    indice_cpu = delta.indice_cpu
    block_offsets = delta.block_offsets
    max_q_seqlen = delta.max_q_seqlen
    max_kv_seqlen = delta.max_kv_seqlen
    sum_kv_seqlen = delta.sum_kv_seqlen
    num_ignored_history = delta.num_ignored_history

    # required inputs — reshape by num_spec_tokens+1 for spec decoding
    inputs_ids = inputs.input_ids.reshape(1, -1, num_spec_tokens + 1)
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

    # for mrope
    mrope_pos_ids = inputs.mrope_pos_ids
    if mrope_pos_ids is not None:
        mrope_pos_ids = mrope_pos_ids.reshape(3, -1, num_spec_tokens + 1)
        mrope_pos_ids = mrope_pos_ids[:, indices].reshape(3, -1)

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
        mrope_pos_ids=mrope_pos_ids,
    )


@dataclass
class ARSpecStepInputs(StepInputs):
    """AR Spec paradigm step inputs."""
    _pad_token_id: int = field(repr=False, default=0)
    _num_spec_tokens: int = field(repr=False, default=0)

    @record_function('StepInputs.merge_prefill')
    def merge_prefill(
        self,
        inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ARSpecExtraOutputs,
    ):
        """Add prefill result into accumulated decode state."""

        # convert prefill inputs → first decode step (with draft tokens)
        next_token_ids_expanded = next_token_ids[:, None]
        next_token_ids_expanded = torch.cat(
            [next_token_ids_expanded, extra_outputs.draft_token_ids], dim=-1)
        max_q_seqlen = next_token_ids_expanded.size(-1)
        next_token_ids_flat = next_token_ids_expanded.flatten()[None, :]
        inputs = get_model_inputs_next_decoding(
            inputs, next_token_ids_flat,
            max_q_seqlen=max_q_seqlen, model_metas=model_metas)

        # update mrope pos ids
        mrope_pos_ids = inputs.mrope_pos_ids
        if mrope_pos_ids is not None:
            offsets = torch.arange(
                max_q_seqlen, dtype=mrope_pos_ids.dtype,
                device=mrope_pos_ids.device)[None, None, :]
            mrope_pos_ids = mrope_pos_ids.unflatten(1, (-1, 1)).repeat(
                1, 1, max_q_seqlen) + offsets
            inputs.mrope_pos_ids = mrope_pos_ids.flatten(1, 2)

        extra_inputs = extra_inputs.clone()

        # advance sampling state
        stopping_criteria = stopping_criteria.clone()
        sampling_delta = _step_sampling_delta(sampling_delta, next_token_ids)

        if self.model_inputs is None:
            self.model_inputs = inputs
            self.extra_inputs = extra_inputs
            self.stopping_criteria = stopping_criteria
            self.sampling_delta = sampling_delta
        else:
            self.model_inputs = merge_model_inputs(self.model_inputs, inputs)
            self.extra_inputs = self.extra_inputs.merge(extra_inputs)
            self.stopping_criteria = self.stopping_criteria.merge(stopping_criteria)
            self.sampling_delta = _merge_sampling_delta(
                self.sampling_delta, sampling_delta, self._pad_token_id)

    def reindex(self, delta: ModelInputsDelta):
        """Shrink batch — keep only sequences at delta.indices."""
        self.model_inputs = _reindex_model_inputs_arspec(
            self.model_inputs, delta, self._num_spec_tokens)
        # reindex extra inputs (output_draft_token_ids, num_rejected_tokens)
        self.extra_inputs = self.extra_inputs.update(delta)
        self.stopping_criteria = self.stopping_criteria.update(delta)
        self.sampling_delta = _reindex_sampling_delta(self.sampling_delta, delta)

    @record_function('StepInputs.step_decode')
    def step_decode(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ARSpecExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ARSpecExtraOutputs,
    ):
        """Advance decode state for next step."""
        num_spec_tokens = self._num_spec_tokens

        # advance model state
        model_inputs.is_decoding = True
        model_inputs.model_metas = model_metas

        # update extra inputs
        extra_inputs.output_token_ids = extra_outputs.draft_token_ids

        # update inputs with rejected token adjustment
        step_seqlens = model_inputs.seq_length - extra_inputs.num_rejected_tokens
        batch_size = step_seqlens.size(0)
        input_ids = next_token_ids.new_empty((batch_size, num_spec_tokens + 1))
        input_ids[:, 0] = next_token_ids
        input_ids[:, 1:] = extra_inputs.output_draft_token_ids
        input_ids = input_ids.flatten()[None, :]
        self.model_inputs = model_inputs.step(input_ids, step_seqlens)
        self.extra_inputs = extra_inputs

        # advance sampling state
        self.stopping_criteria = stopping_criteria.clone()
        self.sampling_delta = _step_sampling_delta(sampling_delta, next_token_ids)
