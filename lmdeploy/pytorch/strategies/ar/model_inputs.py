# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence

import numpy as np
import torch

from lmdeploy.pytorch.model_inputs import ModelInputs

from ..base.model_inputs import MakeDummyMeta, ModelInputsStrategy, make_dummy_inputs


def get_model_inputs_next_decoding(inputs: ModelInputs, input_ids: torch.Tensor, max_q_seqlen: int,
                                   model_metas) -> ModelInputs:
    """Next decoding step."""
    if input_ids.dim() == 1:
        input_ids = input_ids[None, :]
    state_offsets = inputs.state_offsets
    if state_offsets is not None:
        state_offsets = state_offsets.clone()

    # mrope
    mrope_pos_ids = inputs.mrope_pos_ids
    if mrope_pos_ids is not None:
        index = inputs.seq_length.cumsum(0) - 1
        mrope_pos_ids = mrope_pos_ids[:, index] + 1

    # V4 paged allocators: carry per-request identity + CPU kv lengths into
    # the first decode step. The next delta re-supplies fresh values, but the
    # transition forward consumes THESE inputs directly; without them the V4
    # worker falls to slot-keyed allocation (seq_ids=None fallback), whose
    # entries are only freed when the slot disappears -> orphaned pool blocks
    # (one per prefill->decode transition). inputs.kv_seqlens_cpu already
    # covers this prefill forward's tokens; the next forward adds one token
    # per seq (== max_q_seqlen).
    seq_ids = getattr(inputs, 'seq_ids', None)
    kv_seqlens_cpu = getattr(inputs, 'kv_seqlens_cpu', None)
    if kv_seqlens_cpu is not None:
        kv_seqlens_cpu = [kv + max_q_seqlen for kv in kv_seqlens_cpu]
    return ModelInputs(
        input_ids=input_ids,
        seq_length=torch.full_like(inputs.seq_length, max_q_seqlen),
        history_lengths=inputs.history_lengths + inputs.seq_length,
        block_offsets=inputs.block_offsets,
        is_decoding=True,
        num_ignored_history=inputs.num_ignored_history.clone(),
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=inputs.max_kv_seqlen + max_q_seqlen,
        sum_kv_seqlen=inputs.sum_kv_seqlen + inputs.seq_length.numel() * inputs.max_q_seqlen,
        local_adapter_ids=inputs.local_adapter_ids,
        model_metas=model_metas,
        state_offsets=state_offsets,
        mrope_pos_ids=mrope_pos_ids,
        kv_seqlens_cpu=kv_seqlens_cpu,
        seq_ids=seq_ids,
    )


def merge_model_inputs(inputs: ModelInputs, other: ModelInputs) -> ModelInputs:
    """Merge model inputs."""

    def __try_pad_block_offsets(block_offsets: torch.Tensor, target_size: int):
        """Try pad block offsets to target size."""
        cur_size = block_offsets.size(1)
        if cur_size < target_size:
            pad_size = target_size - cur_size
            pad_tensor = torch.zeros((block_offsets.size(0), pad_size),
                                     dtype=block_offsets.dtype,
                                     device=block_offsets.device)
            block_offsets = torch.cat([block_offsets, pad_tensor], dim=1)
        return block_offsets

    assert inputs.is_decoding and other.is_decoding, 'Only support merge in decoding.'
    input_ids = torch.cat([inputs.input_ids, other.input_ids], dim=-1)
    seq_length = torch.cat([inputs.seq_length, other.seq_length], dim=0)
    history_lengths = torch.cat([inputs.history_lengths, other.history_lengths], dim=0)

    # block offsets
    max_blocks = max(inputs.block_offsets.size(1), other.block_offsets.size(1))
    block_offsets0 = __try_pad_block_offsets(inputs.block_offsets, max_blocks)
    block_offsets1 = __try_pad_block_offsets(other.block_offsets, max_blocks)
    block_offsets = torch.cat([block_offsets0, block_offsets1], dim=0)
    num_ignored_history = torch.cat([inputs.num_ignored_history, other.num_ignored_history], dim=0)

    # lora adapter ids
    local_adapter_ids = inputs.local_adapter_ids
    if local_adapter_ids is not None and other.local_adapter_ids is not None:
        local_adapter_ids = torch.cat([local_adapter_ids, other.local_adapter_ids], dim=0)

    # model metas for vl models
    model_metas = None
    if inputs.model_metas is not None and other.model_metas is not None:
        model_metas = inputs.model_metas + other.model_metas

    # ssm
    state_offsets = None
    if inputs.state_offsets is not None:
        state_offsets = torch.cat([inputs.state_offsets, other.state_offsets], dim=0)

    # mrope
    mrope_pos_ids = None
    if inputs.mrope_pos_ids is not None:
        assert other.mrope_pos_ids is not None
        mrope_pos_ids = torch.cat([inputs.mrope_pos_ids, other.mrope_pos_ids], dim=1)

    # V4 paged allocators: carry per-request identity + CPU kv lengths across
    # the merge. Both sides are decode-step inputs; if either side lacks them
    # (e.g. dummy/step()-advanced inputs) drop to None -> the worker's legacy
    # fallback, which is safe but loses seq_id keying for this step only.
    seq_ids = None
    if inputs.seq_ids is not None and other.seq_ids is not None:
        seq_ids = inputs.seq_ids + other.seq_ids
    kv_seqlens_cpu = None
    if inputs.kv_seqlens_cpu is not None and other.kv_seqlens_cpu is not None:
        kv_seqlens_cpu = inputs.kv_seqlens_cpu + other.kv_seqlens_cpu

    return ModelInputs(
        input_ids=input_ids,
        seq_length=seq_length,
        history_lengths=history_lengths,
        block_offsets=block_offsets,
        is_decoding=inputs.is_decoding,
        num_ignored_history=num_ignored_history,
        max_q_seqlen=max(inputs.max_q_seqlen, other.max_q_seqlen),
        max_kv_seqlen=max(inputs.max_kv_seqlen, other.max_kv_seqlen),
        sum_kv_seqlen=inputs.sum_kv_seqlen + other.sum_kv_seqlen,
        local_adapter_ids=local_adapter_ids,
        model_metas=model_metas,
        state_offsets=state_offsets,
        mrope_pos_ids=mrope_pos_ids,
        kv_seqlens_cpu=kv_seqlens_cpu,
        seq_ids=seq_ids,
    )


class ARModelInputsStrategy(ModelInputsStrategy):

    def make_dummy(self,
                   batch_size: int,
                   is_decoding: bool,
                   device: str = 'cpu',
                   dummy_block_id: int = 0,
                   vocab_size: int = 1,
                   meta: MakeDummyMeta | None = None) -> ModelInputs:
        """Create dummy model inputs."""
        return make_dummy_inputs(batch_size,
                                 max_q_seqlen=1,
                                 is_decoding=is_decoding,
                                 device=device,
                                 dummy_block_id=dummy_block_id,
                                 vocab_size=vocab_size,
                                 meta=meta)


def index_select_model_inputs(inputs: ModelInputs,
                              indices: torch.Tensor,
                              indice_cpu: np.ndarray = None,
                              block_offsets: torch.Tensor = None,
                              max_q_seqlen: int | None = None,
                              max_kv_seqlen: int | None = None,
                              sum_kv_seqlen: int | None = None,
                              num_ignored_history: torch.Tensor | None = None,
                              kv_seqlens_cpu: list[int] | None = None,
                              seq_ids: list[int] | None = None,
                              state_prefix_cache_save_src_offsets: Sequence[int] | None = None,
                              state_prefix_cache_save_offsets: Sequence[int] | None = None):
    """Index select model inputs by indices."""
    assert inputs.is_decoding, 'Only support index_select in decoding.'

    if len(indices) == len(inputs.seq_length):
        # we will not change the order of indices
        # so same length means no change
        indices = Ellipsis

    # required inputs
    input_ids = inputs.input_ids[..., indices]
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

    # spec decoding
    target_hidden_states = inputs.target_hidden_states
    if target_hidden_states is not None:
        target_hidden_states = target_hidden_states[indices]
    target_position_ids = inputs.target_position_ids
    if target_position_ids is not None:
        target_position_ids = target_position_ids[indices]

    # mrope
    mrope_pos_ids = inputs.mrope_pos_ids
    if mrope_pos_ids is not None:
        mrope_pos_ids = mrope_pos_ids[:, indices]

    # return new inputs
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
        state_prefix_cache_save_src_offsets=state_prefix_cache_save_src_offsets,
        state_prefix_cache_save_offsets=state_prefix_cache_save_offsets,
        target_hidden_states=target_hidden_states,
        target_position_ids=target_position_ids,
        mrope_pos_ids=mrope_pos_ids,
        kv_seqlens_cpu=kv_seqlens_cpu,
        seq_ids=seq_ids,
    )
