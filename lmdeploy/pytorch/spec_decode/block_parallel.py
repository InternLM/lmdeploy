# Copyright (c) OpenMMLab. All rights reserved.
"""Phase-local metadata for fixed-width DFlash-family proposals.

The plan is made from the target's existing host metadata exchange, before sampling. Accepted lengths never participate
in a host collective.
"""
from dataclasses import dataclass

import torch

from ..model_inputs import DPMeta, ModelInputs


@dataclass(frozen=True)
class BlockDraftStepPlan:
    batches: tuple[int, ...]
    query_ready: tuple[bool, ...]
    global_is_decoding: bool

    @property
    def run_query(self):
        return any(self.query_ready)

    @property
    def batch_bound(self):
        return max(self.batches)


def context_inputs(inputs: ModelInputs) -> ModelInputs:
    """Detach DP metadata for local-only context KV materialization.

    Callers already select local prefill mode (is_decoding=False). Supported materializers only project/write local KV;
    they do not run DP MLP/MoE collectives or the query graph runner. Attention-TP sharding is unchanged. Never inherit
    the verifier's graph-padded counts or mutate its metadata.
    """
    return inputs.clone(dp_meta=None) if inputs.dp_meta is not None else inputs


def prepare_query(proposer, inputs: ModelInputs, cache_engine) -> ModelInputs:
    """Pad physical query rows, reusing reserved block 0 for discarded rows.

    The target allocator reserves block 0; draft caches reuse its block tables and have no independent request
    allocator. Dummy writes may overlap there, but must not touch real KV/state. Dummy outputs are unspecified and
    discarded. All participants execute the transformer AND head. Scheduler-visible output slicing is performed only
    after the head has finished.
    """
    if inputs.dp_meta is None:
        return inputs
    source_meta = inputs.dp_meta
    plan = source_meta.block_plan
    batch = inputs.seq_length.numel()
    bound = plan.batch_bound if plan else max(source_meta.dp_batches)
    global_decode = plan.global_is_decoding if plan else source_meta.dp_is_decoding
    model = proposer.model
    eager = model.backend_config.eager_mode
    if global_decode and not eager:
        bound = model._get_capture_tokens(bound)
    width = inputs.max_q_seqlen
    active = not inputs.is_dummy and not (inputs.is_chunk
                                          and not inputs.is_last_chunk)
    # Warmup dummy rows also use block 0 and execute every collective.
    real = batch if active else 0
    assert real <= bound
    cfg = cache_engine.cache_config
    pages_per_row = (width + cfg.kernel_block_size -
                     1) // cfg.kernel_block_size
    device = inputs.input_ids.device
    table_width = max(inputs.block_offsets.size(1), pages_per_row)
    block_offsets = inputs.block_offsets.new_zeros((bound, table_width))
    if real:
        block_offsets[:real, :inputs.block_offsets.
                      size(1)] = inputs.block_offsets

    def pad_rows(value, shape, fill=0):
        if value is None:
            return None
        out = value.new_full(shape, fill)
        if real:
            out[:real] = value
        return out

    ids = inputs.input_ids.new_full((bound, width),
                                    proposer.specdecode_config.mask_token_id)
    ids[:, 0] = 0
    if real:
        ids[:real] = inputs.input_ids.reshape(batch, width)
    positions = torch.arange(width, device=device).expand(bound, width).clone()
    if real:
        positions[:real] = inputs.target_position_ids.reshape(batch, width)
    history = pad_rows(inputs.history_lengths, (bound, ))
    meta = DPMeta.build(bound * width,
                        [bound * width] * len(source_meta.dp_batches))
    meta.dp_batches = [bound] * len(source_meta.dp_batches)
    meta.dp_is_decoding = global_decode
    model.get_meta().padding_batch_size = bound
    out = inputs.clone(
        input_ids=ids.reshape(1, -1),
        seq_length=inputs.seq_length.new_full((bound, ), width),
        history_lengths=history,
        block_offsets=block_offsets,
        num_ignored_history=pad_rows(inputs.num_ignored_history, (bound, )),
        state_offsets=pad_rows(inputs.state_offsets, (bound, ), -1),
        local_adapter_ids=pad_rows(inputs.local_adapter_ids, (bound, )),
        target_position_ids=positions.reshape(1, -1),
        mrope_pos_ids=None,
        model_metas=None,
        logits_indices=None,
        seq_logit_length=None,
        max_kv_seqlen=max(inputs.max_kv_seqlen if real else 0, width),
        sum_kv_seqlen=(inputs.sum_kv_seqlen if real else 0) +
        (bound - real) * width,
        dp_meta=meta,
    )
    return model.update_inputs(out)
