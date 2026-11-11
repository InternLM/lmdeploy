# Copyright (c) OpenMMLab. All rights reserved.
"""Bridge shared exact-boundary units and request-private Replay runtime.

These kernels move state only at restore/export boundaries. Replay decode math, acceptance and cursor ownership remain
in the existing Replay operators.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_state(runtime, shared, slots, units, histories,
                R0: tl.constexpr, S0: tl.constexpr, SIZE: tl.constexpr,
                NR: tl.constexpr, NS: tl.constexpr, RESTORE: tl.constexpr,
                BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + row).to(tl.int64)
    unit = tl.load(units + row).to(tl.int64)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    valid_slot = (slot > 0) & (slot < NR)
    valid_unit = (unit > 0) & (unit < NS)
    if RESTORE:
        fresh = tl.load(histories + row) == 0
        value = tl.load(shared + unit * S0 + i, (i < SIZE) & valid_unit & ~fresh, other=0)
        tl.store(runtime + slot * R0 + i, value, (i < SIZE) & valid_slot & (fresh | valid_unit))
    else:
        value = tl.load(runtime + slot * R0 + i, (i < SIZE) & valid_slot, other=0)
        tl.store(shared + unit * S0 + i, value, (i < SIZE) & valid_slot & valid_unit)


def restore_replay_checkpoint(state, metadata):
    runtime, shared = state.checkpoint, state.shared_checkpoint
    _copy_checkpoint(runtime, shared, metadata.origin_state_ids, metadata.ssm_state_meta.input_state_ids,
                     metadata.history_lengths, restore=True)


def export_replay_checkpoint(state, metadata):
    """AR force-flush has materialized the current exact boundary."""
    _copy_checkpoint(state.checkpoint, state.shared_checkpoint, metadata.origin_state_ids,
                     metadata.ssm_state_meta.output_state_ids[:, 0], metadata.history_lengths, restore=False)


def _copy_checkpoint(runtime, shared, slots, units, histories, *, restore):
    if (runtime.shape[1:] != shared.shape[1:] or runtime.dtype != shared.dtype
            or runtime.stride(1) != runtime[0].numel() // runtime.shape[1]
            or not runtime[0].is_contiguous() or not shared[0].is_contiguous()):
        raise ValueError('Replay checkpoint payloads must be matching contiguous layer entries.')
    size = runtime[0].numel()
    _copy_state[(slots.numel(), triton.cdiv(size, 256))](
        runtime, shared, slots, units, histories, runtime.stride(0), shared.stride(0), size,
        runtime.shape[0], shared.shape[0], restore, 256)


@triton.jit
def _conv_boundaries(x, runtime, shared, slots, sources, histories, cu, cp_ids, cp_offsets,
                     X_T: tl.constexpr, X_C: tl.constexpr,
                     R0: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr,
                     S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                     CHANNELS: tl.constexpr, WIDTH: tl.constexpr, RING: tl.constexpr,
                     NR: tl.constexpr, NS: tl.constexpr, SPEC: tl.constexpr,
                     DECODE: tl.constexpr, HAS_CP: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + row).to(tl.int64)
    source = tl.load(sources + row).to(tl.int64)
    history = tl.load(histories + row)
    c = tile * BLOCK + tl.arange(0, BLOCK)
    valid = (c < CHANNELS) & (slot > 0) & (slot < NR)
    restore = (source > 0) & (source < NS)
    if HAS_CP:
        cp = tl.load(cp_ids + row).to(tl.int64)
        valid_cp = (cp > 0) & (cp < NS)
    if not DECODE:
        bos = tl.load(cu + row)
        length = tl.load(cu + row + 1) - bos
        if HAS_CP:
            offset = tl.load(cp_offsets + row)
            valid_cp = valid_cp & (offset > 0) & (offset <= length)
    for i in tl.static_range(WIDTH - 1):
        if DECODE:
            # AR runtime is chronological, including the unused oldest entry.
            value = tl.load(runtime + slot * R0 + c * R1 + (i + 1) * R2, valid, other=0)
        else:
            pos = (history - (WIDTH - 1) + i) % RING if SPEC else i + 1
            initial = tl.load(runtime + slot * R0 + c * R1 + pos * R2, valid, other=0)
            restored = tl.load(shared + source * S0 + c * S1 + i * S2,
                               valid & restore, other=0)
            initial = tl.where(restore, restored, initial)
            initial = tl.where(history == 0, 0, initial)
            # Restore the accepted window only; tentative bytes remain invalid.
            tl.store(runtime + slot * R0 + c * R1 + pos * R2, initial,
                     valid & (restore | (history == 0)))
            if HAS_CP:
                token = offset - (WIDTH - 1) + i
                value = tl.load(x + (bos + token) * X_T + c * X_C,
                                valid & valid_cp & (token >= 0), other=0)
                # Short c may still include accepted history before this chunk.
                initial_i = token + WIDTH - 1
                initial_pos = (history + token) % RING if SPEC else initial_i + 1
                old = tl.load(runtime + slot * R0 + c * R1 + initial_pos * R2,
                              valid & valid_cp & (token < 0), other=0)
                old_shared = tl.load(shared + source * S0 + c * S1 + initial_i * S2,
                                     valid & valid_cp & restore & (token < 0), other=0)
                old = tl.where(restore, old_shared, old)
                old = tl.where(history == 0, 0, old)
                value = tl.where(token < 0, old, value)
        if HAS_CP:
            tl.store(shared + cp * S0 + c * S1 + i * S2, value, valid & valid_cp)


def replay_conv_boundaries(x: torch.Tensor, state, metadata, width: int):
    """Restore before conv prefill; export raw history at c, not at final e."""
    runtime, shared = state.conv_state, state.shared_conv_state
    if shared.shape[1:] != (runtime.shape[1], width - 1):
        raise ValueError('Shared Replay conv checkpoint must contain W-1 chronological inputs.')
    decode = metadata.is_decoding
    prefill = metadata.ssm_prefill_meta
    cp_ids = metadata.ssm_state_meta.output_state_ids[:, 0] if decode else prefill.checkpoint_ids
    cp_offsets = None if decode else prefill.checkpoint_offsets
    _conv_boundaries[(metadata.origin_state_ids.numel(), triton.cdiv(runtime.shape[1], 128))](
        x, runtime, shared, metadata.origin_state_ids, metadata.ssm_state_meta.input_state_ids,
        metadata.history_lengths, None if decode else prefill.cu_seqlens, cp_ids, cp_offsets,
        x.stride(1), x.stride(2), *runtime.stride(), *shared.stride(), runtime.shape[1], width,
        runtime.shape[-1], runtime.shape[0], shared.shape[0], state.cursor.Tmax > 1,
        decode, cp_ids is not None, 128)
