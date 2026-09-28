# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _update_kpool_kernel(
    Keys, Scores, TailKeys, TailScores, StateIds, History,
    ClosedKeys, ClosedScores, GroupIds, Valid,
    BATCH: tl.constexpr, STEPS: tl.constexpr, STATES: tl.constexpr, RING: tl.constexpr,
    POOL: tl.constexpr, WIDTH: tl.constexpr, BLOCK_D: tl.constexpr,
    stride_kb: tl.constexpr, stride_kt: tl.constexpr, stride_kd: tl.constexpr,
    stride_sb: tl.constexpr, stride_st: tl.constexpr, stride_sd: tl.constexpr,
    stride_tkb: tl.constexpr, stride_tkr: tl.constexpr,
    stride_tsb: tl.constexpr, stride_tsr: tl.constexpr,
):
    request = tl.program_id(0)
    state_id = tl.load(StateIds + request).to(tl.int64)
    valid = (state_id >= 0) & (state_id < STATES)
    history = tl.maximum(tl.load(History + request).to(tl.int64), 0)
    slot = tl.arange(0, POOL)[:, None]
    d = tl.arange(0, BLOCK_D)[None, :]
    mask = valid & (d < WIDTH)
    tail_k = tl.load(TailKeys + state_id * stride_tkb + history % RING * stride_tkr + slot * WIDTH + d,
                     mask & (slot < history % POOL), other=0)
    tail_s = tl.load(TailScores + state_id * stride_tsb + history % RING * stride_tsr + slot * WIDTH + d,
                     mask & (slot < history % POOL), other=0)
    # A request owns its state row. Keep time sequential in registers, and
    # persist every checkpoint so rejection can resume at any accepted prefix.
    for step in range(STEPS):
        position = (history + step) % POOL
        key = tl.load(Keys + request * stride_kb + step * stride_kt + d * stride_kd, mask, other=0)
        score = tl.load(Scores + request * stride_sb + step * stride_st + d * stride_sd, mask, other=0)
        tail_k = tl.where(slot == position, key, tail_k)
        tail_s = tl.where(slot == position, score, tail_s)
        row = step * BATCH + request
        offset = (row * POOL + slot) * WIDTH + d
        tl.store(ClosedKeys + offset, tail_k, d < WIDTH)
        tl.store(ClosedScores + offset, tail_s, d < WIDTH)
        close = position == POOL - 1
        tl.store(GroupIds + row, (history + step) // POOL)
        tl.store(Valid + row, valid & close)
        tail_k = tl.where(close, 0, tail_k)
        tail_s = tl.where(close, 0, tail_s)
        checkpoint = (history + step + 1) % RING
        tl.store(TailKeys + state_id * stride_tkb + checkpoint * stride_tkr + slot * WIDTH + d, tail_k, mask)
        tl.store(TailScores + state_id * stride_tsb + checkpoint * stride_tsr + slot * WIDTH + d, tail_s, mask)


def update_kpool(keys, scores, tail_keys, tail_scores, state_ids, history_lengths, pool_size):
    """Update decode/verify tails in place, emitting step-major closed pools.

    Inputs use [batch, steps, width]. State uses [states, pool, width] for AR or [states, ring, pool, width] for
    verification. Live state ids must be unique; invalid ids never write state. All output shapes depend only on input
    shapes.
    """
    if keys.ndim != 3 or scores.shape != keys.shape:
        raise ValueError('Expected matching [batch, steps, width] keys and scores.')
    batch, steps, width = keys.shape
    if not batch or not steps or pool_size <= 1 or pool_size & (pool_size - 1):
        raise ValueError('Expected a nonempty batch/sequence and a power-of-two pool size.')
    if state_ids.shape != (batch,) or history_lengths.shape != (batch,):
        raise ValueError('Expected one state id and history length per request.')
    if tail_keys.shape != tail_scores.shape or tail_keys.ndim not in (3, 4):
        raise ValueError('Expected matching AR or verification tail caches.')
    if tail_keys.ndim == 3:
        tail_keys, tail_scores = tail_keys.unsqueeze(1), tail_scores.unsqueeze(1)
    elif steps > tail_keys.size(1):
        raise ValueError('The checkpoint ring must hold every verification step.')
    if tail_keys.shape[2:] != (pool_size, width):
        raise ValueError('Tail cache geometry does not match decode inputs.')
    if tail_keys.stride()[2:] != (width, 1) or tail_scores.stride()[2:] != (width, 1):
        raise ValueError('Tail cache rows must be contiguous.')
    if keys.dtype != tail_keys.dtype or scores.dtype != tail_scores.dtype:
        raise ValueError('Keys and scores must match their respective tail cache dtypes.')
    closed_keys = keys.new_empty((steps * batch, pool_size, width))
    closed_scores = scores.new_empty((steps * batch, pool_size, width))
    groups = history_lengths.new_empty(steps * batch)
    valid = torch.empty(steps * batch, device=keys.device, dtype=torch.bool)
    _update_kpool_kernel[(batch,)](
        keys, scores, tail_keys, tail_scores, state_ids.contiguous(), history_lengths.contiguous(),
        closed_keys, closed_scores, groups, valid,
        batch, steps, tail_keys.size(0), tail_keys.size(1), pool_size, width, triton.next_power_of_2(width),
        *keys.stride(), *scores.stride(), *tail_keys.stride()[:2], *tail_scores.stride()[:2], num_warps=4)
    return closed_keys, closed_scores, groups, valid


@triton.jit
def _prefill_offsets_kernel(Q, KV, Counts, Starts, QEnds,
                            BATCH: tl.constexpr, POOL: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.arange(0, BLOCK)
    q = tl.load(Q + request, request < BATCH, other=0).to(tl.int32)
    groups = tl.load(KV + request, request < BATCH, other=0).to(tl.int32) // POOL
    tl.store(Counts + request, groups, request < BATCH)
    tl.store(Starts + request, tl.cumsum(groups) - groups, request < BATCH)
    tl.store(QEnds + request, tl.cumsum(q), request < BATCH)


@triton.jit
def _prefill_query_metadata_kernel(KV, QEnds, Starts, Seq, Lengths, QueryStarts, QueryEnds,
                                   ROWS: tl.constexpr, BATCH: tl.constexpr, POOL: tl.constexpr,
                                   BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), BATCH, tl.int32)
    # Upper-bound search skips empty requests without a host ragged loop.
    while tl.sum((low < high).to(tl.int32), 0) > 0:
        mid = (low + high) // 2
        end = tl.load(QEnds + mid, mid < BATCH, other=2147483647)
        active = low < high
        right = row >= end
        low = tl.where(active & right, mid + 1, low)
        high = tl.where(active & ~right, mid, high)
    request = tl.minimum(low, BATCH - 1)
    seq = tl.load(KV + request).to(tl.int32) + row - tl.load(QEnds + request) + 1
    start = tl.load(Starts + request)
    tl.store(Seq + row, seq, row < ROWS)
    tl.store(Lengths + row, seq // POOL, row < ROWS)
    tl.store(QueryStarts + row, start, row < ROWS)
    tl.store(QueryEnds + row, start + seq // POOL, row < ROWS)


def kpool_prefill_metadata(q_seqlens, kv_seqlens, rows, pool_size):
    """Build compressed-cache offsets and causal query bounds on device."""
    batch = q_seqlens.numel()
    counts = torch.empty(batch, device=q_seqlens.device, dtype=torch.int32)
    starts = torch.empty_like(counts)
    q_ends = torch.empty_like(counts)
    seq = torch.empty(rows, device=q_seqlens.device, dtype=torch.int32)
    lengths = torch.empty_like(seq)
    query_starts = torch.empty_like(seq)
    query_ends = torch.empty_like(seq)
    _prefill_offsets_kernel[(1,)](
        q_seqlens.contiguous(), kv_seqlens.contiguous(), counts, starts, q_ends,
        batch, pool_size, triton.next_power_of_2(batch))
    if rows:
        _prefill_query_metadata_kernel[(triton.cdiv(rows, 128),)](
            kv_seqlens.contiguous(), q_ends, starts, seq, lengths, query_starts, query_ends,
            rows, batch, pool_size, 128)
    return counts, starts, seq, lengths, query_starts, query_ends


@triton.jit
def _partition_kpool_kernel(
    Keys, Scores, TailKeys, TailScores, StateIds, QLens, KVLens,
    ClosedKeys, ClosedScores, GroupIds, RequestIds, Valid, NextKeys, NextScores,
    BATCH: tl.constexpr, CAPACITY: tl.constexpr, STATES: tl.constexpr,
    POOL: tl.constexpr, WIDTH: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
    STRIDE_TK: tl.constexpr, STRIDE_TS: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, POOL)
    d = tl.arange(0, BLOCK_D)
    group = tl.full((), 0, tl.int64)
    if row < CAPACITY:
        batches = tl.arange(0, BLOCK_B)
        q = tl.load(QLens + batches, batches < BATCH, other=0)
        kv = tl.load(KVLens + batches, batches < BATCH, other=0)
        counts = ((kv - q) % POOL + q) // POOL
        ends = tl.cumsum(counts)
        request = tl.minimum(tl.sum(((row >= ends) & (batches < BATCH)).to(tl.int32)), BATCH - 1)
        group_start = tl.sum(tl.where(batches == request, ends - counts, 0))
        token_start = tl.sum(tl.where(batches < request, q, 0))
        valid = row < tl.sum(counts)
        q_len = tl.load(QLens + request)
        kv_len = tl.load(KVLens + request)
        history = kv_len - q_len
        offsets = (row - group_start) * POOL + slots - history % POOL
        group = (history // POOL + row - group_start).to(tl.int64)
    else:
        request = row - CAPACITY
        batches = tl.arange(0, BLOCK_B)
        q = tl.load(QLens + batches, batches < request, other=0)
        token_start = tl.sum(q)
        q_len = tl.load(QLens + request)
        kv_len = tl.load(KVLens + request)
        history = kv_len - q_len
        offsets = ((history % POOL + q_len) // POOL) * POOL + slots - history % POOL
        valid = True
    state_id = tl.load(StateIds + request)
    valid = valid & (state_id >= 0) & (state_id < STATES)
    active_slots = tl.full((POOL,), True, tl.int1)
    if row >= CAPACITY:
        active_slots = slots < kv_len % POOL
    prior_mask = valid & active_slots & (offsets < 0)
    token_mask = valid & active_slots & (offsets >= 0) & (offsets < q_len)
    prior_offsets = offsets + history % POOL
    old_k = tl.load(TailKeys + state_id * STRIDE_TK + prior_offsets[:, None] * WIDTH + d[None, :],
                    prior_mask[:, None] & (d[None, :] < WIDTH), other=0)
    old_s = tl.load(TailScores + state_id * STRIDE_TS + prior_offsets[:, None] * WIDTH + d[None, :],
                    prior_mask[:, None] & (d[None, :] < WIDTH), other=0)
    new_k = tl.load(Keys + (token_start + offsets[:, None]) * WIDTH + d[None, :],
                    token_mask[:, None] & (d[None, :] < WIDTH), other=0)
    new_s = tl.load(Scores + (token_start + offsets[:, None]) * WIDTH + d[None, :],
                    token_mask[:, None] & (d[None, :] < WIDTH), other=0)
    key = tl.where((offsets < 0)[:, None], old_k, new_k)
    score = tl.where((offsets < 0)[:, None], old_s, new_s)
    if row < CAPACITY:
        tl.store(ClosedKeys + (row * POOL + slots[:, None]) * WIDTH + d[None, :], key, d[None, :] < WIDTH)
        tl.store(ClosedScores + (row * POOL + slots[:, None]) * WIDTH + d[None, :], score, d[None, :] < WIDTH)
        tl.store(GroupIds + row, group)
        tl.store(RequestIds + row, request)
        tl.store(Valid + row, valid)
    else:
        tl.store(NextKeys + (request * POOL + slots[:, None]) * WIDTH + d[None, :], key, d[None, :] < WIDTH)
        tl.store(NextScores + (request * POOL + slots[:, None]) * WIDTH + d[None, :], score, d[None, :] < WIDTH)


def partition_kpool(keys, scores, tail_keys, tail_scores, state_ids, q_seqlens, kv_seqlens, pool_size):
    """Assemble ragged closed pools and next tails without host metadata reads.

    Group capacity depends only on input shapes. Invalid group rows are zero padded and masked; persistent state is read
    but never modified here.
    """
    batch = q_seqlens.numel()
    tokens, width = keys.shape
    if pool_size <= 1 or pool_size & (pool_size - 1) or not batch or scores.shape != keys.shape:
        raise ValueError('Expected matching keys/scores, a nonempty batch and a power-of-two pool size.')
    if state_ids.shape != (batch,) or kv_seqlens.shape != (batch,):
        raise ValueError('Expected one state id and KV length per request.')
    if tail_keys.shape != tail_scores.shape or tail_keys.shape[1:] != (pool_size, width):
        raise ValueError('Expected matching [states, pool_size, width] tail caches.')
    if tail_keys.stride()[1:] != (width, 1) or tail_scores.stride()[1:] != (width, 1):
        raise ValueError('Tail cache rows must be contiguous.')
    capacity = (tokens + batch * (pool_size - 1)) // pool_size
    closed_keys = keys.new_empty((capacity, pool_size, width), dtype=torch.promote_types(keys.dtype, tail_keys.dtype))
    closed_scores = scores.new_empty((capacity, pool_size, width),
                                     dtype=torch.promote_types(scores.dtype, tail_scores.dtype))
    groups = state_ids.new_empty(capacity)
    requests = state_ids.new_empty(capacity)
    valid = torch.empty(capacity, device=keys.device, dtype=torch.bool)
    next_keys = tail_keys.new_empty((batch, pool_size, width))
    next_scores = tail_scores.new_empty((batch, pool_size, width))
    _partition_kpool_kernel[(capacity + batch,)](
        keys.contiguous(), scores.contiguous(), tail_keys, tail_scores,
        state_ids.contiguous(), q_seqlens.contiguous(), kv_seqlens.contiguous(),
        closed_keys, closed_scores, groups, requests, valid, next_keys, next_scores,
        batch, capacity, tail_keys.size(0), pool_size, width, triton.next_power_of_2(batch),
        triton.next_power_of_2(width), tail_keys.stride(0), tail_scores.stride(0), num_warps=4)
    return closed_keys, closed_scores, groups, requests, valid, next_keys, next_scores


@triton.jit
def _normalized_hadamard(value, WIDTH: tl.constexpr, LEVELS: tl.constexpr):
    d = tl.arange(0, WIDTH)
    if len(value.shape) == 2:
        d = d[None, :]
    for level in tl.static_range(LEVELS):
        stride = 1 << level
        other = tl.gather(value, tl.broadcast_to(d ^ stride, value.shape), axis=len(value.shape) - 1)
        value = tl.where((d & stride) == 0, value + other, other - value)
    return value * (WIDTH**-0.5)


@triton.jit
def _rotate_kpool_query_kernel(Query, Out, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                              LEVELS: tl.constexpr, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets = rows[:, None] * WIDTH + tl.arange(0, WIDTH)[None, :]
    value = tl.load(Query + offsets, rows[:, None] < ROWS, other=0).to(tl.float32)
    value = _normalized_hadamard(value, WIDTH, LEVELS)
    tl.store(Out + offsets, value, rows[:, None] < ROWS)


def rotate_kpool_query(query: torch.Tensor) -> torch.Tensor:
    """Reuse the compression butterfly, retaining FP32 arithmetic and the
    output cast."""
    width = query.size(-1)
    if width <= 0 or width & (width - 1):
        raise ValueError('Query width must be a positive power of two.')
    out = torch.empty_like(query, memory_format=torch.contiguous_format)
    rows = query.numel() // width
    if rows:
        _rotate_kpool_query_kernel[(triton.cdiv(rows, 4),)](
            query.contiguous(), out, rows, width, width.bit_length() - 1, 4,
            num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _compress_kpool_kernel(K, S, A, O, Scale, WIDTH: tl.constexpr, POOL: tl.constexpr,
                    ONLINE: tl.constexpr, ROUND: tl.constexpr, LEVELS: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, WIDTH)
    maximum = tl.full((WIDTH,), -float('inf'), tl.float32)
    denominator = tl.full((WIDTH,), 0, tl.float32)
    accumulator = tl.full((WIDTH,), 0, tl.float32)
    if not ONLINE:
        for slot in tl.static_range(POOL):
            score = tl.load(S + (row * POOL + slot) * WIDTH + d).to(tl.float32)
            score += tl.load(A + slot * WIDTH + d).to(tl.float32)
            maximum = tl.maximum(maximum, score)
    for slot in tl.static_range(POOL):
        score = tl.load(S + (row * POOL + slot) * WIDTH + d).to(tl.float32)
        score += tl.load(A + slot * WIDTH + d).to(tl.float32)
        if ONLINE:
            new_maximum = tl.maximum(maximum, score)
            rescale = libdevice.exp(maximum - new_maximum)
            denominator = denominator * rescale
            accumulator = accumulator * rescale
            maximum = new_maximum
        probability = libdevice.exp(score - maximum)
        key = tl.load(K + (row * POOL + slot) * WIDTH + d).to(tl.float32)
        denominator = denominator + probability
        accumulator = accumulator + key * probability
    value = tl.div_rn(accumulator, denominator).to(tl.bfloat16).to(tl.float32)
    value = _normalized_hadamard(value, WIDTH, LEVELS).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0), 1e-4) * (1.0 / 448.0)
    if ROUND:
        scale = libdevice.exp2(libdevice.ceil(libdevice.log2(scale)))
    quant = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    tl.store(O + row * WIDTH + d, quant.to(tl.float8e4nv))
    tl.store(Scale + row, scale)


def compress_kpool(keys: torch.Tensor, scores: torch.Tensor, ape: torch.Tensor,
                   *, mode: str, round_scale: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse weighted pooling, BF16 Hadamard rotation and one-block FP8
    quantization.

    The online/two-pass reduction order and both BF16 round trips match the reference KPool operation. Precise libdevice
    functions and disabled FMA fusion preserve its FP32 arithmetic boundaries.
    """
    if mode not in ('extend', 'decode'):
        raise ValueError(f'Unsupported pool compression mode: {mode}')
    if keys.ndim != 3 or scores.shape != keys.shape or ape.shape != keys.shape[1:]:
        raise ValueError('Expected matching [groups, pool, width] keys/scores and [pool, width] APE.')
    groups, pool, width = keys.shape
    if width <= 0 or width & (width - 1):
        raise ValueError('Pool width must be a positive power of two.')
    out = torch.empty(groups, width, device=keys.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(groups, 1, device=keys.device, dtype=torch.float32)
    if groups:
        _compress_kpool_kernel[(groups,)](
            keys.contiguous(), scores.contiguous(), ape.contiguous(), out, scale, width, pool,
            mode == 'extend', round_scale, width.bit_length() - 1, num_warps=4, enable_fp_fusion=False)
    return out, scale
