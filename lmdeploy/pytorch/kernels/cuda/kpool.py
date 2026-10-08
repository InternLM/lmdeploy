# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _update_kpool_kernel(
    Keys, Scores, TailKeys, TailScores, StateIds, History, QLens,
    ClosedKeys, ClosedScores, GroupIds, Valid,
    BATCH: tl.constexpr, STEPS: tl.constexpr, STATES: tl.constexpr, RING: tl.constexpr,
    POOL: tl.constexpr, WIDTH: tl.constexpr, BLOCK_D: tl.constexpr, FROM_LENGTHS: tl.constexpr,
    stride_kb: tl.constexpr, stride_kt: tl.constexpr, stride_kd: tl.constexpr,
    stride_sb: tl.constexpr, stride_st: tl.constexpr, stride_sd: tl.constexpr,
    stride_tkb: tl.constexpr, stride_tkr: tl.constexpr,
    stride_tsb: tl.constexpr, stride_tsr: tl.constexpr,
):
    request = tl.program_id(0)
    state_id = tl.load(StateIds + request).to(tl.int64)
    valid = (state_id >= 0) & (state_id < STATES)
    history = tl.load(History + request).to(tl.int64)
    if FROM_LENGTHS:
        history -= tl.load(QLens + request).to(tl.int64)
    history = tl.maximum(history, 0)
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
        close = position == POOL - 1
        tl.store(ClosedKeys + offset, tail_k, valid & close & (d < WIDTH))
        tl.store(ClosedScores + offset, tail_s, valid & close & (d < WIDTH))
        tl.store(GroupIds + row, (history + step) // POOL)
        tl.store(Valid + row, valid & close)
        tail_k = tl.where(close, 0, tail_k)
        tail_s = tl.where(close, 0, tail_s)
        checkpoint = (history + step + 1) % RING
        tl.store(TailKeys + state_id * stride_tkb + checkpoint * stride_tkr + slot * WIDTH + d, tail_k, mask)
        tl.store(TailScores + state_id * stride_tsb + checkpoint * stride_tsr + slot * WIDTH + d, tail_s, mask)


def update_kpool(keys, scores, tail_keys, tail_scores, state_ids, history_lengths, pool_size,
                 *, q_seqlens=None, kv_seqlens=None):
    """Update decode/verify tails in place, emitting step-major closed pools.

    Inputs use [batch, steps, width]. State uses [states, pool, width] for AR or [states, ring, pool, width] for
    verification. Live state ids must be unique; invalid ids never write state. All output shapes depend only on input
    shapes. If history_lengths is None, query/KV lengths are required and history subtraction is fused into the kernel.
    """
    if keys.ndim != 3 or scores.shape != keys.shape:
        raise ValueError('Expected matching [batch, steps, width] keys and scores.')
    batch, steps, width = keys.shape
    if not batch or not steps or pool_size <= 1 or pool_size & (pool_size - 1):
        raise ValueError('Expected a nonempty batch/sequence and a power-of-two pool size.')
    from_lengths = history_lengths is None
    if from_lengths:
        if q_seqlens is None or kv_seqlens is None or q_seqlens.shape != (batch,) or kv_seqlens.shape != (batch,):
            raise ValueError('Expected one query/KV length per request when history is not precomputed.')
        history_lengths = kv_seqlens
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
        q_seqlens.contiguous() if from_lengths else None,
        closed_keys, closed_scores, groups, valid,
        batch, steps, tail_keys.size(0), tail_keys.size(1), pool_size, width,
        triton.next_power_of_2(width), from_lengths,
        *keys.stride(), *scores.stride(), *tail_keys.stride()[:2], *tail_scores.stride()[:2], num_warps=4)
    return closed_keys, closed_scores, groups, valid


@triton.jit
def _update_raw_kpool_kernel(
    Keys, Scores, RawKeys, RawScores, StateIds, QLens, KVLens, Starts,
    ClosedKeys, ClosedScores, GroupIds, Valid,
    BATCH: tl.constexpr, STEPS: tl.constexpr, STATES: tl.constexpr, CAPACITY: tl.constexpr,
    POOL: tl.constexpr, WIDTH: tl.constexpr, BLOCK_D: tl.constexpr,
    stride_kt: tl.constexpr, stride_kd: tl.constexpr,
    stride_st: tl.constexpr, stride_sd: tl.constexpr,
    stride_rkb: tl.constexpr, stride_rsb: tl.constexpr,
):
    request = tl.program_id(0)
    state = tl.load(StateIds + request).to(tl.int64)
    length = tl.load(QLens + request).to(tl.int64)
    history = tl.load(KVLens + request).to(tl.int64) - length
    start = tl.load(Starts + request).to(tl.int64)
    live = (state >= 0) & (state < STATES)
    slot = tl.arange(0, POOL)[:, None]
    feature = tl.arange(0, BLOCK_D)[None, :]
    tail_length = history % POOL
    position = (history - tail_length + slot) % CAPACITY
    mask = live & (slot < tail_length) & (feature < WIDTH)
    tail_keys = tl.load(RawKeys + state * stride_rkb + position * WIDTH + feature, mask, other=0)
    tail_scores = tl.load(RawScores + state * stride_rsb + position * WIDTH + feature, mask, other=0)
    for step in range(STEPS):
        active = live & (step < length)
        key = tl.load(Keys + (start + step) * stride_kt + feature * stride_kd,
                      active & (feature < WIDTH), other=0)
        score = tl.load(Scores + (start + step) * stride_st + feature * stride_sd,
                        active & (feature < WIDTH), other=0)
        insert = (history + step) % POOL
        tail_keys = tl.where(slot == insert, key, tail_keys)
        tail_scores = tl.where(slot == insert, score, tail_scores)
        row = step * BATCH + request
        close = active & (insert == POOL - 1)
        output = (row * POOL + slot) * WIDTH + feature
        tl.store(ClosedKeys + output, tail_keys, close & (feature < WIDTH))
        tl.store(ClosedScores + output, tail_scores, close & (feature < WIDTH))
        tl.store(GroupIds + row, (history + step) // POOL)
        tl.store(Valid + row, close)
        raw_position = (history + step) % CAPACITY
        tl.store(RawKeys + state * stride_rkb + raw_position * WIDTH + feature,
                 key, active & (feature < WIDTH))
        tl.store(RawScores + state * stride_rsb + raw_position * WIDTH + feature,
                 score, active & (feature < WIDTH))


def update_raw_kpool(keys, scores, raw_keys, raw_scores, state_ids, q_seqlens, kv_seqlens,
                     cu_seqlens_q, pool_size):
    """Assemble decode pools and persist raw tokens without materializing scratch tails.

    Live state rows and writable cache owners must be request-private. Input rows are packed by cu_seqlens_q;
    graph padding may have shorter query lengths. Capacity retains every trial token plus the accepted-history tail,
    so rejection can reconstruct any accepted prefix from the raw ring. Closed pools use step-major ordering.
    """
    batch = state_ids.numel()
    if keys.ndim != 2 or scores.shape != keys.shape or not batch or keys.size(0) % batch:
        raise ValueError('Expected matching flattened decode keys/scores and a nonempty uniform batch capacity.')
    steps, width = keys.size(0) // batch, keys.size(1)
    if not steps or pool_size <= 1 or pool_size & (pool_size - 1):
        raise ValueError('Expected positive decode capacity and a power-of-two pool size.')
    if raw_keys.ndim != 3 or raw_scores.shape != raw_keys.shape or raw_keys.size(2) != width:
        raise ValueError('Expected matching [states, capacity, width] raw rings.')
    if raw_keys.size(1) < steps + pool_size - 1:
        raise ValueError('Raw ring capacity must retain the trial tokens and the previous incomplete pool.')
    if raw_keys.stride()[1:] != (width, 1) or raw_scores.stride()[1:] != (width, 1):
        raise ValueError('Raw ring token and feature dimensions must be contiguous.')
    if keys.dtype != raw_keys.dtype or scores.dtype != raw_scores.dtype:
        raise ValueError('Projected keys/scores must match their raw ring dtypes.')
    if state_ids.shape != (batch,) or q_seqlens.shape != (batch,) or kv_seqlens.shape != (batch,):
        raise ValueError('Expected one state id and query/KV length per request.')
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() not in (batch, batch + 1):
        raise ValueError('Expected one packed input start per request.')
    tensors = (keys, scores, raw_keys, raw_scores, state_ids, q_seqlens, kv_seqlens, cu_seqlens_q)
    if not keys.is_cuda or any(value.device != keys.device for value in tensors):
        raise ValueError('Raw-ring decode requires CUDA tensors on the same device.')
    closed_keys = keys.new_empty((steps * batch, pool_size, width))
    closed_scores = scores.new_empty((steps * batch, pool_size, width))
    groups = kv_seqlens.new_empty(steps * batch)
    valid = torch.empty(steps * batch, device=keys.device, dtype=torch.bool)
    _update_raw_kpool_kernel[(batch,)](
        keys, scores, raw_keys, raw_scores, state_ids.contiguous(), q_seqlens.contiguous(),
        kv_seqlens.contiguous(), cu_seqlens_q.contiguous(), closed_keys, closed_scores, groups, valid,
        batch, steps, raw_keys.size(0), raw_keys.size(1), pool_size, width, triton.next_power_of_2(width),
        *keys.stride(), *scores.stride(), raw_keys.stride(0), raw_scores.stride(0), num_warps=4)
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
    ClosedKeys, ClosedScores, GroupIds, RequestIds, Valid,
    BATCH: tl.constexpr, STATES: tl.constexpr,
    POOL: tl.constexpr, WIDTH: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
    STRIDE_TK: tl.constexpr, STRIDE_TS: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, POOL)
    d = tl.arange(0, BLOCK_D)
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
    state_id = tl.load(StateIds + request)
    valid = valid & (state_id >= 0) & (state_id < STATES)
    prior_mask = valid & (offsets < 0)
    token_mask = valid & (offsets >= 0) & (offsets < q_len)
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
    tl.store(ClosedKeys + (row * POOL + slots[:, None]) * WIDTH + d[None, :], key, d[None, :] < WIDTH)
    tl.store(ClosedScores + (row * POOL + slots[:, None]) * WIDTH + d[None, :], score, d[None, :] < WIDTH)
    tl.store(GroupIds + row, group)
    tl.store(RequestIds + row, request)
    tl.store(Valid + row, valid)


def partition_kpool(keys, scores, tail_keys, tail_scores, state_ids, q_seqlens, kv_seqlens, pool_size):
    """Assemble only ragged closed pools without host metadata reads.

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
    if capacity == 0:
        return closed_keys, closed_scores, groups, requests, valid
    _partition_kpool_kernel[(capacity,)](
        keys.contiguous(), scores.contiguous(), tail_keys, tail_scores,
        state_ids.contiguous(), q_seqlens.contiguous(), kv_seqlens.contiguous(),
        closed_keys, closed_scores, groups, requests, valid,
        batch, tail_keys.size(0), pool_size, width, triton.next_power_of_2(batch),
        triton.next_power_of_2(width), tail_keys.stride(0), tail_scores.stride(0), num_warps=4)
    return closed_keys, closed_scores, groups, requests, valid


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
def _compress_kpool_kernel(K, S, A, O, Scale, Valid, HAS_VALID: tl.constexpr, WIDTH: tl.constexpr, POOL: tl.constexpr,
                    ONLINE: tl.constexpr, ROUND: tl.constexpr, LEVELS: tl.constexpr):
    row = tl.program_id(0)
    if HAS_VALID:
        if not tl.load(Valid + row):
            return
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
                   *, mode: str, round_scale: bool,
                   valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse weighted pooling, BF16 Hadamard rotation and one-block FP8
    quantization.

    The online/two-pass reduction order and both BF16 round trips match the reference KPool operation. Precise libdevice
    functions and disabled FMA fusion preserve its FP32 arithmetic boundaries. Invalid rows are left unwritten and must
    be masked by the cache writer.
    """
    if mode not in ('extend', 'decode'):
        raise ValueError(f'Unsupported pool compression mode: {mode}')
    if keys.ndim != 3 or scores.shape != keys.shape or ape.shape != keys.shape[1:]:
        raise ValueError('Expected matching [groups, pool, width] keys/scores and [pool, width] APE.')
    groups, pool, width = keys.shape
    if valid is not None and (valid.shape != (groups,) or valid.dtype != torch.bool or valid.device != keys.device):
        raise ValueError('Expected a boolean validity mask with one entry per pool on the same device.')
    if valid is not None:
        valid = valid.contiguous()
    if width <= 0 or width & (width - 1):
        raise ValueError('Pool width must be a positive power of two.')
    out = torch.empty(groups, width, device=keys.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(groups, 1, device=keys.device, dtype=torch.float32)
    if groups:
        _compress_kpool_kernel[(groups,)](
            keys.contiguous(), scores.contiguous(), ape.contiguous(), out, scale, valid, valid is not None, width, pool,
            mode == 'extend', round_scale, width.bit_length() - 1, num_warps=4, enable_fp_fusion=False)
    return out, scale


@triton.jit
def _read_tail(K, S, Ids, History, OutK, OutS,
               KS: tl.constexpr, SS: tl.constexpr, CAP: tl.constexpr,
               DIM: tl.constexpr, POOL: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    sid = tl.load(Ids + b)
    h = tl.load(History + b)
    slot = tl.arange(0, POOL)
    d = tl.arange(0, D)
    n = h % POOL
    pos = (h - n + slot) % CAP
    valid = (sid >= 0) & (slot[:, None] < n) & (d[None, :] < DIM)
    k = tl.load(K + sid * KS + pos[:, None] * DIM + d[None, :], valid, 0)
    s = tl.load(S + sid * SS + pos[:, None] * DIM + d[None, :], valid, 0)
    out = b * POOL * DIM + slot[:, None] * DIM + d[None, :]
    tl.store(OutK + out, k, d[None, :] < DIM)
    tl.store(OutS + out, s, d[None, :] < DIM)


@triton.jit
def _write_ring(K, S, NewK, NewS, Ids, History, Lengths, Starts,
                KS: tl.constexpr, SS: tl.constexpr,
                NK: tl.constexpr, NS: tl.constexpr, CAP: tl.constexpr,
                DIM: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    sid = tl.load(Ids + b)
    h = tl.load(History + b)
    q = tl.load(Lengths + b)
    start = tl.load(Starts + b)
    i = tl.arange(0, C)
    d = tl.arange(0, D)
    # One writer per ring cell even when a prefill chunk exceeds capacity.
    local = tl.maximum(q - CAP, 0) + i
    valid = (sid >= 0) & (i[:, None] < CAP) & (local[:, None] < q) & (d[None, :] < DIM)
    k = tl.load(NewK + (start + local[:, None]) * NK + d[None, :], valid, 0)
    s = tl.load(NewS + (start + local[:, None]) * NS + d[None, :], valid, 0)
    pos = (h + local) % CAP
    tl.store(K + sid * KS + pos[:, None] * DIM + d[None, :], k, valid)
    tl.store(S + sid * SS + pos[:, None] * DIM + d[None, :], s, valid)


def read_raw_tail(keys, scores, state_ids, history, pool_size=4):
    """Read the incomplete pool on CUDA at accepted history, never trial end.

    Capacity must retain Q new rows plus the previous pool_size-1 rows. Slot
    strides may include other layers; only the token/dimension axes are dense.
    Invalid graph rows produce zero scratch and never touch a live state.
    """
    if not keys.is_cuda:
        raise ValueError('read_raw_tail requires CUDA tensors.')
    shape = (state_ids.numel(), pool_size, keys.size(-1))
    out_k, out_s = keys.new_empty(shape), scores.new_empty(shape)
    _read_tail[(shape[0],)](
        keys, scores, state_ids, history, out_k, out_s,
        keys.stride(0), scores.stride(0), keys.size(1), keys.size(2),
        pool_size, triton.next_power_of_2(keys.size(2)))
    return out_k, out_s


def write_raw_ring(keys, scores, new_keys, new_scores, state_ids, history, lengths, starts):
    """Persist the newest capacity rows on CUDA, after all old-tail consumers.

    Prefill compresses the entire chunk separately. Rejected proposals may
    remain in storage but cannot be read beyond the subsequent accepted history.
    """
    if not keys.is_cuda:
        raise ValueError('write_raw_ring requires CUDA tensors.')
    cap, dim = keys.shape[1:]
    _write_ring[(state_ids.numel(),)](
        keys, scores, new_keys, new_scores, state_ids, history, lengths, starts,
        keys.stride(0), scores.stride(0), new_keys.stride(0), new_scores.stride(0),
        cap, dim, triton.next_power_of_2(cap), triton.next_power_of_2(dim))


@triton.jit
def _score_pages(Q, W, Cache, Lengths, Table, Out,
                 QS0: tl.constexpr, QS1: tl.constexpr,
                 WS0: tl.constexpr, TS0: tl.constexpr,
                 PAGE_STRIDE: tl.constexpr, PAGE: tl.constexpr,
                 HEADS: tl.constexpr, DIM: tl.constexpr, WIDTH: tl.constexpr,
                 H: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    groups = tl.program_id(1) * TILE + tl.arange(0, TILE)
    heads = tl.arange(0, H)
    ds = tl.arange(0, D)
    length = tl.load(Lengths + row)
    valid = (groups < length) & (groups < WIDTH)
    pages = tl.load(Table + row * TS0 + groups // PAGE, valid, 0)
    slots = groups % PAGE
    ptr = Cache + pages[None, :] * PAGE_STRIDE + slots[None, :] * DIM + ds[:, None]
    k = tl.load(ptr, valid[None, :] & (ds[:, None] < DIM), 0).to(tl.float8e4nv, bitcast=True)
    q = tl.load(Q + row * QS0 + heads[:, None] * QS1 + ds[None, :],
                (heads[:, None] < HEADS) & (ds[None, :] < DIM), 0.)
    dot = tl.dot(q, k, max_num_imprecise_acc=0)
    weight = tl.load(W + row * WS0 + heads, heads < HEADS, 0)
    logits = tl.sum(tl.maximum(dot, 0.) * weight[:, None], axis=0)
    scale_ptr = (Cache + pages * PAGE_STRIDE + PAGE * DIM + slots * 4).to(tl.pointer_type(tl.float32))
    scale = tl.load(scale_ptr, valid, 0)
    logits = tl.where(valid, logits * scale, -float('inf'))
    tl.store(Out + row * WIDTH + groups, logits, groups < WIDTH)


def score_paged(query, weight, cache, lengths, table):
    """Gather four compact 16-entry pages into each 64-key compute tile.

    No global repacking buffer or device-to-host length read. Query strides
    and fragmented physical page IDs are independent of storage geometry.
    """
    rows, heads, dim = query.shape
    width = table.size(1) * cache.size(1)
    result = torch.empty((rows, width), dtype=torch.float32, device=query.device)
    if not rows or not width:
        return result
    _score_pages[(rows, triton.cdiv(width, 64))](
        query, weight, cache, lengths, table, result,
        query.stride(0), query.stride(1), weight.stride(0), table.stride(0),
        cache.stride(0), cache.size(1), heads, dim, width,
        max(16, triton.next_power_of_2(heads)), triton.next_power_of_2(dim), 64)
    return result


@triton.jit
def _gather_token_tail_kernel(Cache, Blocks, Q, KV, Keys, Scores, StateIds,
                              stride_cb: tl.constexpr, stride_ct: tl.constexpr,
                              stride_cs: tl.constexpr, stride_cd: tl.constexpr,
                              stride_bb: tl.constexpr, stride_bp: tl.constexpr,
                              PAGES: tl.constexpr, PAGE_SIZE: tl.constexpr,
                              POOL: tl.constexpr, WIDTH: tl.constexpr,
                              BLOCK_D: tl.constexpr):
    request = tl.program_id(0)
    history = tl.load(KV + request).to(tl.int64) - tl.load(Q + request).to(tl.int64)
    tail = history % POOL
    slot = tl.arange(0, POOL)
    position = history - tail + slot
    page = position // PAGE_SIZE
    valid = (slot < tail) & (position >= 0) & (page < PAGES)
    block = tl.load(Blocks + request * stride_bb + page * stride_bp, valid, other=0)
    dim = tl.arange(0, BLOCK_D)
    ptr = Cache + block[:, None] * stride_cb + (position % PAGE_SIZE)[:, None] * stride_ct
    ptr += dim[None, :] * stride_cd
    mask = valid[:, None] & (dim[None, :] < WIDTH)
    keys = tl.load(ptr, mask, other=0)
    scores = tl.load(ptr + stride_cs, mask, other=0)
    out = (request * POOL + slot[:, None]) * WIDTH + dim[None, :]
    tl.store(Keys + out, keys, dim[None, :] < WIDTH)
    tl.store(Scores + out, scores, dim[None, :] < WIDTH)
    tl.store(StateIds + request, request)


def gather_kpool_token_tail(cache, block_offsets, q_seqlens, kv_seqlens, pool_size):
    """Reconstruct private draft tails from pageable accepted token history."""
    batch, width = q_seqlens.numel(), cache.size(-1)
    keys = cache.new_empty((batch, pool_size, width))
    scores = torch.empty_like(keys)
    state_ids = torch.empty(batch, device=cache.device, dtype=torch.int64)
    _gather_token_tail_kernel[(batch,)](
        cache, block_offsets, q_seqlens, kv_seqlens, keys, scores, state_ids,
        *cache.stride(), *block_offsets.stride(), block_offsets.size(1), cache.size(1),
        pool_size, width, triton.next_power_of_2(width), num_warps=4)
    return keys, scores, state_ids


@triton.jit
def _write_token_cache_kernel(Cache, Blocks, Q, KV, Starts, Keys, Scores,
                               stride_cb: tl.constexpr, stride_ct: tl.constexpr,
                               stride_cs: tl.constexpr, stride_cd: tl.constexpr,
                               stride_bb: tl.constexpr, stride_bp: tl.constexpr,
                               stride_kt: tl.constexpr, stride_kd: tl.constexpr,
                               stride_st: tl.constexpr, stride_sd: tl.constexpr,
                               BATCH: tl.constexpr, PAGES: tl.constexpr,
                               PAGE_SIZE: tl.constexpr, WIDTH: tl.constexpr,
                               BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    request_ids = tl.arange(0, BLOCK_B)
    starts = tl.load(Starts + request_ids, request_ids < BATCH, other=2147483647)
    request = tl.sum((row >= starts).to(tl.int32), 0) - 1
    history = tl.load(KV + request).to(tl.int64) - tl.load(Q + request).to(tl.int64)
    position = history + row - tl.load(Starts + request)
    page = position // PAGE_SIZE
    valid = (position >= 0) & (page < PAGES)
    block = tl.load(Blocks + request * stride_bb + page * stride_bp, valid, other=0)
    dim = tl.arange(0, BLOCK_D)
    keys = tl.load(Keys + row * stride_kt + dim * stride_kd, dim < WIDTH, other=0)
    scores = tl.load(Scores + row * stride_st + dim * stride_sd, dim < WIDTH, other=0)
    ptr = Cache + block * stride_cb + (position % PAGE_SIZE) * stride_ct + dim * stride_cd
    tl.store(ptr, keys, valid & (dim < WIDTH))
    tl.store(ptr + stride_cs, scores, valid & (dim < WIDTH))


def write_kpool_token_cache(cache, keys, scores, block_offsets, q_seqlens,
                            kv_seqlens, cu_seqlens_q):
    """Write raw draft projections without materializing token index
    tensors."""
    batch, width = q_seqlens.numel(), cache.size(-1)
    _write_token_cache_kernel[(keys.size(0),)](
        cache, block_offsets, q_seqlens, kv_seqlens, cu_seqlens_q, keys, scores,
        *cache.stride(), *block_offsets.stride(), *keys.stride(), *scores.stride(),
        batch, block_offsets.size(1), cache.size(1), width,
        triton.next_power_of_2(batch), triton.next_power_of_2(width), num_warps=4)


@triton.jit
def _decode_metadata_kernel(Q, KV, Blocks, Seq, Groups, Context, ScheduleLengths, Table,
                            STEPS: tl.constexpr, POOL: tl.constexpr, PAGES: tl.constexpr, PAGE_STEP: tl.constexpr,
                            stride_bb: tl.constexpr, stride_bp: tl.constexpr,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    request, step = row // STEPS, row % STEPS
    length = tl.load(KV + request).to(tl.int64) - tl.load(Q + request).to(tl.int64) + step + 1
    # Triton integer division truncates; match Torch floor division for padded rows.
    groups = (length - tl.where(length < 0, POOL - 1, 0)) // POOL
    if tile == 0:
        tl.store(Seq + row, length)
        tl.store(Groups + row, groups)
        tl.store(Context + row, groups)
        tl.store(ScheduleLengths + row, tl.maximum(groups, 1))
    pages = tile * BLOCK + tl.arange(0, BLOCK)
    block = tl.load(Blocks + request * stride_bb + pages * PAGE_STEP * stride_bp,
                    pages < PAGES, other=0)
    tl.store(Table + row * PAGES + pages, block, pages < PAGES)


def prepare_kpool_decode_metadata(q_seqlens, kv_seqlens, block_offsets, rows,
                                   pool_size, with_scores=True, *, page_step=None):
    """Fill graph-owned sequence and pooled-page metadata once per forward."""
    batch = q_seqlens.numel()
    if not batch or rows % batch:
        raise ValueError('Decode rows must be a multiple of the request count.')
    if page_step is None:
        page_step = pool_size
    if page_step <= 0:
        raise ValueError('Pooled page stride must be positive.')
    pages = triton.cdiv(block_offsets.size(1), page_step) if with_scores else 0
    seq = torch.empty(rows, device=q_seqlens.device, dtype=torch.int64)
    groups = torch.empty_like(seq)
    context = torch.empty((rows, 1), device=q_seqlens.device, dtype=torch.int32)
    schedule_lengths = torch.empty_like(context)
    table = torch.empty((rows, pages), device=q_seqlens.device, dtype=torch.int32)
    _decode_metadata_kernel[(rows, max(1, triton.cdiv(pages, 256)))](
        q_seqlens, kv_seqlens, block_offsets, seq, groups, context, schedule_lengths,
        table, rows // batch, pool_size, pages, page_step, *block_offsets.stride(), 256, num_warps=4)
    return seq, groups, context, schedule_lengths, table
