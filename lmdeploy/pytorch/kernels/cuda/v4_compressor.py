# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _fill_compress_state_kernel(
    kv_ptr,
    score_ptr,
    ape_ptr,
    kv_state_ptr,
    score_state_ptr,
    state_idx_ptr,
    cu_q_seqlens_ptr,
    kv_seqlens_ptr,
    kv_stride_s,
    kv_stride_d: tl.constexpr,
    score_stride_s,
    score_stride_d: tl.constexpr,
    ape_stride_r: tl.constexpr,
    ape_stride_d: tl.constexpr,
    kvc_stride_n,
    kvc_stride_r: tl.constexpr,
    kvc_stride_d: tl.constexpr,
    scorec_stride_n,
    scorec_stride_r: tl.constexpr,
    scorec_stride_d: tl.constexpr,
    D: tl.constexpr,
    ratio: tl.constexpr,
    overlap: tl.constexpr,
):
    """Fill kv_state and score_state for the Compressor ring buffer.

    Called BEFORE score_kv (score_kv reads state, this writes state).

    Grid: (max_write, B) for prefill, (1, B) for decode.
    Each CTA handles one token position within a batch sequence.
    Only the last max_write tokens per sequence are stored.

    Ring buffer layout (overlap=True, ratio=4):
      State shape [N, 2*ratio, D] where D = coff * head_dim = 2*head_dim.
      Write position for abs_pos p: (p + ratio) % (2*ratio).
      This maps:
        abs_pos [0, ratio)     -> rows [ratio, 2*ratio)  (curr window)
        abs_pos [ratio, 2*ratio) -> rows [0, ratio)      (prev window)
      The read head in score_kv is: (start_pos // ratio % 2) * ratio.

    Ring buffer layout (overlap=False, ratio=128):
      State shape [N, ratio, D] where D = head_dim.
      Write position: abs_pos % ratio. Simple circular buffer.

    Stored values (full D dimensions):
      kv_state[sid, row]    = kv[kv_pos]
      score_state[sid, row] = score[kv_pos] + ape[abs_pos % ratio]
    """
    t_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    seq_start = tl.load(cu_q_seqlens_ptr + batch_id)
    seq_end = tl.load(cu_q_seqlens_ptr + batch_id + 1)
    seqlen = seq_end - seq_start
    kvlen = tl.load(kv_seqlens_ptr + batch_id)
    state_id = tl.load(state_idx_ptr + batch_id)

    if overlap:
        max_num_write = ratio * 2
    else:
        max_num_write = ratio
    t_size = min(seqlen, max_num_write)

    if t_id >= t_size:
        return

    kv_pos = seq_start + seqlen - t_size + t_id
    abs_pos = kvlen - t_size + t_id

    offs_d = tl.arange(0, D)
    kv_ptrs = kv_ptr + kv_pos * kv_stride_s + offs_d * kv_stride_d
    score_ptrs = score_ptr + kv_pos * score_stride_s + offs_d * score_stride_d
    ape_ptrs = ape_ptr + (abs_pos % ratio) * ape_stride_r + offs_d * ape_stride_d
    if overlap:
        state_row = (abs_pos + ratio) % max_num_write
    else:
        state_row = abs_pos % max_num_write
    kv_state_ptrs = kv_state_ptr + state_id * kvc_stride_n + state_row * kvc_stride_r + offs_d * kvc_stride_d
    score_state_ptrs = (score_state_ptr + state_id * scorec_stride_n + state_row * scorec_stride_r +
                        offs_d * scorec_stride_d)

    kv = tl.load(kv_ptrs)
    score = tl.load(score_ptrs)
    ape = tl.load(ape_ptrs)

    tl.store(kv_state_ptrs, kv)
    tl.store(score_state_ptrs, score + ape)


@triton.jit
def _score_kv_kernel(
    kv_ptr,
    score_ptr,
    ape_ptr,
    kv_state_ptr,
    score_state_ptr,
    state_idx_ptr,
    cu_q_seqlens_ptr,
    kv_seqlens_ptr,
    ckv_ptr,
    kv_stride_s,
    kv_stride_d: tl.constexpr,
    score_stride_s,
    score_stride_d: tl.constexpr,
    ape_stride_r: tl.constexpr,
    ape_stride_d: tl.constexpr,
    kvc_stride_n,
    kvc_stride_r: tl.constexpr,
    kvc_stride_d: tl.constexpr,
    scorec_stride_n,
    scorec_stride_r: tl.constexpr,
    scorec_stride_d: tl.constexpr,
    ckv_stride_s,
    ckv_stride_d: tl.constexpr,
    head_dim: tl.constexpr,
    ratio: tl.constexpr,
    overlap: tl.constexpr,
    is_decoding: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute compressed kv via softmax-weighted sum, write back to
    kv[:head_dim].

    Compression points are at abs_pos = n*ratio - 1 (0-indexed).
    For ratio=4, this means kv positions 3, 7, 11, 15, ...

    == DECODE PATH (is_decoding=True, seqlen=1) ==
    Grid: (1, B). One CTA per batch.
    - Emit condition: (start_pos + 1) % ratio == 0.
      Only writes back when emit=True (static graph, no branch).
    - Loads current token's kv/score from kv_ptr/score_ptr.
      overlap:  reads RIGHT half (kv[..., head_dim:])
      no-overlap: reads LEFT half (kv[..., :head_dim])
    - Loads state rows as [ratio, BLOCK_D] blocks.
      overlap:  prev from rows [head..head+ratio) LEFT half,
                curr from rows [head+ratio..head+2*ratio) RIGHT half.
                Manual softmax (Triton tl.cat only supports 1D).
      no-overlap: single [ratio, BLOCK_D] block, tl.softmax.
    - Replaces current token's slot in the loaded block via tl.where.
    - Write-back: kv[seq_start, :head_dim] (masked by emit).

    == PREFILL PATH (is_decoding=False) ==
    Grid: (num_groups, B). One CTA per compression point per batch.
    - Compression points aligned to abs_pos:
        first_compress = ceil((start_pos+1) / ratio) * ratio - 1
      = ((start_pos + ratio) // ratio) * ratio - 1
    - Each CTA handles compress_abs = first_compress + group_id * ratio.
    - Write-back position: kv[seq_start + (compress_abs - start_pos), :head_dim].

    Prefill overlap data windows per compression point at compress_abs:
      prev window: abs_pos in [compress_abs - 2*ratio + 1, compress_abs - ratio]
        - If prev_abs_base < 0:        zeros / -1e30 (no history)
        - If prev_abs_base < start_pos: read from state (ring buffer rows)
        - Else:                         read from kv LEFT half + score + ape
      curr window: abs_pos in [compress_abs - ratio + 1, compress_abs]
        - Always in new tokens: read from kv RIGHT half + score + ape
      Manual softmax over [2*ratio] (prev+curr as separate blocks).

    Prefill non-overlap data windows:
      abs_pos in [compress_abs - ratio + 1, compress_abs]
        - Read from kv + score + ape, tl.softmax over [ratio].

    == Ring buffer read logic (overlap decode) ==
      head = (start_pos // ratio % 2) * ratio
      cap  = 2 * ratio
      prev rows: (head + tl.arange(0, ratio)) % cap
      curr rows: (head + ratio + tl.arange(0, ratio)) % cap
    This matches fill_compress_state's write positions where
    abs_pos p is stored at row (p + ratio) % cap.
    """
    group_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    seq_start = tl.load(cu_q_seqlens_ptr + batch_id)
    seq_end = tl.load(cu_q_seqlens_ptr + batch_id + 1)
    seqlen = seq_end - seq_start
    kvlen = tl.load(kv_seqlens_ptr + batch_id)
    state_id = tl.load(state_idx_ptr + batch_id)
    start_pos = kvlen - seqlen

    if is_decoding:
        emit = (start_pos + 1) % ratio == 0
    else:
        first_compress = ((start_pos + ratio) // ratio) * ratio - 1
        last_pos = start_pos + seqlen - 1
        num_groups = 0
        if first_compress <= last_pos:
            num_groups = ((last_pos - first_compress) // ratio + 1).to(num_groups.dtype)
        if group_id >= num_groups:
            return

    if overlap:
        cap = ratio * 2

    row_ids = tl.arange(0, ratio)

    n_tiles = head_dim // BLOCK_D
    for d_tile in range(n_tiles):
        d_off = d_tile * BLOCK_D
        offs_d = d_off + tl.arange(0, BLOCK_D)

        if is_decoding:
            # ======== DECODE PATH ========
            pos_in_ratio = start_pos % ratio
            if overlap:
                cur_kv = tl.load(
                    kv_ptr + seq_start * kv_stride_s + (head_dim + d_off + offs_d) * kv_stride_d).to(
                        tl.float32)
                cur_score = tl.load(
                    score_ptr + seq_start * score_stride_s + (head_dim + d_off + offs_d) *
                    score_stride_d).to(tl.float32)
                cur_ape = tl.load(
                    ape_ptr + pos_in_ratio * ape_stride_r + (head_dim + d_off + offs_d) *
                    ape_stride_d).to(tl.float32)
            else:
                cur_kv = tl.load(kv_ptr + seq_start * kv_stride_s + offs_d * kv_stride_d).to(tl.float32)
                cur_score = tl.load(score_ptr + seq_start * score_stride_s + offs_d * score_stride_d).to(tl.float32)
                cur_ape = tl.load(
                    ape_ptr + pos_in_ratio * ape_stride_r + (d_off + offs_d) * ape_stride_d).to(
                        tl.float32)
            cur_score = cur_score + cur_ape

            if overlap:
                head = (start_pos // ratio % 2) * ratio
                # prev window from state (LEFT half: :head_dim)
                prev_row_ids = (head + row_ids) % cap
                prev_kv = tl.load(
                    kv_state_ptr + state_id * kvc_stride_n + prev_row_ids[:, None] * kvc_stride_r +
                    offs_d[None, :] * kvc_stride_d).to(tl.float32)
                prev_score = tl.load(
                    score_state_ptr + state_id * scorec_stride_n +
                    prev_row_ids[:, None] * scorec_stride_r +
                    offs_d[None, :] * scorec_stride_d).to(tl.float32)

                # curr window from state (RIGHT half: head_dim:)
                curr_row_ids = (head + ratio + row_ids) % cap
                curr_kv = tl.load(
                    kv_state_ptr + state_id * kvc_stride_n + curr_row_ids[:, None] * kvc_stride_r +
                    (head_dim + d_off + offs_d[None, :]) * kvc_stride_d).to(tl.float32)
                curr_score = tl.load(
                    score_state_ptr + state_id * scorec_stride_n +
                    curr_row_ids[:, None] * scorec_stride_r +
                    (head_dim + d_off + offs_d[None, :]) * scorec_stride_d).to(tl.float32)

                # replace current token's slot in curr window
                row_mask = row_ids == pos_in_ratio
                curr_kv = tl.where(row_mask[:, None], cur_kv[None, :], curr_kv)
                curr_score = tl.where(row_mask[:, None], cur_score[None, :], curr_score)

                # mask unwritten state rows: rows beyond pos_in_ratio haven't been
                # filled by fill_compress_state yet (score_state defaults to 0, not -inf).
                # For prev window: valid only when start_pos >= ratio AND row_id <= pos_in_ratio
                curr_valid = row_ids <= pos_in_ratio
                prev_valid = (start_pos >= ratio) & curr_valid
                NEG_INF = -1e30
                prev_score = tl.where(prev_valid[:, None], prev_score, NEG_INF)
                curr_score = tl.where(curr_valid[:, None], curr_score, NEG_INF)

                # manual softmax over [2*ratio]
                global_max = tl.maximum(tl.max(prev_score, 0), tl.max(curr_score, 0))
                exp_prev = tl.exp(prev_score - global_max[None, :])
                exp_curr = tl.exp(curr_score - global_max[None, :])
                sum_exp = tl.sum(exp_prev, 0) + tl.sum(exp_curr, 0)
                compressed = (tl.sum(prev_kv * exp_prev, 0) + tl.sum(curr_kv * exp_curr, 0)) / sum_exp
            else:
                # non-overlap: load [ratio, BLOCK_D] from state
                merged_kv = tl.load(
                    kv_state_ptr + state_id * kvc_stride_n + row_ids[:, None] * kvc_stride_r +
                    offs_d[None, :] * kvc_stride_d).to(tl.float32)
                merged_score = tl.load(
                    score_state_ptr + state_id * scorec_stride_n +
                    row_ids[:, None] * scorec_stride_r +
                    offs_d[None, :] * scorec_stride_d).to(tl.float32)
                # replace current token's slot
                row_mask = row_ids == pos_in_ratio
                merged_kv = tl.where(row_mask[:, None], cur_kv[None, :], merged_kv)
                merged_score = tl.where(row_mask[:, None], cur_score[None, :], merged_score)
                # mask unwritten state rows (score_state defaults to 0, not -inf)
                merged_score = tl.where((row_ids <= pos_in_ratio)[:, None], merged_score, -1e30)

                soft_score = tl.softmax(merged_score, 0)
                compressed = tl.sum(merged_kv * soft_score, 0)

            out_ptrs = ckv_ptr + seq_start * ckv_stride_s + offs_d * ckv_stride_d
            tl.store(out_ptrs, compressed, mask=emit)

        else:
            # ======== PREFILL PATH ========
            compress_abs = first_compress + group_id * ratio
            write_pos = seq_start + (compress_abs - start_pos)

            if overlap:
                prev_abs_base = compress_abs - 2 * ratio + 1
                curr_abs_base = compress_abs - ratio + 1

                if prev_abs_base < 0:
                    prev_kv = tl.zeros((ratio, BLOCK_D), dtype=tl.float32)
                    prev_score = tl.full((ratio, BLOCK_D), -1e30, dtype=tl.float32)
                elif prev_abs_base < start_pos:
                    _prev_head = (prev_abs_base // ratio % 2) * ratio
                    _prev_row_ids = (_prev_head + row_ids) % cap
                    prev_kv = tl.load(
                        kv_state_ptr + state_id * kvc_stride_n +
                        _prev_row_ids[:, None] * kvc_stride_r +
                        offs_d[None, :] * kvc_stride_d).to(tl.float32)
                    prev_score = tl.load(
                        score_state_ptr + state_id * scorec_stride_n +
                        _prev_row_ids[:, None] * scorec_stride_r +
                        offs_d[None, :] * scorec_stride_d).to(tl.float32)
                else:
                    _prev_kv_pos = seq_start + (prev_abs_base - start_pos)
                    _prev_abs_pos = prev_abs_base + row_ids
                    prev_kv = tl.load(
                        kv_ptr + (_prev_kv_pos + row_ids[:, None]) * kv_stride_s +
                        offs_d[None, :] * kv_stride_d).to(tl.float32)
                    _prev_score_raw = tl.load(
                        score_ptr + (_prev_kv_pos + row_ids[:, None]) * score_stride_s +
                        offs_d[None, :] * score_stride_d).to(tl.float32)
                    _prev_ape = tl.load(
                        ape_ptr + (_prev_abs_pos % ratio)[:, None] * ape_stride_r +
                        offs_d[None, :] * ape_stride_d).to(tl.float32)
                    prev_score = _prev_score_raw + _prev_ape

                _curr_kv_pos = seq_start + (curr_abs_base - start_pos)
                _curr_abs_pos = curr_abs_base + row_ids
                curr_kv = tl.load(
                    kv_ptr + (_curr_kv_pos + row_ids[:, None]) * kv_stride_s +
                    (head_dim + d_off + offs_d[None, :]) * kv_stride_d).to(tl.float32)
                _curr_score_raw = tl.load(
                    score_ptr + (_curr_kv_pos + row_ids[:, None]) * score_stride_s +
                    (head_dim + d_off + offs_d[None, :]) * score_stride_d).to(tl.float32)
                _curr_ape = tl.load(
                    ape_ptr + (_curr_abs_pos % ratio)[:, None] * ape_stride_r +
                    (head_dim + d_off + offs_d[None, :]) * ape_stride_d).to(tl.float32)
                curr_score = _curr_score_raw + _curr_ape

                global_max = tl.maximum(tl.max(prev_score, 0), tl.max(curr_score, 0))
                exp_prev = tl.exp(prev_score - global_max[None, :])
                exp_curr = tl.exp(curr_score - global_max[None, :])
                sum_exp = tl.sum(exp_prev, 0) + tl.sum(exp_curr, 0)
                compressed = (tl.sum(prev_kv * exp_prev, 0) + tl.sum(curr_kv * exp_curr, 0)) / sum_exp
            else:
                _abs_pos_base = compress_abs - ratio + 1
                _kv_pos_base = seq_start + (_abs_pos_base - start_pos)
                merged_kv = tl.load(
                    kv_ptr + (_kv_pos_base + row_ids[:, None]) * kv_stride_s +
                    offs_d[None, :] * kv_stride_d).to(tl.float32)
                merged_score = tl.load(
                    score_ptr + (_kv_pos_base + row_ids[:, None]) * score_stride_s +
                    offs_d[None, :] * score_stride_d).to(tl.float32)
                _ape_vals = tl.load(
                    ape_ptr + ((_abs_pos_base + row_ids) % ratio)[:, None] * ape_stride_r +
                    offs_d[None, :] * ape_stride_d).to(tl.float32)
                merged_score = merged_score + _ape_vals

                soft_score = tl.softmax(merged_score, 0)
                compressed = tl.sum(merged_kv * soft_score, 0)

            out_ptrs = ckv_ptr + write_pos * ckv_stride_s + offs_d * ckv_stride_d
            tl.store(out_ptrs, compressed)


def fill_compress_state(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    state_ids: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    ):
    """Fill kv_state and score_state for the Compressor ring buffer.

    Called AFTER score_kv in the execution order: score_kv reads prev state,
    then this overwrites state with new values — no synchronization needed.

    Ring buffer layout (overlap=True, ratio=4):
      State shape [N, 2*ratio, D] where D = 2*head_dim.
      Write position for abs_pos p: (p + ratio) % (2*ratio).
      This maps:
        abs_pos [0, ratio)       -> rows [ratio, 2*ratio)  (curr window)
        abs_pos [ratio, 2*ratio) -> rows [0, ratio)        (prev window)
      The read head in score_kv is: (start_pos // ratio % 2) * ratio.

    Ring buffer layout (overlap=False, ratio=128):
      State shape [N, ratio, D] where D = head_dim.
      Write position: abs_pos % ratio. Simple circular buffer.

    Stored values (full D dimensions):
      kv_state[sid, row]    = kv[kv_pos]
      score_state[sid, row] = score[kv_pos] + ape[abs_pos % ratio]

    Only the last max_write tokens per sequence are stored; earlier tokens
    are overwritten in the ring buffer.

    Args:
        kv: [S, D] flat kv tensor, D = coff * head_dim, indexed via cu_q_seqlens.
        score: [S, D] flat score tensor.
        ape: [ratio, D] additive positional encoding.
        kv_state: [N, ratio * coff, D] kv state buffer (fp32).
        score_state: [N, ratio * coff, D] score state buffer (fp32, init with -inf).
        state_ids: [B] state index per batch.
        cu_q_seqlens: [B + 1] cumulative query sequence lengths.
        kv_seqlens: [B] total kv sequence lengths (history + new).
    """
    assert state_ids.is_contiguous()
    assert cu_q_seqlens.is_contiguous()
    assert kv_seqlens.is_contiguous()

    ratio = ape.size(0)
    D = ape.size(1)
    B = state_ids.size(0)
    coff = kv_state.size(1) // ratio
    overlap = coff > 1

    is_decoding = kv.size(0) == B
    max_write = ratio * 2 if overlap else ratio

    grid = (1 if is_decoding else max_write, B, )
    _fill_compress_state_kernel[grid](
        kv, score, ape, kv_state, score_state, state_ids, cu_q_seqlens, kv_seqlens,
        *kv.stride(),
        *score.stride(),
        *ape.stride(),
        *kv_state.stride(),
        *score_state.stride(),
        D, ratio, overlap
    )


def score_kv(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    state_ids: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    compressed_kv: torch.Tensor,
    overlap: bool,
    max_seqlen_q: int = None,
    ):
    """Compute compressed kv via softmax-weighted sum, write to compressed_kv.

    Called BEFORE fill_compress_state: reads prev state, then fill_compress_state
    overwrites state with new values.

    Compression points are at abs_pos = n*ratio - 1 (0-indexed).
    For ratio=4, this means kv positions 3, 7, 11, 15, ...
    The compressed result is written to compressed_kv[compress_pos].

    == Decode path (seqlen=1) ==
    One CTA per batch. Emit only when (start_pos + 1) % ratio == 0.
    Overlap: loads prev/curr windows from state (ring buffer rows),
      merges with current token's RIGHT-half kv/score, manual softmax.
    Non-overlap: loads [ratio, BLOCK_D] from state, tl.softmax.

    == Prefill path ==
    One CTA per compression point per batch.
    Groups aligned to abs_pos boundaries:
      first_compress = ((start_pos + ratio) // ratio) * ratio - 1
    Overlap: prev window may come from state (history) or kv LEFT half;
      curr window from kv RIGHT half. Manual softmax.
    Non-overlap: reads kv + score + ape directly, tl.softmax.

    Args:
        kv: [S, D] flat kv tensor, D = coff * head_dim. Read-only.
        score: [S, D] flat score tensor (not modified).
        ape: [ratio, D] additive positional encoding.
        kv_state: [N, ratio * coff, D] kv state buffer (fp32).
        score_state: [N, ratio * coff, D] score state buffer (fp32, init with -inf).
        state_ids: [B] state index per batch.
        cu_q_seqlens: [B + 1] cumulative query sequence lengths.
        kv_seqlens: [B] total kv sequence lengths (history + new).
        compressed_kv: [S, head_dim] output tensor for compressed results.
            Written at compression point positions.
        overlap: whether overlap mode (True when ratio=4).
        max_seqlen_q: max query seq len (used for prefill grid size).
    """
    ratio = ape.size(0)
    D = ape.size(1)
    head_dim = D // (1 + overlap)
    B = state_ids.size(0)

    is_decoding = kv.size(0) == B

    BLOCK_D = 128

    if is_decoding:
        grid = (1, B)
    else:
        num_groups = (max_seqlen_q + ratio - 1) // ratio
        grid = (num_groups, B)

    _score_kv_kernel[grid](
        kv, score, ape, kv_state, score_state, state_ids, cu_q_seqlens, kv_seqlens,
        compressed_kv,
        *kv.stride(),
        *score.stride(),
        *ape.stride(),
        *kv_state.stride(),
        *score_state.stride(),
        *compressed_kv.stride(),
        head_dim=head_dim, ratio=ratio,
        overlap=overlap, is_decoding=is_decoding,
        BLOCK_D=BLOCK_D,
    )


@triton.jit
def _fill_compressed_kv_kernel(
    ckv_ptr,
    kv_cache_ptr,
    cu_q_seqlens_ptr,
    kv_seqlens_ptr,
    block_offsets_ptr,
    ckv_stride_s,
    ckv_stride_d: tl.constexpr,
    kvc_stride_b,
    kvc_stride_s: tl.constexpr,
    kvc_stride_d: tl.constexpr,
    stride_boff0,
    stride_boff1: tl.constexpr,
    fp8_nope_rope_ptr,
    fp8nr_stride_b,
    fp8nr_stride_s: tl.constexpr,
    fp8nr_stride_d: tl.constexpr,
    fp8_rope_bf16_ptr,
    fp8rbf16_stride_b,
    fp8rbf16_stride_s: tl.constexpr,
    fp8rbf16_stride_d: tl.constexpr,
    fp8_scales_u8_ptr,
    fp8sc_stride_b,
    fp8sc_stride_s: tl.constexpr,
    fp8sc_stride_d: tl.constexpr,
    head_dim: tl.constexpr,
    compress_ratio: tl.constexpr,
    block_size: tl.constexpr,
    is_decoding: tl.constexpr,
    has_bf16: tl.constexpr,
    has_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    seq_start = tl.load(cu_q_seqlens_ptr + batch_id)
    seq_end = tl.load(cu_q_seqlens_ptr + batch_id + 1)
    seqlen = seq_end - seq_start
    kvlen = tl.load(kv_seqlens_ptr + batch_id)
    start_pos = kvlen - seqlen

    entries_per_block: tl.constexpr = block_size // compress_ratio

    if is_decoding:
        emit = (start_pos + 1) % compress_ratio == 0
        if not emit:
            return
        p = start_pos // compress_ratio
        write_pos = seq_start
    else:
        first_compress = ((start_pos + compress_ratio) // compress_ratio) * compress_ratio - 1
        last_pos = start_pos + seqlen - 1
        num_groups = 0
        if first_compress <= last_pos:
            num_groups = ((last_pos - first_compress) // compress_ratio + 1).to(tl.int32)
        if group_id >= num_groups:
            return
        compress_abs = first_compress + group_id * compress_ratio
        p = compress_abs // compress_ratio
        write_pos = seq_start + (compress_abs - start_pos)

    token_pos = p * compress_ratio
    block_idx = token_pos // block_size
    block_off = p % entries_per_block
    phys_block = tl.load(block_offsets_ptr + batch_id * stride_boff0 + block_idx * stride_boff1)

    # ---- Write to BF16 paged block cache ----
    if has_bf16:
        n_tiles = head_dim // BLOCK_D
        for d_tile in range(n_tiles):
            d_off = d_tile * BLOCK_D
            offs_d = d_off + tl.arange(0, BLOCK_D)

            compressed = tl.load(ckv_ptr + write_pos * ckv_stride_s + offs_d * ckv_stride_d)
            cache_ptrs = kv_cache_ptr + phys_block * kvc_stride_b + block_off * kvc_stride_s + offs_d * kvc_stride_d
            tl.store(cache_ptrs, compressed.to(kv_cache_ptr.dtype.element_ty))

    # ---- Write to FP8 paged block cache (MODEL1 sparse format) ----
    if has_fp8:
        # FlashMLA MODEL1 sparse FP8 layout (matches C++ kernel addressing):
        #   NoPE+RoPE region: [num_blocks, entries_per_block, 576] as e4m3fn
        #     per-token: [NoPE 448 fp8 | RoPE 128 bytes (64 bf16)]
        #     token stride = 576 bytes
        #   Scales region: [num_blocks, entries_per_block, 8] as uint8
        #     per-token: [7 e8m0 scale bytes | 1 padding]
        #     located at byte offset entries_per_block * 576 within each block
        #
        # Three pointers (all views of the same underlying fp8_cache tensor):
        #   fp8_nope_rope_ptr  — e4m3fn, stride_b/stride_s for NoPE write
        #   fp8_rope_bf16_ptr  — bfloat16 view of the RoPE region
        #   fp8_scales_u8_ptr  — uint8 view of the scales region
        D_NOPE: tl.constexpr = 448
        D_ROPE: tl.constexpr = 64
        TILE_SIZE: tl.constexpr = 64
        NUM_TILES: tl.constexpr = 7
        FP8_MAX: tl.constexpr = 448.0

        fp8_elem_ty = fp8_nope_rope_ptr.dtype.element_ty
        nope_base = fp8_nope_rope_ptr + phys_block * fp8nr_stride_b + block_off * fp8nr_stride_s

        # Quantize 7 NoPE tiles
        offs_tile = tl.arange(0, TILE_SIZE)
        for tile_idx in range(NUM_TILES):
            d_base = tile_idx * TILE_SIZE
            tile_ptrs = ckv_ptr + write_pos * ckv_stride_s + (d_base + offs_tile) * ckv_stride_d
            tile_bf16 = tl.load(tile_ptrs)
            tile_f32 = tile_bf16.to(tl.float32)

            amax = tl.max(tl.abs(tile_f32), axis=0)
            ceil_log2 = tl.math.ceil(tl.math.log2(tl.maximum(amax / FP8_MAX, 1e-4)))
            scale_inv = tl.math.exp2(ceil_log2)
            quantized = (tile_f32 / scale_inv).to(fp8_elem_ty)

            nope_ptrs = nope_base + (d_base + offs_tile) * fp8nr_stride_d
            tl.store(nope_ptrs, quantized)

            # e8m0fnu scale byte: raw byte = ceil_log2 + bias(127)
            scale_byte = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)
            sc_base = fp8_scales_u8_ptr + phys_block * fp8sc_stride_b + block_off * fp8sc_stride_s
            tl.store(sc_base + tile_idx * fp8sc_stride_d, scale_byte)

        # Copy RoPE dims: store 64 BF16 values via the bf16 view pointer.
        # fp8_rope_bf16_ptr is already sliced to the RoPE region
        # (bf16 view starting at D_NOPE/2=224 within each token's NoPE+RoPE),
        # so we write directly at offset 0..63.
        rope_offs_bf16 = tl.arange(0, D_ROPE)
        rope_ptrs = ckv_ptr + write_pos * ckv_stride_s + (D_NOPE + rope_offs_bf16) * ckv_stride_d
        rope_bf16 = tl.load(rope_ptrs)
        rope_base = fp8_rope_bf16_ptr + phys_block * fp8rbf16_stride_b + block_off * fp8rbf16_stride_s
        tl.store(rope_base + rope_offs_bf16 * fp8rbf16_stride_d, rope_bf16)


def fill_compressed_kv(
    compressed_kv: torch.Tensor,
    kv_cache: torch.Tensor | None,
    cu_q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_offsets: torch.Tensor,
    compress_ratio: int,
    block_size: int,
    max_seqlen_q: int,
    fp8_cache: torch.Tensor | None = None,
    ):
    """Write compressed KV entries from compressed_kv into paged caches.

    After score_kv produces compressed entries at compression points
    (abs_pos = n*ratio - 1), this kernel scatters those entries into the
    block-paged kv_cache used by the decode-phase sparse attention.

    When fp8_cache is provided, also writes MODEL1 sparse FP8 packed entries
    directly into fp8_cache, eliminating the need for a separate Python-side
    packing step.

    When kv_cache is None, the BF16 write is skipped (only FP8 is written).

    == Addressing scheme ==
    The kv_cache is a paged block table:
      kv_cache: [num_blocks, entries_per_block, head_dim]
      where entries_per_block = block_size // compress_ratio.

    A compressed entry at logical position `p` (= abs_pos // ratio) maps to:
      token_pos   = p * compress_ratio          (position in original token space)
      block_idx   = token_pos // block_size      (logical block index)
      block_off   = p % entries_per_block        (slot within that block)
      phys_block  = block_offsets[batch_id, block_idx]  (physical block in kv_cache)
      write target: kv_cache[phys_block, block_off]

    == FP8 MODEL1 sparse format ==
    When fp8_cache is not None, the kernel also writes to:
      fp8_cache: [num_blocks, entries_per_block, packed_dim=584]
    Per-token layout: [NoPE 448 FP8 | RoPE 128 BF16-as-bytes | 7 E8M0 scales | 1 pad]

    Args:
        compressed_kv: [S, head_dim] flat tensor with compressed results from score_kv.
            Read-only.
        kv_cache: [num_blocks, entries_per_block, head_dim] paged block cache, or None
            to skip the BF16 write. Modified in-place at the computed
            (phys_block, block_off) slots.
        cu_q_seqlens: [B + 1] cumulative query sequence lengths.
        kv_seqlens: [B] total kv sequence lengths (history + new).
        block_offsets: [B, max_blocks] logical-to-physical block index mapping.
            block_offsets[batch_id, block_idx] gives the physical block number.
        compress_ratio: compression ratio (4 or 128).
        block_size: number of original tokens per block (e.g. 128).
        max_seqlen_q: max query seq len (used for prefill grid size).
        fp8_cache: optional [num_blocks, entries_per_block, packed_dim] FP8 cache.
            When provided, MODEL1 sparse FP8 entries are written directly.
    """
    B = kv_seqlens.size(0)
    head_dim = compressed_kv.size(-1)

    is_decoding = compressed_kv.size(0) == B

    BLOCK_D = 128

    if is_decoding:
        grid = (1, B)
    else:
        num_groups = (max_seqlen_q + compress_ratio - 1) // compress_ratio
        grid = (num_groups, B)

    has_bf16 = kv_cache is not None
    has_fp8 = fp8_cache is not None

    # Dummy tensor for when kv_cache or fp8_cache is None.
    # Must be 3D to match the kernel's expected stride count (3 strides).
    dummy = compressed_kv.view(1, 1, -1)[:, :, :1]  # [1, 1, 1] — never accessed due to constexpr guards

    if not has_bf16:
        kv_cache = dummy

    if has_fp8:
        # FlashMLA MODEL1 sparse FP8 layout: the fp8_cache tensor is
        # [num_blocks, entries_per_block, 584] but the actual memory layout
        # has NoPE+RoPE at stride 576 bytes per token, with scales in a
        # separate region. We create three views matching FlashMLA's addressing:
        num_blocks = fp8_cache.size(0)
        entries_per_block_val = fp8_cache.size(1)
        D_NOPE = 448
        D_ROPE_BF16 = 64  # 64 bf16 values = 128 bytes
        NR_DIM = D_NOPE + 2 * D_ROPE_BF16  # 576 bytes per token

        # NoPE+RoPE view: [num_blocks, entries_per_block, 576] e4m3fn
        fp8_flat = fp8_cache.view(num_blocks, -1)
        fp8_nope_rope = fp8_flat[:, :entries_per_block_val * NR_DIM].view(
            num_blocks, entries_per_block_val, NR_DIM)

        # RoPE bf16 view: same memory, viewed as bf16 at offset D_NOPE
        # NoPE+RoPE as bf16: [num_blocks, entries_per_block, 288]
        fp8_nope_rope_bf16 = fp8_nope_rope.view(torch.bfloat16)
        # RoPE starts at bf16 offset D_NOPE//2 = 224 within each token
        fp8_rope_bf16 = fp8_nope_rope_bf16[:, :, D_NOPE // 2:]

        # Scales view: [num_blocks, entries_per_block, 8] uint8
        fp8_scales_u8 = fp8_flat[:, entries_per_block_val * NR_DIM:].view(
            num_blocks, entries_per_block_val, 8).view(torch.uint8)
    else:
        # Dummy views (kernel won't access due to has_fp8=False)
        fp8_nope_rope = dummy
        fp8_rope_bf16 = dummy.view(torch.bfloat16)
        fp8_scales_u8 = dummy.view(torch.uint8)

    _fill_compressed_kv_kernel[grid](
        compressed_kv, kv_cache, cu_q_seqlens, kv_seqlens, block_offsets,
        *compressed_kv.stride(),
        *kv_cache.stride(),
        *block_offsets.stride(),
        fp8_nope_rope,
        *fp8_nope_rope.stride(),
        fp8_rope_bf16,
        *fp8_rope_bf16.stride(),
        fp8_scales_u8,
        *fp8_scales_u8.stride(),
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        block_size=block_size,
        is_decoding=is_decoding,
        has_bf16=has_bf16,
        has_fp8=has_fp8,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
