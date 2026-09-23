# Copyright (c) OpenMMLab. All rights reserved.
"""DeepSeek V4 DSA 6-tuple KV cache paging + metadata builder (Task B-3).

This module wires the per-layer heterogeneous DSA KV caches (the "6-tuple":
compress_kv / swa_kv / state / indexer_state / indexer_k / indexer_scale) into
the lmdeploy engine path.  It is the engine-level replacement for the manually
constructed ``dsa_inputs`` used by the single-NPU forward tests.

Design notes
------------
* The standard lmdeploy paged ``k_cache`` (``past_key_values[layer][0]``) is
  reused as the ``swa_kv`` cache (shape ``[num_blocks, 128, 1, 512]`` bf16,
  num_kv_heads=1, head_dim=512, block_size=MLA_BS=128).  ``v_cache`` is unused
  by V4 (single shared KV) but still allocated by the engine.
* The remaining 5 caches are NOT representable by ``ModelConfig.cache_shapes``
  because that mechanism forces ``kernel_block_size`` as the first dim, while
  the V4 ``state`` / ``indexer_state`` caches use their own block sizes (8 for
  c4, 32 for c128) and the c4/c128 ``state`` dims differ (2048 vs 1024).  The
  DSA ops are sensitive to the exact block size (a uniform block_size=128 for
  the state cache produces NaN).  Hence the extra caches are allocated here,
  per-layer, with the exact shapes that the dlinfer ops expect.
* Caches are allocated once (lazily) and persisted across forward steps so that
  prefill-written compressed/indexer state survives into decode.
* The DSA block tables (compress_kv / state / indexer_k / indexer_state) are
  managed with a simple contiguous-block allocator per request (not the lmdeploy
  block manager), sufficient for single-request prefill+decode.  Full
  multi-request paging is a follow-up.
"""
import math
import os
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

import dlinfer.ops.llm as dllm

# V4 state-cache block sizes. These are the state pool's OWN block size
# (independent of the engine's --cache-block-seq-len), read by the compressor
# kernel as state_cache.shape[1]. vllm-ascend derives them from
# _DSV4_BLOCK_SIZES[engine_bs][0][2/3]; with its engine block_size=32 that is
# c4_state=2 / c128_state=8. We hardcode the same 2/8 here (our engine runs at
# block_size=128, but the state pool is a SEPARATE cache -- its block size need
# not match the engine's). bs=2/8 is what vllm runs the compressor op with
# (verified), and it is 4x more memory-efficient than 8/32: a max-batch-30
# pool at columns=606 needs ~6GB (c4 state) vs ~24GB at bs=8. A uniform 128
# for the state cache produces NaN (the op is bs-sensitive), but 2/8 avoid it.
MLA_BS = 128
SWA_BS = 128
C4_STATE_BS = 2
C128_STATE_BS = 8
# Extra token headroom beyond max_prefill_token_num for the recurrent
# state_bt column count. The compressor kernel indexes state_bt by ORIGINAL
# token position / state_bs (curSeqIdx = bStartPos + sIdx), so the columns
# must cover not just the prefill chunk (max_pt) but also the decode tokens a
# request emits afterwards (positions max_pt..max_pt+decode) -- otherwise the
# first decode past max_pt indexes column max_pt//state_bs, which is one past
# the m_cr=max_pt//state_bs table width -> aicore OOB (507057). 128 tokens
# (one cache block) of headroom covers short decode tails; full max_model_len
# decode needs paged state (TODO), infeasible at batch 30.
# Env-overridable: raise for long-decode workloads (e.g. GPQA CoT / thinking
# traces that emit thousands of decode tokens past the prefill chunk). With
# V4_STATE_PAGED=1 the int32 TABLE grows to session_len//state_bs columns
# (no OOB), but the state POOL (HBM) is still sized at
# max_num_seqs * (max_pt + margin) // state_bs blocks -- so long decode also
# needs a larger margin (and/or lower max_num_seqs) to avoid
# "[V4] state pool exhausted". HBM cost grows ~linearly with the margin.
V4_STATE_DECODE_MARGIN = int(os.environ.get('V4_STATE_DECODE_MARGIN', '128'))
# State-pool BUDGET override (block count). Default 0 = auto (see
# _state_pool_size): with V4_STATE_WINDOW=0 the auto size is the worst-case
# slab max_num_seqs * m_cr + 1 (every concurrent req decodes the full max
# length simultaneously -- over-provisioned, HBM-linear in max_num_seqs,
# OOMs at batch~8). With V4_STATE_WINDOW>0 the auto size is reclaim-aware:
# (max_num_seqs-1)*W + 2*m_cr + 1 (per-req decode live capped at W; only the
# in-flight prefill carries full m_cr), ~15x smaller at max_batch=16 so env=0
# no longer OOMs. vllm-ascend instead treats the compressor state as a unified
# paged KV cache budget (gpu_memory_utilization-driven num_blocks, LRU/evict),
# which is why it runs 16-concurrent DSV4 bf16 where lmdeploy caps at 2-4.
# Setting V4_STATE_POOL_BLOCKS_C4 / _C128 to a fixed budget DECOUPLES the pool
# from max_num_seqs (so raising --max-batch-size no longer linearly blows up
# the state pool). The pool stays a shared free-list sized to the expected
# concurrent working set; if it exhausts (rare: all reqs decode long at once),
# _V4StateAlloc._alloc_ids raises a clean RuntimeError (no aicore crash) --
# graceful per-request failure, not a serve-killer. Pick the budget so that
# N_concurrent * (typ_decode_len / state_bs) blocks fit. c128 auto-derives
# from the c4 budget by the state_bs ratio if _C128 is 0.
V4_STATE_POOL_BLOCKS_C4 = int(os.environ.get('V4_STATE_POOL_BLOCKS_C4', '0'))
V4_STATE_POOL_BLOCKS_C128 = int(os.environ.get('V4_STATE_POOL_BLOCKS_C128', '0'))
# Sliding-window state-block reclamation (vllm-ascend's AscendSlidingWindowMLASpec
# model). The compressor state is a recurrent chain: block j encodes the
# cumulative state up to j*state_bs tokens; the kernel reads only the ANCHOR
# block at start_pos//state_bs each call to continue the chain, and WRITES
# checkpoints forward -- it does NOT re-read historical blocks (verified by
# the _compressor_call arg flow: state_block_table indexed by curSeqIdx/state_bs
# = current position only; vllm-ascend evicts old compressor-state blocks via
# sliding-window, confirming they are not re-read). So unlike the slab/budget
# pool (which holds N/state_bs blocks per req -- concurrency HBM-linear in
# decode length), each req only needs the last W blocks live; older blocks are
# reclaimable. Setting V4_STATE_WINDOW=W keeps the last W state blocks per req
# live and NULLS older table columns (pool id -> 0, the null sentinel the kernel
# guards with `if (idInBlockTable != 0)`), returning their pool ids to the
# free-list. Per-req live state becomes W (constant) regardless of decode
# length -> max_batch is DECOUPLED from max decode length (16 concurrent x 16k
# decode fits a tiny budget = 16*W blocks). 0 = OFF (current append-only
# behavior, backward-compat). W must be >= 2 (current block + previous for the
# recurrent dependency); default in serve scripts is 8 (generous margin for
# chunk boundaries / off-by-one). CORRECTNESS-CRITICAL: if the kernel ever
# re-reads a reclaimed (nulled) column the state silently corrupts (wrong
# output, NOT a crash) -- MUST verify with GSM8K accuracy + W=0/W=8 output
# parity before trusting. See memory dsv4-state-eviction-gap.
V4_STATE_WINDOW = int(os.environ.get('V4_STATE_WINDOW', '0'))
INT_MAX = (1 << 63) - 1

# DSA compress_kv/indexer_k pool budget (block count). These two caches are
# paged by MLA_BS=128 over FULL-kv block indices (one entry per 128-token
# block of the ORIGINAL sequence -- compress_kv[block j] holds the compressed
# KV for tokens [j*128,(j+1)*128), indexer_k[block j] the quantized full kv).
# Unlike swa_kv (windowed/evicted) they MUST retain FULL history -- the
# indexer's top-k selects sparse positions across the whole sequence, so
# evicting an old block drops retrievable history (and, under the old
# swa_block_table aliasing, cross-request reuse of the freed physical block
# injected foreign compressed KV -> the 0903 contamination bug). They
# therefore live in their OWN pool with a slot-stable, append-only
# (state_window=0) block table (_V4StateAlloc reused), separate from swa's
# evicted pool. Pool sizing: env override V4_DSA_KV_POOL_BLOCKS, else auto
# max_num_seqs * cdiv(session_len, MLA_BS) + 1 (full-history worst case).
# Both paths count TOTAL pool blocks incl. the id-0 null sentinel, so the
# usable budget is value-1; the scheduler's dsa_kv admission gate reserves
# against the same usable budget so requests whose full-history demand
# exceeds it stay in waiting (backpressure) instead of being admitted and
# dying at decode time. HBM: compress_kv bf16 128KiB/block, indexer_k int8
# 16KiB/block -- ~6.6GiB/64k or ~3.3GiB/32k over 41 cr>1 / 21 c4 layers.
# On exhaustion _V4StateAlloc raises RuntimeError -- with the scheduler
# gate active this is an invariant violation (a bug), not an expected
# serving condition.
V4_DSA_KV_POOL_BLOCKS = int(os.environ.get('V4_DSA_KV_POOL_BLOCKS', '0'))


def _dsa_kv_pool_size(max_num_seqs, session_len):
    """dsa_kv (compress_kv + indexer_k) pool block count: env budget or auto.

    Auto = max_num_seqs * cdiv(session_len, MLA_BS) + 1 (every concurrent
    req at full session length -- over-provisioned but correct; these
    caches are append-only full-history, not windowed; +1 for the id-0
    null sentinel). Env override decouples from max_num_seqs/session_len
    for HBM tuning and is interpreted the same way: TOTAL pool blocks
    (usable = value - 1)."""
    if V4_DSA_KV_POOL_BLOCKS:
        return V4_DSA_KV_POOL_BLOCKS + 1
    n = max(1, max_num_seqs)
    return n * _cdiv(max(1, session_len), MLA_BS) + 1


def _state_pool_size(cr, max_num_seqs, m_cr):
    """State pool block count: fixed budget (env) or auto.

    Auto (env=0) sizing depends on whether sliding-window reclaim is on:

    - W=0 (append-only, legacy): every concurrent req's full state stays
      live, so the worst case is ``max_num_seqs * m_cr`` (HBM-linear in
      max_batch; OOMs at batch~8 / ~24GB per card).
    - W>0 (reclaim on): per-req DECODE live is capped at W (constant,
      regardless of decode length); only the in-flight prefill needs its
      full ``m_cr`` blocks live (reclaim is decode-only by the Bug B gate,
      so prefill chunks keep every chunk's last block). Peak = ``(max_num_seqs
      - 1) * W`` [decode steady] + ``m_cr`` [1 prefill transient] + ``m_cr``
      [2nd concurrent prefill margin]. This is ~15x smaller than the W=0
      slab at max_batch=16 (e.g. 16*5248=83968 -> 15*8+2*5248=10656 for
      c4), so env=0 no longer OOMs once reclaim is enabled.
    """
    if cr == 4 and V4_STATE_POOL_BLOCKS_C4:
        return V4_STATE_POOL_BLOCKS_C4
    if cr != 4 and V4_STATE_POOL_BLOCKS_C128:
        return V4_STATE_POOL_BLOCKS_C128
    n = max(1, max_num_seqs)
    if V4_STATE_WINDOW > 0:
        # reclaim caps per-req decode live at W; only in-flight prefills
        # carry full m_cr. (n-1)*W decode steady + 2*m_cr prefill margin.
        return (n - 1) * V4_STATE_WINDOW + 2 * m_cr + 1
    return n * m_cr + 1


def _cdiv(a: int, b: int) -> int:
    """Ceiling integer division (a / b rounded up)."""
    return -(-a // b)

# DSA slot-mapping block offset (mirrors vllm DSA_COMPRESSOR_SLOT_MAPPING_*).
_DSA_SLOT_MAPPING_BLOCK_OFFSET = 2


# ---------------------------------------------------------------------------
# YaRN RoPE cos/sin (ported from the single-NPU test / vllm ComplexExpRotary)
# ---------------------------------------------------------------------------
def _yarn_get_inv_freq(rotary_dim, original_max, base, factor,
                       beta_fast=32, beta_slow=1):
    def find_correction_dim(num_rotations):
        return (rotary_dim * math.log(original_max / (num_rotations * 2 * math.pi))) \
            / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot):
        low = math.floor(find_correction_dim(low_rot))
        high = math.ceil(find_correction_dim(high_rot))
        return max(low, 0), min(high, rotary_dim - 1)

    def linear_ramp_mask(low, high):
        if low == high:
            high += 0.001
        ramp = (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(ramp, 0, 1)

    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    inv_freq_extrap = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range(beta_fast, beta_slow)
    inv_freq_mask = (1 - linear_ramp_mask(low, high)) * 1
    return inv_freq_interp * (1 - inv_freq_mask) + inv_freq_extrap * inv_freq_mask


def build_cos_sin(seq_len, rotary_dim, max_pos, base, factor,
                 beta_fast, beta_slow, device, dtype):
    """Main RoPE cos/sin [seq_len, 1, 1, rotary_dim] for MLA q/kv rotary.

    Cast to the model dtype (bf16). NOTE: an fp32 upcast was tested to match
    vllm-ascend (which keeps the table in fp32) but it made the per-layer drift
    WORSE (first drift L19->L8, maxabs L18 0.4->3.78) -- our
    inplace_partial_rotary_mul / compressor op path is closer to vllm with the
    bf16 table than with fp32, so the residual drift is NOT the cos/sin dtype.
    """
    inv_freq = _yarn_get_inv_freq(rotary_dim, max_pos, base, factor,
                                 beta_fast, beta_slow).to(device=device)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return (cos.unsqueeze(1).unsqueeze(1).to(device=device, dtype=dtype),
            sin.unsqueeze(1).unsqueeze(1).to(device=device, dtype=dtype))


def build_full_compress_cos_sin(max_pos, rotary_dim, base, factor,
                                original_max, beta_fast, beta_slow,
                                device, dtype):
    """Full compressor RoPE cos/sin [max_pos, rotary_dim] (2D)."""
    inv_freq = _yarn_get_inv_freq(rotary_dim, original_max, base, factor,
                                  beta_fast, beta_slow).to(device=device)
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return (cos.to(device=device, dtype=dtype),
            sin.to(device=device, dtype=dtype))


# ---------------------------------------------------------------------------
# Per-layer 6-tuple cache allocation
# ---------------------------------------------------------------------------
def _layer_compress_ratios(hf_config):
    cr = getattr(hf_config, 'compress_ratios', None)
    n = hf_config.num_hidden_layers
    if cr is None:
        cr = [0] * n
    return [int(cr[i]) if i < len(cr) else 0 for i in range(n)]


def allocate_v4_caches(num_blocks, hf_config, device, dtype,
                       max_num_seqs=1, max_prefill_token_num=2048,
                       session_len=65536):
    """Allocate the per-layer heterogeneous 6-tuple DSA caches.

    Returns a list (len=num_layers) where each entry is the 6-tuple
    ``(compress_kv, swa_kv, state, indexer_state, indexer_k, indexer_scale)``
    with the layer-appropriate caches allocated and the unused ones set to
    ``None``.  ``swa_kv`` is left ``None`` here because the engine's paged
    ``k_cache`` is reused as swa_kv (wired in ``build_v4_dsa_inputs``).

    Pool sizing:
    * ``compress_kv`` / ``indexer_k`` / ``indexer_scale`` are packed into the
      engine's main paged cache and indexed by the per-request main block table
      (``block_offsets``), so they are sized at ``num_blocks`` (the main KV
      block count) -- one entry per main block.
    * the recurrent ``state`` / ``indexer_state`` caches are SEPARATE pools
      (block_size C4_STATE_BS=8 / C128_STATE_BS=32) with their own per-request
      block tables, NOT the main one. A single request at ``max_prefill_token_num``
      needs ``M = cdiv(max_prefill_token_num // min_cr, C4_STATE_BS)`` state
      blocks (min_cr=4 -> c4 is the densest). The shared state block table has
      ``M`` columns, so the pool must hold ``max_num_seqs * M`` blocks to give
      every concurrent request its own contiguous ``[req*M, (req+1)*M)`` range.
    """
    # Record the per-rank max_num_seqs (the pool-sizing ceiling) for the
    # decode path: build_v4_dsa_inputs pads every decode step up to this
    # size so the single captured decode graph (get_capture_batch_sizes ->
    # [max_num_seqs]) sees a constant batch -- otherwise _pin reallocates
    # the persistent DSA buffers on shape change -> stale graph address.
    try:
        _DECODE_META.max_num_seqs = max(max_num_seqs,
                                        _DECODE_META.max_num_seqs or 0)
    except Exception:
        pass
    # dsa_kv (compress_kv + indexer_k) own pool sizing: append-only
    # full-history (MLA_BS-paged, state_window=0). Stash pool size + table
    # column count for build_v4_dsa_inputs -> _get_dsa_kv_alloc. See
    # _dsa_kv_pool_size / V4_DSA_KV_POOL_BLOCKS doc above.
    _dsa_kv_pool = _dsa_kv_pool_size(max_num_seqs, session_len)
    _dsa_kv_max_cols = _cdiv(max(1, session_len), MLA_BS)
    try:
        _DECODE_META.dsa_kv_pool = max(_dsa_kv_pool,
                                        getattr(_DECODE_META, 'dsa_kv_pool', 0))
        _DECODE_META.dsa_kv_max_cols = _dsa_kv_max_cols
        # session_len for the in-graph swa block-table full-extent
        # reconstruction (build_v4_decode_meta_in_graph: max_full_cols =
        # cdiv(session_len, swa_block_size)). See dsv4-sw128-compaction-vs-nullpad-rootcause.
        _DECODE_META.session_len = max(
            session_len, getattr(_DECODE_META, 'session_len', 0))
    except Exception:
        pass
    head_dim = getattr(hf_config, 'head_dim', 512)
    index_head_dim = getattr(hf_config, 'index_head_dim', 128)
    ratios = _layer_compress_ratios(hf_config)
    import os as _os
    _dbg = _os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
    # Per-compress-ratio state block-table COLUMN count per request. The
    # compressor kernel (compressor_block_vec_perf.h SaveState/ReadFromCacheState)
    # indexes state_bt[batchIdx * maxBlockNumPerBatch + curSeqIdx/blockSize]
    # where curSeqIdx = bStartPos + sIdx, and bStartPos = start_pos[bIdx] is in
    # ORIGINAL token units (start_pos is the prefill start position in original
    # tokens, not compressed). So the column index reaches up to
    # (max token position) / state_bs, i.e. the table needs
    # max_position // state_bs columns -- NOT max_position // cr (compressed).
    # This was confirmed empirically: the old "compressed" sizing
    # max_pt//cr+seqs gave 577 cols (c4), which is > num_cmp(335) yet the
    # 1336-prefill STILL crashed at the compressor -- because the kernel
    # indexes by ORIGINAL position (needs 1336//2=668 > 577). vllm-ascend
    # matches: its state_bt has max_model_len//state_bs columns (4096 for c4
    # at bs=2, len 8192). We size at max_pt//state_bs (covers prefill +
    # short decode; full max_model_len decode would need max_model_len//bs
    # but that is memory-infeasible at batch 30 -- paging TODO). Each layer's
    # pool = max_num_seqs * M_cr so every concurrent request gets a contiguous
    # [req*M_cr, (req+1)*M_cr) range.
    def _m_for_cr(cr):
        state_bs = C4_STATE_BS if cr == 4 else C128_STATE_BS
        return (max_prefill_token_num + V4_STATE_DECODE_MARGIN) // state_bs
    m_c4, m_c128 = _m_for_cr(4), _m_for_cr(128)
    if _dbg:
        n_c4 = sum(1 for r in ratios if r == 4)
        n_c128 = sum(1 for r in ratios if r > 4)
        sp_c4 = _state_pool_size(4, max_num_seqs, m_c4)
        sp_c128 = _state_pool_size(128, max_num_seqs, m_c128)
        _env_c4 = V4_STATE_POOL_BLOCKS_C4
        _env_c128 = V4_STATE_POOL_BLOCKS_C128
        print(f'[V4-ALLOC] num_blocks={num_blocks} n_layers={len(ratios)} '
              f'n_c4={n_c4} n_c128={n_c128} head_dim={head_dim} '
              f'idx_head_dim={index_head_dim} dtype={dtype} '
              f'max_num_seqs={max_num_seqs} max_pt={max_prefill_token_num} '
              f'm_c4={m_c4}(pool {sp_c4}'
              f'{", env=" + str(_env_c4) if _env_c4 else ", slab"}) '
              f'm_c128={m_c128}(pool {sp_c128}'
              f'{", env=" + str(_env_c128) if _env_c128 else ", slab"}) '
              f'state_window={V4_STATE_WINDOW}',
              flush=True)
    caches = []
    # Reset the dsa_kv tensor registry (one alloc per worker; robust to
    # re-init). Populated per same-cr layer below for zero-on-free.
    _DSA_KV_TENSORS.clear()
    for cr in ratios:
        if cr <= 1:
            # SWA layer: only swa_kv (the paged k_cache); no extra caches.
            caches.append((None, None, None, None, None, None))
            continue
        coff = 1 + (cr == 4)            # overlap buffer on c4
        # compress_kv: [dsa_kv_pool, MLA_BS, 1, head_dim] bf16 -- OWN pool
        # (NOT the main num_blocks/swa pool), indexed by the separate
        # full-history dsa_kv_bt block table (see _get_dsa_kv_alloc). Old
        # design aliased swa_block_table -> swa eviction cross-request-
        # contaminated compress_kv (0903 structural bug). Now isolated.
        compress_kv = torch.zeros(_dsa_kv_pool, MLA_BS, 1, head_dim,
                                  dtype=dtype, device=device)
        # state cache (separate, per-cr pool, own per-request block table):
        #   c4   -> [pool_c4, C4_STATE_BS, 1, coff*2*head_dim] f32
        #   c128 -> [pool_c128, C128_STATE_BS, 1, 2*head_dim] f32
        # pool = max_num_seqs * M_cr (see _m_for_cr above).
        m_cr = _m_for_cr(cr)
        # +1 reserves block 0 as the null sentinel: the compressor kernel
        # (compressor_block_vec_perf.h WriteToCacheState/ReadState) guards
        # writes/reads with `if (idInBlockTable != 0)` -- block 0 is treated
        # as unallocated/null. vllm-ascend's state allocator never assigns
        # block 0 (real blocks start at 1+). lmdeploy's arange previously
        # started at 0, so the first state block hit the null path (write
        # skipped, recurrent read of an unwritten/zeroed block) and the
        # recurrent state chain diverged -> aicore MTE OOB on long prefills
        # (num_cmp > 1). Reserving block 0 (ids start at 1) matches vllm.
        state_pool_cr = _state_pool_size(cr, max_num_seqs, m_cr)
        if cr == 4:
            state_dim = coff * 2 * head_dim          # 2*2*512 = 2048
            state = torch.zeros(state_pool_cr, C4_STATE_BS, 1, state_dim,
                                dtype=torch.float32, device=device)
        else:  # c128
            state_dim = 2 * head_dim                 # 1024
            state = torch.zeros(state_pool_cr, C128_STATE_BS, 1, state_dim,
                                dtype=torch.float32, device=device)
        indexer_state = None
        indexer_k = None
        indexer_scale = None
        if cr == 4:
            idx_state_dim = coff * 2 * index_head_dim   # 2*2*128 = 512
            indexer_state = torch.zeros(state_pool_cr, C4_STATE_BS, 1,
                                        idx_state_dim, dtype=torch.float32,
                                        device=device)
            indexer_k = torch.zeros(_dsa_kv_pool, MLA_BS, 1, index_head_dim,
                                    dtype=torch.int8, device=device)
            indexer_scale = torch.zeros(_dsa_kv_pool, MLA_BS, 1, 1,
                                         dtype=torch.float16, device=device)
        caches.append((compress_kv, None, state, indexer_state,
                       indexer_k, indexer_scale))
        # Register this layer's dsa_kv tensors for zero-on-free. The dsa_kv_bt
        # is per-cr shared, so a freed block id must be zeroed across all
        # same-cr layers (the bt indexes the same virtual block in each).
        _DSA_KV_TENSORS.setdefault(cr, []).append(
            (compress_kv, indexer_k, indexer_scale))
    return caches


# ---------------------------------------------------------------------------
# DSA metadata builder (per-layer, single-request prefill + decode)
# ---------------------------------------------------------------------------
class _V4ReqState:
    """Per-request DSA block-table / metadata state, persisted across steps.

    A single active request is tracked (sufficient for the reduced run and
    single-prompt serve).  Multi-request paging is a follow-up.
    """

    def __init__(self):
        self.prefill_kv_len = 0          # kv length at prefill time
        # per-layer block tables for the extra caches (built on prefill)
        self.compress_bt = None
        self.state_bt = None
        self.indexer_k_bt = None
        self.indexer_state_bt = None
        # per-layer num_compressed_tokens at prefill
        self.num_cmp = None
        self.compress_cos = None
        self.compress_sin = None


# ---------------------------------------------------------------------------
# Paged per-request state block allocator (V4_STATE_PAGED=1)
# ---------------------------------------------------------------------------
class _V4StateAlloc:
    """Paged per-request state-block allocator for one compress-ratio's pool.

    Replaces the contiguous ``arange(1, num_reqs*M+1).view(num_reqs, M)`` state
    block table, whose M was capped at ``max_prefill_token_num//state_bs``
    (c4: 1216 cols -> 2432-token ceiling) -> aicore MTE OOB on any sequence
    past 2432 tokens (the compressor kernel indexes
    ``state_bt[bIdx, (bStartPos+sIdx)/state_bs]`` by ORIGINAL token position;
    multi-chunk prefill chunk2 has bStartPos=2304, so it reaches column 1506).

    The state_bt block table is ``[num_reqs, max_cols = session_len//state_bs]``:
    entry ``[r, j]`` maps request r's logical state block j -> a POOL block id.
    The state pool (allocated in ``allocate_v4_caches`` at
    ``max_num_seqs * m_cr`` blocks, e.g. 15*1216=18240 c4) is a SHARED BUDGET,
    not a per-request contiguous range -- so a single long request (4k prefill
    needs 4096//2=2048 blocks) fits the over-provisioned pool WITHOUT growing
    HBM (idle HBM ~6.5GB; full session_len per-request sizing = ~36GB ->
    infeasible). Only the cheap int32 TABLE grows (to session_len//state_bs
    columns, ~245KB/layer); the pool stays the same.

    Assignment is STABLE across steps: a request's logical blocks 0..n-1 keep
    the same pool ids forever (recurrent reads hit the same pool location --
    correctness; the recurrent state MUST live at a stable address). New
    blocks are appended as ``kv_seqlen`` grows. Completion needs NO engine free
    hook: a request's kv_seqlen only GROWS during its life, so a slot whose
    kv_len DECREASES was freed-and-reused by a new (shorter) request -> free
    the old ids and re-alloc. Slots that vanish (``r >= num_reqs``) are freed.
    The only miss is a new request whose kv_len happens to equal its
    predecessor's (vanishingly rare): it would reuse stale state, affecting only
    that request's output -- not a crash.

    Graph safety: the table is pinned ONCE at ``[max_num_seqs, max_cols]``
    (mode-(a) _pin, same as swa_block_table); the active
    ``[num_reqs, max_cols]`` prefix is copy_'d in each eager pre-step, and the
    captured decode graph reads the (stable) data_ptr whose content was updated
    before replay -- exactly how swa_block_table already works in the graph.
    """

    def __init__(self, state_bs, max_cols, pool_size, max_num_seqs, device,
                 state_window=0, zero_fn=None):
        self.state_bs = state_bs
        self.max_cols = int(max_cols)
        self.pool_size = int(pool_size)        # ids 1..pool_size-1 valid (0=null)
        self.max_num_seqs = int(max_num_seqs)
        self.device = device
        # Optional zero-on-free callback: invoked with the list of real (non-
        # null) pool ids that are being returned to the free-list, so the
        # caller can zero the backing pool tensors at those indices. Used by
        # the dsa_kv (compress_kv/indexer_k) allocator to break the 0904
        # cross-request contamination: a freed block was reused with stale
        # data, and the HCA/indexer read historical positions the new owner
        # had not (yet) written -> foreign-text bleed (accuracy decayed
        # 71%->33% as the 2048-block pool filled). Write-before-read was
        # assumed to self-cleanse (see _get_dsa_kv_alloc L796) but that only
        # covers positions the CURRENT request writes; a recycled block's
        # unwritten positions still hold a prior request's data. Zeroing on
        # free guarantees a clean block for the next owner. Zero HBM cost.
        self._zero_fn = zero_fn
        # Sliding-window reclamation (0=off, append-only). When >0, only the
        # last `state_window` blocks per slot stay live; older columns are
        # nulled (pool id -> 0) and their ids returned to the free-list. See
        # V4_STATE_WINDOW env doc above. Per-req live state -> W (constant),
        # decoupling max_batch from max decode length.
        self.state_window = int(state_window)
        from collections import deque
        # free-list of pool block ids (1..pool_size-1); 0 is the null sentinel.
        self._free = deque(range(1, self.pool_size))
        self._slot_ids = {}    # slot -> list[int] pool ids (0 = reclaimed/null)
        self._slot_kv = {}      # slot -> last kv_len (reset detection)
        # high-water mark of reclaimed columns per slot (avoids O(N) re-scan:
        # only columns [upto, need-W) are newly reclaimed each step).
        self._slot_reclaimed_upto = {}
        # ---- seq_id-keyed state (the ROBUST path) ----
        # The slot index is NOT stable: reindex() compacts the batch when a
        # request finishes -- survivors renumber to lower slots. The old
        # slot-keyed turnover check (`kv > prev + max_pt`) then misfired on a
        # surviving long-decode request (its new slot held a FINISHED req's
        # short prev) -> _free_slot + FULL-HISTORY re-alloc (need=kv//bs up to
        # ~9714 blocks at 19k-token decode) -> "state pool exhausted" spike
        # (the 0903 batch8 x 32k crash). Keying by the scheduler's STABLE
        # seq_id (a monotonic counter, never reused) makes blocks FOLLOW the
        # request across any renumber; a request's kv only grows, so there is
        # NO turnover to detect and NO spike. Used when assign() is passed
        # seq_ids (always, on the V4 / ar-strategy path). Falls back to slot-
        # keying when seq_ids is None (non-V4 / un-plumbed paths).
        self._req_ids = {}            # seq_id -> list[int] pool ids
        self._req_kv = {}             # seq_id -> last kv_len
        self._req_reclaimed_upto = {}  # seq_id -> reclaimed high-water
        # ---- lazy LRU eviction (preemption-safe free) ----
        # A request can be ABSENT from the current batch for two reasons:
        # (a) FINISHED (gone forever) -- its blocks should be recycled, OR
        # (b) PREEMPTED / swap-out under KV pressure (will return) -- its
        # blocks MUST be preserved, else on return _req_kv[key] is gone and
        # the allocator re-allocates the FULL history -> state pool exhausted
        # spike (the 0903 preemption crash: per_key_kv={2: None}, need +4064).
        # The old code freed on FIRST absence, conflating (a) and (b) -> the
        # preemption spike. Lazy LRU: never free on absence; track last-seen
        # step; evict the LONGEST-ABSENT keys ONLY when _alloc_ids can't
        # satisfy (genuine pool pressure). A preempted request that returns
        # quickly keeps its blocks; only truly-finished (long-absent) keys
        # are evicted, and only under pressure.
        self._last_seen = {}           # seq_id -> step counter of last presence
        self._step = 0                 # monotonic step counter
        self._cur_keys = set()         # keys present in the current batch
                                       # (set by assign; read by _alloc_ids LRU)
        self._backing = None    # [max_num_seqs, max_cols] int32, graph-stable
        import numpy as _np
        self._np = _np
        self._cpu = None         # [max_num_seqs, max_cols] int32 numpy scratch
        # dirty-skip: the captured graph reads the (stable) backing data_ptr;
        # its CONTENT only changes when a new pool block is allocated or a slot
        # is freed/reused. On steady decode that is rare (c4 ~every state_bs=2
        # steps, c128 ~every 128) -- so skip the H2D (the only remaining host
        # sync / 507018 racer) on unchanged steps; backing[:num_reqs] already
        # holds the byte-identical value. Set True by _alloc_ids/_free_slot/
        # sliding-window reclaim.
        self._changed = True
        self._last_num_reqs = -1

    def _alloc_ids(self, n):
        if len(self._free) < n:
            # Lazy LRU rescue: before raising, evict the longest-ABSENT
            # (finished) seq_id-keyed requests to replenish the free list.
            # A finished request is absent from every batch; a preempted
            # request is absent transiently. Evicting oldest-absent-first
            # recycles finished requests' blocks while preserving recently-
            # preempted ones (minimizing the re-alloc spike on return).
            # Only the seq_id path tracks last_seen; skip if unused.
            if self._last_seen and self._req_ids:
                # Two-pass eviction. Pass 1: keys absent >= guard steps --
                # finished requests (absent forever) and long-preempted ones.
                # Pass 2 (LAST RESORT): recently-absent keys. A chunked
                # prefill is absent BETWEEN its own chunks (a few steps);
                # with decode keys present at every decode step, oldest-
                # first alone picks the mid-chunk prefill key (0909 run6:
                # a 12.1k-token prompt's between-chunk key was the oldest
                # absent key) -- evicting it mid-flight loses its compressor
                # chain state AND its re-entry re-allocates the FULL extent
                # (1522 of 767 blocks -> engine death). Guard-first keeps
                # recycling to genuinely-finished requests.
                _guard = 256
                _now = self._step
                absent = sorted(
                    ((s, k) for k, s in self._last_seen.items()
                     if k not in self._cur_keys),
                    reverse=True)
                _old = [e for e in absent if _now - e[0] >= _guard]
                _recent = [e for e in absent if _now - e[0] < _guard]
                for _pass, _group in ((1, _old), (2, _recent)):
                    for _step_val, k in _group:
                        if len(self._free) >= n:
                            break
                        self._free_key(k, self._req_ids, self._req_kv,
                                       self._req_reclaimed_upto)
                        self._last_seen.pop(k, None)
                        if _pass == 2:
                            print(f'[V4-LRU-LASTRESORT] evicted '
                                  f'recently-absent sid={k} (absent '
                                  f'{self._step - _step_val} steps < guard '
                                  f'{_guard}) -- mid-flight state-loss '
                                  f'risk; free={len(self._free)}', flush=True)
                        elif os.environ.get('V4_DEBUG_ALLOC', '0') == '1':
                            print(f'[V4-LRU] evicted seq_id={k} '
                                  f'absent_since_step={_step_val} '
                                  f'now_step={self._step} '
                                  f'free={len(self._free)}', flush=True)
            if len(self._free) < n:
                # Diagnostic: per-key block counts so the exhaustion cause is
                # self-evident -- 1 key holding ~pool (a leak / per-layer double
                # counting) vs N keys each holding ~(pool/N) (too many concurrent).
                # Report the active keying: seq_id-keyed (_req_*) if in use, else
                # the legacy slot-keyed (_slot_*).
                if self._req_ids:
                    _per_key = {k: len(ids) for k, ids in self._req_ids.items()}
                    _per_kv = {k: self._req_kv.get(k) for k in self._req_ids}
                    _mode = 'seq_id'
                else:
                    _per_key = {r: len(ids) for r, ids in self._slot_ids.items()}
                    _per_kv = {r: self._slot_kv.get(r) for r in self._slot_ids}
                    _mode = 'slot'
                raise RuntimeError(
                    f"[V4] state pool exhausted: need +{n} blocks, only "
                    f"{len(self._free)} free of {self.pool_size - 1} "
                    f"(state_bs={self.state_bs} max_cols={self.max_cols} "
                    f"max_num_seqs={self.max_num_seqs} keying={_mode}). "
                    f"per_key_blocks={_per_key} per_key_kv={_per_kv}. "
                    f"If 1 key holds ~pool under keying=seq_id, that is a "
                    f"reclaim bug (state_window not bounding live). If N keys "
                    f"each hold ~(pool/N), too many concurrent long sequences "
                    f"-- raise the state cache budget or reduce concurrency.")
        self._changed = True          # new pool ids -> table content changed
        return [self._free.popleft() for _ in range(n)]

    def _free_key(self, key, ids_map, kv_map, reclaim_map):
        """Free a key's pool ids (filtering 0 reclaimed sentinels) and reset
        its kv/reclaim tracking. Generic over the keying (slot index or
        seq_id) -- caller passes the right maps."""
        ids = ids_map.pop(key, [])
        if ids:
            # Filter 0 (null sentinel): with sliding-window reclaim, a key's
            # ids list contains 0 for already-reclaimed columns -- those must
            # NOT re-enter the free-list (0 is the kernel's null guard;
            # handing it out as a real id would silently skip state writes).
            freed = [x for x in ids if x]
            # Zero-on-free: break cross-request contamination for the dsa_kv
            # pool (compress_kv/indexer_k). A block returning to the free-
            # list still holds its prior owner's data; without zeroing, the
            # next owner reads stale data at positions it has not written
            # yet (HCA/indexer historical reads) -> foreign-text bleed.
            # self._zero_fn is None for the state pool (write-before-read
            # suffices there) and set only for the dsa_kv allocator.
            if freed and self._zero_fn is not None:
                self._zero_fn(freed)
            self._free.extend(freed)
            self._changed = True      # freed ids -> table content changed
        kv_map.pop(key, None)
        reclaim_map.pop(key, None)

    def _free_slot(self, r):
        """Legacy slot-keyed free (kept for the slot-keyed fallback path)."""
        self._free_key(r, self._slot_ids, self._slot_kv,
                       self._slot_reclaimed_upto)

    def _reclaim_keep(self, is_decoding, max_pt):
        """Blocks kept live at the tail after sliding-window reclaim:
        DECODE keeps the last ``state_window`` (W) blocks; PREFILL keeps
        ``ceil(max_pt/bs) + W + 2`` (the chunk's write range plus the
        previous chunk's anchor block, with margin). See the assign
        docstring for the chain-safety argument."""
        if is_decoding:
            return self.state_window
        return _cdiv(max_pt, self.state_bs) + self.state_window + 2

    def assign(self, num_reqs, kv_lens_cpu, is_decoding, max_pt,
               seq_ids=None):
        """Return the pinned ``[num_reqs, max_cols]`` int32 state block table.

        ``kv_lens_cpu`` is a CPU list of per-request total kv lengths (one
        ``.tolist()`` sync per step, shared across all cr layers). Reclaim
        window: DECODE keeps the last ``state_window`` (W) blocks live;
        PREFILL keeps ``ceil(max_pt/bs) + W + 2`` -- chunk-local writes
        [prev_need, need) plus the previous chunk's last block (the anchor
        the recurrent chain reads). Older blocks are never read again: the
        compressor state chain is strictly sequential (block j reads only
        j-1), so prefill-phase reclaim is safe and bounds live state to
        ~2 chunks instead of the unbounded full extent (GPQA 16k: a 12k
        session's c128 extent 1555 blocks >> pool 768 -> 0908 "state pool
        exhausted" engine crash). Reclaim runs BEFORE the extend so the
        freed blocks cover the chunk growth. ``max_pt`` bounds legitimate
        per-step kv growth (one prefill chunk); also used by the slot-keyed
        fallback (see turnover note below).

        ``seq_ids`` (list[int], slot order) keys block state by the scheduler's
        STABLE request id instead of the slot index. The slot index is NOT
        stable: ``reindex()`` compacts the batch when a request finishes
        (survivors renumber to lower slots), so the old slot-keyed turnover
        check (``kv > prev + max_pt``) misfired on a surviving long-decode
        request whose new slot held a FINISHED request's short prev ->
        ``_free_key`` + FULL-HISTORY re-alloc (need=kv//bs up to ~9714 blocks
        at 19k-token decode) -> "state pool exhausted" spike (0903 batch8 x
        32k crash). With seq_id keying a request's kv only grows and seq_id is
        never reused, so there is NO turnover to detect -> no spike. When
        ``seq_ids`` is None (non-V4 / un-plumbed paths), falls back to the
        legacy slot-keyed path with turnover detection."""
        _np = self._np
        use_sid = seq_ids is not None and len(seq_ids) >= num_reqs
        if use_sid:
            ids_map = self._req_ids
            kv_map = self._req_kv
            reclaim_map = self._req_reclaimed_upto
            keys = list(seq_ids[:num_reqs])
            self._step += 1
            self._cur_keys = set(keys)
            # 1. Record presence for LRU. Do NOT free absent keys here -- a
            # preempted request (swap-out under KV pressure) is absent but
            # will return; freeing it loses _req_kv[key] -> full-realloc
            # spike on return (the 0903 preemption crash). _alloc_ids does
            # lazy LRU eviction (longest-absent-first) only under genuine
            # pool pressure, so finished requests are recycled while
            # preempted ones keep their blocks.
            for k in keys:
                self._last_seen[k] = self._step
            # 2. per active slot: grow to ceil(kv/bs). NO turnover branch: a
            # real request's kv is monotonic (only grows) and seq_id is never
            # reused, so `kv > prev + max_pt` cannot fire for the same request.
            # The old spike was a slot-identity confusion, impossible here.
            for r in range(num_reqs):
                k = keys[r]
                kv = int(kv_lens_cpu[r])
                prev = kv_map.get(k)
                if prev is not None and kv < prev:
                    if is_decoding:
                        # Decode-step dip = a transient accounting artifact
                        # (kv for a live seq_id is monotonic; a genuine
                        # restart re-enters through PREFILL, never decode).
                        # Clamp the baseline UP to prev instead of resetting:
                        # resetting freed + re-allocated the whole extent and
                        # jumped the reclaim watermark over the fresh blocks
                        # (pinned forever) -- the 0908 run4 c128 pool drain
                        # ("need +85, 1 free of 767"). Clamping also keeps
                        # the sid_incr source (_req_kv + 1) anchored at the
                        # true high kv, so the extent keeps covering the
                        # kernel's real write window.
                        kv = prev
                    else:
                        # Prefill-step dip: genuine restart re-entry (the
                        # fresh chain starts from a shorter kv). Full reset
                        # so the fresh chain's low columns get real block
                        # ids -- a state write to sentinel column 0 is
                        # silently skipped by the kernel.
                        self._free_key(k, ids_map, kv_map, reclaim_map)
                cur = ids_map.setdefault(k, [])
                need = _cdiv(kv, self.state_bs)
                # Sliding-window reclaim (keyed by seq_id; decode keeps W,
                # prefill keeps chunk+W+2 -- see the assign docstring).
                # Runs BEFORE the extend so freed blocks cover the chunk
                # growth (the 0908 GPQA exhaustion: +1496-block prefill jump
                # with 526 free killed the engine at extend time).
                keep = self._reclaim_keep(is_decoding, max_pt)
                if (self.state_window > 0 and not cur and need > keep):
                    # Re-entry with NO live state at a long kv = the key was
                    # evicted mid-flight (LRU between its own chunks). A
                    # full-extent re-alloc (need real ids) can exceed the
                    # whole pool (0909 run6: a 12.1k-token prompt's last
                    # chunk -> need=1522 of 767 -> engine death). The dead
                    # prefix columns were already conceptually reclaimed;
                    # seed them with sentinel 0s so the extend below
                    # allocates ONLY the live tail (keep blocks). The
                    # reclaimed watermark then advances normally. WARNING:
                    # the compressor chain anchor state for this seq was
                    # lost with the eviction -- its output may be degraded,
                    # but the engine survives (graceful degradation).
                    cur.extend([0] * (need - keep))
                    print(f'[V4-ALLOC-REENTRY] WARNING: sid={k} re-entered '
                          f'with no live state at kv={kv} (evicted '
                          f'mid-flight under pool pressure) -- chain anchor '
                          f'lost, output may be degraded; allocating tail '
                          f'keep={keep} of need={need}', flush=True)
                if (self.state_window > 0 and need > keep):
                    new_upto = need - keep
                    upto = reclaim_map.get(k, 0)
                    if new_upto > upto:
                        for j in range(upto, min(new_upto, len(cur))):
                            if cur[j]:
                                self._free.append(cur[j])
                                cur[j] = 0
                                self._changed = True
                        # Cap the watermark at what was actually walked
                        # (len(cur) BEFORE the extend below). Normally
                        # new_upto <= len(cur) (keep >= one chunk's growth),
                        # so this is a no-op -- but right after a reset cur
                        # is EMPTY while new_upto can be large: advancing the
                        # watermark past an extent that has not been
                        # allocated yet would pin the fresh blocks past the
                        # reclaim loop forever (freed-never -- the 0908 run4
                        # c128 drain). Capped, the fresh extend lands below
                        # the watermark and the next step's reclaim frees it.
                        reclaim_map[k] = min(new_upto, len(cur))
                        self._reclaim_cnt = getattr(self, '_reclaim_cnt', 0) + 1
                        if (self._reclaim_cnt <= 6
                                and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'):
                            _live = sum(1 for x in cur if x)
                            print(f'[V4-RECLAIM] bs={self.state_bs} '
                                  f'W={self.state_window} sid={k} need={need} '
                                  f'upto={upto}->{min(new_upto, len(cur))} '
                                  f'cur_len={len(cur)} '
                                  f'live_now={_live} free={len(self._free)}',
                                  flush=True)
                if need > len(cur):
                    cur.extend(self._alloc_ids(need - len(cur)))
                kv_map[k] = kv
        else:
            ids_map = self._slot_ids
            kv_map = self._slot_kv
            reclaim_map = self._slot_reclaimed_upto
            keys = list(range(num_reqs))
            # 1. free disappeared slots (engine dropped them: r >= num_reqs).
            for r in list(ids_map.keys()):
                if r >= num_reqs:
                    self._free_key(r, ids_map, kv_map, reclaim_map)
            # 2. per active slot: detect reuse (kv shrank) + turnover (kv jumped
            # past max_pt = the slot was reused by a different request whose
            # cached prev is stale) -> reset before extending.
            for r in range(num_reqs):
                kv = int(kv_lens_cpu[r])
                prev = kv_map.get(r)
                if prev is not None and (kv < prev or kv > prev + max_pt):
                    self._free_key(r, ids_map, kv_map, reclaim_map)
                cur = ids_map.setdefault(r, [])
                need = _cdiv(kv, self.state_bs)
                # Sliding-window reclaim (slot-keyed fallback; decode keeps W,
                # prefill keeps chunk+W+2 -- mirror of the seq_id branch above,
                # which documents the chain-safety argument).
                keep = self._reclaim_keep(is_decoding, max_pt)
                if (self.state_window > 0 and not cur and need > keep):
                    # Mirror of the seq_id branch: re-entry after a reset
                    # (turnover/freed slot) at a long kv must NOT re-allocate
                    # the full extent -- seed the dead prefix with sentinels
                    # so only the live tail is allocated (0909 run6 spike).
                    cur.extend([0] * (need - keep))
                    print(f'[V4-ALLOC-REENTRY] WARNING: slot={r} re-entered '
                          f'with no live state at kv={kv} -- allocating tail '
                          f'keep={keep} of need={need}', flush=True)
                if (self.state_window > 0 and need > keep):
                    new_upto = need - keep
                    upto = reclaim_map.get(r, 0)
                    if new_upto > upto:
                        for j in range(upto, min(new_upto, len(cur))):
                            if cur[j]:
                                self._free.append(cur[j])
                                cur[j] = 0
                                self._changed = True
                        # Mirror of the seq_id branch: cap the watermark at
                        # the walked extent so a fresh extend never lands
                        # past it (freed-never).
                        reclaim_map[r] = min(new_upto, len(cur))
                        self._reclaim_cnt = getattr(self, '_reclaim_cnt', 0) + 1
                        if (self._reclaim_cnt <= 6
                                and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'):
                            _live = sum(1 for x in cur if x)
                            print(f'[V4-RECLAIM] bs={self.state_bs} '
                                  f'W={self.state_window} slot={r} need={need} '
                                  f'upto={upto}->{min(new_upto, len(cur))} '
                                  f'cur_len={len(cur)} '
                                  f'live_now={_live} free={len(self._free)}',
                                  flush=True)
                if need > len(cur):
                    cur.extend(self._alloc_ids(need - len(cur)))
                kv_map[r] = kv
        # 3. dirty-skip: if no block was allocated/freed/reused this step and
        # the batch size is unchanged, the backing[:num_reqs] prefix already
        # holds the byte-identical table -- the captured graph reads the stable
        # data_ptr, so SKIP the H2D (the only remaining host sync / 507018
        # racer). On steady decode this skips all but the boundary-cross steps
        # (c4 ~every 2 steps, c128 ~every 128). A reindex compaction changes
        # num_reqs -> skips the dirty-skip -> rebuilds (slot->key map changed).
        if (not self._changed and self._backing is not None
                and num_reqs == self._last_num_reqs):
            return self._backing[:num_reqs]
        # 4. build the active [num_reqs, max_cols] int32 table; pin graph-stable.
        if (self._cpu is None or self._cpu.shape[0] != self.max_num_seqs
                or self._cpu.shape[1] != self.max_cols):
            self._cpu = _np.zeros((self.max_num_seqs, self.max_cols),
                                  dtype=_np.int32)
        else:
            self._cpu[:num_reqs] = 0
        for r in range(num_reqs):
            ids = ids_map.get(keys[r], [])
            if ids:
                n = min(len(ids), self.max_cols)
                self._cpu[r, :n] = _np.asarray(ids[:n], dtype=_np.int32)
        if (self._backing is None
                or self._backing.shape[0] != self.max_num_seqs
                or self._backing.shape[1] != self.max_cols):
            self._backing = torch.zeros((self.max_num_seqs, self.max_cols),
                                        dtype=torch.int32, device=self.device)
        self._backing[:num_reqs] = torch.from_numpy(self._cpu[:num_reqs]).to(
            self.device)
        if num_reqs < self.max_num_seqs:
            self._backing[num_reqs:].zero_()
        self._changed = False
        self._last_num_reqs = num_reqs
        return self._backing[:num_reqs]


_STATE_ALLOCS = {}        # cr -> _V4StateAlloc singleton
_PAGED_BT_CACHE = {}      # (cr) -> pinned view, per-step (cleared each step)
_V4_BC_STEP = 0           # per-rank build_v4_dsa_inputs call counter (debug)
_V4_GATE_CNT = {}        # load-bearing per-rank timing crutch (see build_v4_dsa_inputs)

# dsa_kv pool tensors (compress_kv / indexer_k / indexer_scale), registered
# per cr by allocate_v4_caches so the dsa_kv allocator's zero-on-free can
# reach them. The dsa_kv_bt is per-cr SHARED (block j indexes the same
# virtual block across all same-cr layers), so a freed block j must be
# zeroed in EVERY same-cr layer's tensors. One entry per same-cr layer on
# this rank: (compress_kv, indexer_k, indexer_scale). indexer_k/scale are
# None on c128 layers (c4-only). Populated once per worker at cache alloc.
_DSA_KV_TENSORS = {}      # cr -> list[(compress_kv, indexer_k, indexer_scale)]


def _make_dsa_kv_zero_fn(cr):
    """Build the zero-on-free callback for the dsa_kv (compress_kv+indexer_k)
    pool of compress-ratio ``cr``. Zeros the freed block ids across every
    same-cr layer's tensors so the next owner reads zeros, not a prior
    request's stale compressed KV (the 0904 cross-request contamination)."""
    def _zero(ids):
        layers = _DSA_KV_TENSORS.get(cr)
        if not layers or not ids:
            return
        for compress_kv, indexer_k, indexer_scale in layers:
            # Advanced-index dim 0 (block id) -> zero the whole block.
            # ids is a small python list (a few blocks per step); the index
            # tensor creation is negligible vs the graph replay it precedes.
            compress_kv[ids] = 0
            if indexer_k is not None:
                indexer_k[ids] = 0
            if indexer_scale is not None:
                indexer_scale[ids] = 0
    return _zero


def _get_state_alloc(cr, state_bs, max_cols, pool_size, max_num_seqs, device):
    a = _STATE_ALLOCS.get(cr)
    if (a is None or a.max_cols != max_cols or a.pool_size != pool_size
            or a.max_num_seqs != max_num_seqs):
        a = _V4StateAlloc(state_bs, max_cols, pool_size, max_num_seqs, device,
                         state_window=V4_STATE_WINDOW)
        _STATE_ALLOCS[cr] = a
        if os.environ.get('V4_DEBUG_ALLOC', '0') == '1':
            print(f'[V4-STATE-ALLOC] cr={cr} bs={state_bs} pool={pool_size} '
                  f'max_cols={max_cols} max_num_seqs={max_num_seqs} '
                  f'state_window={a.state_window}', flush=True)
    return a


def _get_dsa_kv_alloc(cr, max_cols, pool_size, max_num_seqs, device):
    """Slot-stable append-only allocator for the compress_kv+indexer_k SHARED
    block table (the dsa_kv_bt). Mirrors _get_state_alloc but:
      * state_bs = MLA_BS (these caches are 128-token paged, not state_bs).
      * state_window = 0 (append-only FULL history -- the indexer's top-k
        selects across the whole sequence; evicting an old block drops
        retrievable history AND, under the old swa_block_table alias, the
        freed physical block was reused cross-request -> the 0903
        contamination. No reclaim.)
      * keyed ('dsa_kv', cr) so it never collides with the state allocs.
    Reuses _V4StateAlloc verbatim (it is a generic slot-stable paged
    allocator; the state pool already runs it cleanly in FULL-graph).
    zero-on-free IS wired (zero_fn=_make_dsa_kv_zero_fn(cr)): the earlier
    write-before-read self-cleanse assumption (below) was refuted by the
    0904 contamination -- it only covers positions the CURRENT request
    writes during its own prefill+decode, but the HCA/indexer READS
    historical positions via dsa_kv_bt; a recycled block's unwritten
    positions still hold a prior request's data -> foreign-text bleed
    (accuracy decayed 71%->33% as the pool filled). Zeroing freed blocks
    guarantees the next owner reads zeros, not stale data. Zero HBM cost
    (the dsa_kv pool sizing was HBM-blocked: cards run ~64.6/65.5 GB at
    idle, so pool-sizing to ~8192 was infeasible)."""
    key = ('dsa_kv', cr)
    a = _STATE_ALLOCS.get(key)
    if (a is None or a.max_cols != max_cols or a.pool_size != pool_size
            or a.max_num_seqs != max_num_seqs):
        a = _V4StateAlloc(MLA_BS, max_cols, pool_size, max_num_seqs, device,
                         state_window=0, zero_fn=_make_dsa_kv_zero_fn(cr))
        _STATE_ALLOCS[key] = a
        if os.environ.get('V4_DEBUG_ALLOC', '0') == '1':
            print(f'[V4-DSA-KV-ALLOC] cr={cr} bs={MLA_BS} pool={pool_size} '
                  f'max_cols={max_cols} max_num_seqs={max_num_seqs} '
                  f'state_window=0 (append-only full-history)', flush=True)
    return a


def _build_sas_metadata(cfg, cr, seq_len, query_start_loc, seq_lens,
                        kv_len, device, is_decoding=False,
                        max_seqlen_q=None, max_seqlen_kv=None,
                        in_graph=False):
    """Build sparse_attn_sharedkv metadata for a layer (prefill or decode).

    Mirrors vllm-ascend dsa_v1.py's three metadata_op branches:
      * cr<=1 (SWA):  cmp_ratio=1, has_cmp_kv=False, no cmp_topk/mask
      * cr==4 (CSA):  cmp_ratio=4, has_cmp_kv=True, cmp_topk=index_topk,
                      cmp_mask_mode=3
      * cr==128 (HCA): cmp_ratio=128, has_cmp_kv=True, cmp_mask_mode=3
    TP-local q heads, scalar int max_seqlens, and batch_size / seqused derived
    from the actual per-request tensors (warmup + serve run batched decode, so a
    hardcoded single-request assumption trips AICPU validation, retCode 0x2a).

    max_seqlen_q / max_seqlen_kv are the SAME across all layers (they are
    per-request q/kv maxima). The caller (build_v4_dsa_inputs) already computes
    them once as seq_len/kv_len; passing them in avoids 2 `.max().item()` host
    syncs PER LAYER (86 syncs/step at 43 layers) -- the dominant cost of
    build_step_context (~406ms/decode step was host sync, not NPU compute).
    Falls back to the per-call `.max().item()` only if not supplied (standalone
    use / prefill safety).
    """
    from lmdeploy.pytorch.distributed import get_tp_world_rank
    try:
        _tp_ws, _ = get_tp_world_rank('attn')
    except Exception:
        _tp_ws = 1
    _tp_ws = _tp_ws or 1
    n_local_heads = cfg.num_attention_heads // _tp_ws

    num_reqs = max(query_start_loc.numel() - 1, 1)
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    if max_seqlen_q is None:
        max_seqlen_q = int(q_lens.max().item()) if q_lens.numel() else int(seq_len)
    kv_slice = seq_lens[:num_reqs] if seq_lens.numel() >= num_reqs else seq_lens
    if max_seqlen_kv is None:
        max_seqlen_kv = int(kv_slice.max().item()) if kv_slice.numel() else int(kv_len)

    has_cmp = cr > 1
    eff_ratio = cr if has_cmp else 1
    # vllm-ascend passes cu_seqlens_ori_kv = query_start_loc for prefill
    # (q_len==kv_len, so it IS the kv cumsum). For PA_ND paged decode the op
    # expects an EMPTY cu_seqlens (vllm non-A5/A2 path: torch.tensor([]),
    # dsa_v1.py:466). The op is NOT ignoring it: feeding a REAL decode cumsum
    # cat([0, cumsum(seq_lens)]) (the vllm A5 path, device_op.py:1726) at decode
    # on A2 makes the op take a different code path that aicore-OOBs at capture
    # (verified on-device 2026-08-19). So decode cu_seqlens_ori_kv MUST be
    # empty-semantics (numel==0), not a real cumsum.
    #
    # The problem: a bare torch.empty(0) (0-element storage) is replay-unsafe --
    # the aclnn op's Contiguous preprocessing dereferences the data_ptr, and a
    # 0-elem storage is null/reclaimable. vllm's NPUTaskGroup handles 0-elem
    # external tensors; dlinfer does not. The eager path (in_graph=False) keeps a
    # bare empty (fine, no replay).
    #
    # FIX (step2 OOB root cause): the 1-elem 0-VIEW must be backed by a
    # PERSISTENT EXTERNAL tensor, NOT a graph-pool alloc. A graph-pool
    # torch.zeros(1,...)[0:0] created in-graph is RECLAIMED by dlinfer's
    # AscendSingleGraphRunner pool between replays -> the 0-view's data_ptr
    # goes null/stale -> the op's OpParamMaker/aclrtLaunchKernelWithHostArgs
    # rejects the null ptr at the 2nd replay (507011 launch failure; the
    # async aicore MTE variant surfaces at the next .cpu() sync). Step 1
    # works because the backing is still valid on the first replay; step 2
    # OOBs once it's reclaimed. vllm-ascend avoids this by holding
    # cu_seqlens_ori_kv as a PERSISTENT module attr (self.cu_seqlens_ori_kv
    # = torch.tensor([]), dsa_v1.py:466) -- an EXTERNAL tensor the graph
    # pool never owns/reclaims. The earlier "external -> capture MTE"
    # conclusion was conflated with the seqused_kv=1 temp-sizing issue
    # (now fixed by the sas_seqused_kv ceiling), so a persistent external
    # backing is now safe AND replay-stable. Allocated ONCE in the eager
    # pre-step (build_v4_dsa_inputs) so it is never graph-pool owned; the
    # in-graph op reads its 0-view (data_ptr == persistent backing, stable
    # across all replays). numel==0 preserves correct A2 empty decode
    # semantics (vllm non-A5 path).
    if is_decoding:
        if in_graph:
            _csl = getattr(_DECODE_META, 'sas_cu_seqlens', None)
            if _csl is None:
                _csl = torch.zeros(1, dtype=torch.int32, device=device)
                _DECODE_META.sas_cu_seqlens = _csl
            cu_seqlens_ori_kv = _csl[0:0]
        else:
            cu_seqlens_ori_kv = torch.empty(0, dtype=torch.int32, device=device)
    else:
        cu_seqlens_ori_kv = query_start_loc
    # FULL-graph C1 step2b fix: the SAS op sizes its INTERNAL executor temps
    # by the capture-time seqused_kv VALUE (not max_seqlen_kv -- baking
    # max_seqlen_kv=ceiling alone did NOT stop the replay OOB). vllm-ascend
    # captures with seqused_kv = SEQ_LEN_WITH_MAX_PA_WORKSPACE (=6144) so the
    # aclnn op's temps are max-sized; dlinfer captures with seqused_kv=1
    # (decode warmup) -> temps sized for 1 -> aicore OOB at the 1st replay
    # (real kv_len=6 > 1). The attention op is unaffected: it sizes by
    # max_seqlen_kv=ceiling (already replay-safe -- proven by the 2a eager-SAS
    # path where the attention op captured at kv_len=1 replays at 6/7/...).
    # So the SAS op's seqused_kv is a SEPARATE pinned buffer built by
    # build_v4_dsa_inputs (_DECODE_META.sas_seqused_kv_view): =ceiling at
    # capture (sizing only; the SAS metadata is kv_len-invariant -- vllm
    # computes it once and reuses, so the capture-time ceiling value is
    # discarded), =real per-req kv_len at replay (the op recomputes metadata
    # each replay from the refreshed buffer, fitting in the ceiling-sized
    # temps). It CANNOT reuse seq_lens (=kv_slice): the attention op reads
    # seq_lens for the real KV read, so inflating it would OOB the attention
    # op's KV read at capture.
    _sas_sv = getattr(_DECODE_META, 'sas_seqused_kv_view', None)
    if in_graph and is_decoding and _sas_sv is not None:
        seqused_kv = _sas_sv[:num_reqs] if _sas_sv.numel() >= num_reqs else _sas_sv
    else:
        seqused_kv = kv_slice.to(torch.int32)
    kw = dict(
        device='npu',
        num_heads_q=n_local_heads,
        num_heads_kv=1,
        head_dim=cfg.head_dim,
        cu_seqlens_q=query_start_loc,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=None,
        seqused_q=q_lens.to(torch.int32),
        seqused_kv=seqused_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        batch_size=num_reqs,
        cmp_ratio=eff_ratio,
        ori_mask_mode=4,
        ori_win_left=cfg.sliding_window - 1,
        ori_win_right=0,
        layout_q='TND',
        layout_kv='PA_ND',
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    if has_cmp:
        kw['cmp_mask_mode'] = 3
        if cr == 4:
            kw['cmp_topk'] = cfg.index_topk
    _sas_md = dllm.sparse_attn_sharedkv_metadata(**kw)
    # V4_SAS_INVARIANCE: on the WORKING eager path (in_graph=False), fingerprint
    # the SAS metadata tensor at decode step-1 (kv_len=6) vs step-2 (kv_len=7)
    # per cr, to settle whether the metadata is kv_len-invariant. If invariant
    # (same shape/values), in-graph capture at a session_len ceiling SHOULD be
    # replay-safe and the step-2 OOB is a fixable address/stale issue; if the
    # VALUES differ with kv_len, the captured aclnn op must re-read the
    # refreshed seqused_kv at replay (else in-graph SAS is impossible). rank0
    # (attn-tp + dp) only, first 2 decode steps per cr, then auto-disable.
    import os as _os2
    if (not in_graph and is_decoding
            and _os2.environ.get('V4_SAS_INVARIANCE', '0') == '1'):
        try:
            _tp_ws, _tp_rk = get_tp_world_rank('attn')
            from lmdeploy.pytorch.distributed import get_dist_manager
            _dp_rk = get_dist_manager().current_context().dist_config.dp_rank \
                if hasattr(get_dist_manager().current_context(), 'dist_config') else 0
        except Exception:
            _tp_ws, _tp_rk, _dp_rk = 1, 0, 0
        if _tp_rk == 0:
            if not hasattr(_build_sas_metadata, '_inv_cntr'):
                _build_sas_metadata._inv_cntr = {}
            _cntr = _build_sas_metadata._inv_cntr
            _key = (cr, max_seqlen_kv)
            # skip warmup (kv_len==1): only fingerprint REAL decode steps so the
            # cap covers kv_len=6,7,... (the in-graph OOB is at step2=kv7).
            if int(kv_len) > 1:
                _cntr[_key] = _cntr.get(_key, 0) + 1
            if int(kv_len) > 1 and _cntr[_key] <= 4:
                try:
                    _fp = (f'shape={tuple(_sas_md.shape)} dtype={_sas_md.dtype} '
                           f'numel={_sas_md.numel()} '
                           f'sum={float(_sas_md.float().sum().item()):.4f} '
                           f'min={float(_sas_md.float().min().item())} '
                           f'max={float(_sas_md.float().max().item())} '
                           f'head0={_sas_md.reshape(-1)[:8].cpu().tolist()} '
                           f'kv_len={int(kv_len)} msl_kv={int(max_seqlen_kv)}')
                    print(f'[V4-SASINV] cr={cr} call#{_cntr[_key]} {_fp}',
                          flush=True)
                except Exception as _e:
                    print(f'[V4-SASINV] cr={cr} call#{_cntr[_key]} fp-fail {_e}',
                          flush=True)
    return _sas_md


def _build_qli_metadata(cfg, query_start_loc, seq_lens, max_seqlen_q,
                        max_seqlen_k, device):
    """Build the lightning-indexer metadata op output.

    Factored out of build_v4_dsa_inputs so the SAME call can be made either
    eager (pre-step) or IN the captured forward (FULL-graph C1 step2b:
    V4_META_IN_GRAPH=1 defers the op to the attention forward, where the
    pinned query_start_loc/seq_lens tensors are graph inputs). Layer-
    INDEPENDENT (depends only on cfg + the shared seq tensors), so it is
    cacheable/deducible across c4 layers.
    """
    _nr = max(query_start_loc.numel() - 1, 1)
    _kv = (seq_lens[:_nr] if seq_lens.numel() >= _nr else seq_lens)
    return dllm.lightning_indexer_metadata(
        num_heads_q=cfg.index_n_heads, num_heads_k=1,
        head_dim=cfg.index_head_dim,
        query_quant_mode=0, key_quant_mode=0,
        actual_seq_lengths_query=query_start_loc[1:].clone(),
        actual_seq_lengths_key=_kv.to(torch.int32).clone(),
        batch_size=_nr,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_query='TND', layout_key='PA_BSND',
        sparse_count=cfg.index_topk, sparse_mode=3,
        pre_tokens=INT_MAX, next_tokens=INT_MAX, cmp_ratio=4,
        device='npu')


def _build_compress_cos_sin_for_len(max_pos, cfg, device, dtype):
    return build_full_compress_cos_sin(
        max_pos,
        rotary_dim=cfg.qk_rope_head_dim,
        base=cfg.compress_rope_theta,
        factor=cfg.rope_scaling['factor'],
        original_max=cfg.rope_scaling['original_max_position_embeddings'],
        beta_fast=cfg.rope_scaling['beta_fast'],
        beta_slow=cfg.rope_scaling['beta_slow'],
        device=device, dtype=dtype)


def build_v4_dsa_inputs(step_context, v4_caches, model_config):
    """Build the per-layer DSA input dicts from a step context.

    Mirrors the single-NPU test's ``build_swa/c4/c128_layer_inputs`` but
    derives sequence layout from ``step_context`` and reuses the engine-paged
    ``swa_kv`` (``kv_caches[layer][0]``) plus the persistently-allocated extra
    caches in ``v4_caches``.

    Returns a list (len=num_layers) of dicts compatible with
    ``DeepseekV4DecoderLayer.forward``.
    """
    hf_config = model_config.hf_config
    ratios = _layer_compress_ratios(hf_config)
    device = step_context.block_offsets.device
    dtype = model_config.dtype

    # sequence layout (single request assumed)
    is_decoding = step_context.is_decoding
    q_seqlens = step_context.q_seqlens
    kv_seqlens = step_context.kv_seqlens
    q_start_loc = step_context.q_start_loc
    block_offsets = step_context.block_offsets
    position_ids = step_context.position_ids

    # FULL-graph (V4_FULL_GRAPH_DECODE=1): once a graph has been captured
    # (replay path -- attn_metadata.kv_seqlens is a DEVICE graph-input buffer
    # refreshed by fill_buffers_cudagraph), compute the SHARED decode metadata
    # (query_start_loc / seq_lens / cos+sin gather / slot_mapping /
    # swa_block_table / start_pos) IN-GRAPH inside the captured
    # DeepseekV4Model.forward (build_v4_decode_meta_in_graph) instead of
    # eagerly here. This eliminates the ~26 per-step eager aclnn racers (the
    # cos gather + slot_mapping construction + _pin_meta copy_ ops) that
    # overlap the graph replay's CANN workspace -> 507018 timing race. At
    # replay the captured graph replays the in-graph ops (graph-pool stable
    # addresses, no _pin needed) and the Python dsa_inputs are not re-read,
    # so the None sentinels are harmless.
    #
    # The elision is gated on ``global_is_decoding`` -- the EXACT replay
    # discriminator: ``DeepseekV4ForCausalLM.support_cuda_graph`` returns
    # ``bool(context.global_is_decoding())``, and ``AscendGraphRunner.__call__``
    # replays iff that is True. So global_is_decoding=True  -> this step WILL
    # replay -> attn_metadata.kv_seqlens is the device graph-input buffer ->
    # the model forward's in-graph override fires and fills the None sentinels.
    # global_is_decoding=False -> a mixed DP step (one group prefilling, the
    # other decoding-dummy) -> the step falls back to ``forward_eager``
    # (CPU kv_seqlens, NO fill_buffers) -> the in-graph override does NOT fire
    # (its gate is kv_seqlens.device=='npu'), so build MUST compute the eager
    # metadata here, else the rotary op receives a null cos (EZ1001). The
    # has_captured/capturing latch additionally skips elision before any graph
    # exists (pre-capture eager warmup, where kv_seqlens is also CPU and
    # in-graph wouldn't fire either). callonce SAS/QLI OPs are unaffected
    # (step-invariant, cached cross-step, 0 ops at steady state).
    _full_graph = False
    _v4_fg_env = __import__('os').environ.get('V4_FULL_GRAPH_DECODE', '0')
    # one-shot diagnostic: write the gating decision to a per-rank file on the
    # first 8 build_v4_dsa_inputs calls (any is_decoding), to confirm
    # reachability + the env value + the replay discriminator + kv device.
    # Per-rank files bypass ray stdout redirection; ALL ranks write (rank0
    # alone may be the driver, not a model worker).
    # NOTE: this block is load-bearing for sampling stability, not just
    # diagnostic -- its per-layer host work (get_rank/is_available/
    # is_initialized + dict ops) shifts the eager sampling op's timing
    # relative to the aclgraph replay's CANN workspace allocation just enough
    # to dodge the nondeterministic 507035 vector-core racer on the EP rank.
    # Removing it re-surfaces the crash (2/2 probe-removed runs crashed vs
    # 5/5 probe-present). Keep until the perf-plan FULL graph (0 eager ops
    # between replays) makes the timing fragility moot.
    try:
        import torch.distributed as _d
        _rk = (_d.get_rank() if (_d.is_available()
                                 and _d.is_initialized()) else -1)
        _key = f'rk{_rk}'
        _V4_GATE_CNT[_key] = _V4_GATE_CNT.get(_key, 0) + 1
        if _V4_GATE_CNT[_key] <= 8:
            _gd = 'n/a'
            try:
                _gd = bool(step_context.global_is_decoding())
            except Exception as _e:
                _gd = f'err:{_e}'
            try:
                from dlinfer.framework.lmdeploy_ext.cudagraph.\
                    ascend_cudagraph import AscendGraphRunner as _AGR_p
                _hc = getattr(_AGR_p, 'has_captured', False)
                _cap = getattr(_AGR_p, 'capturing', False)
            except Exception as _e:
                _hc = _cap = f'imp-fail:{_e}'
            _msg = (f'[V4-GATE-{_V4_GATE_CNT[_key]}] rk={_rk} '
                    f'fg_env={_v4_fg_env} is_decoding={is_decoding} '
                    f'global_decoding={_gd} '
                    f'has_captured={_hc} capturing={_cap} '
                    f'kv_dev={step_context.kv_seqlens.device.type} '
                    f'num_reqs={step_context.q_seqlens.numel()}\n')
            with open(f'/tmp/v4_gate_rk{_rk}.log', 'a') as _f:
                _f.write(_msg)
    except Exception:
        pass
    if _v4_fg_env == '1' and is_decoding:
        # Elide on EVERY post-capture decode step (replay AND forward_eager
        # mixed-decode). Safe because op_backend's V4-decode fast path now sets
        # attn_metadata.kv_seqlens to the DEVICE step_context.kv_seqlens (no
        # .cpu()), so the model-forward in-graph gate
        # (attn_metadata.kv_seqlens.device=='npu') fires on forward_eager too
        # -> the None cos/slot_mapping/block_table sentinels are always filled
        # in-graph (no EZ1001). Eliding on mixed-decode steps avoids the
        # build-side eager-cos racer that the global_is_decoding-only gate
        # would have introduced there. `capturing` latches capture-warmup
        # build calls (has_captured is set at end of capture, so pre-capture
        # build computes eager cos; the capture forward's in-graph override
        # then re-computes+captures it).
        try:
            from dlinfer.framework.lmdeploy_ext.cudagraph.\
                ascend_cudagraph import AscendGraphRunner as _AGR
            if getattr(_AGR, 'has_captured', False) or \
                    getattr(_AGR, 'capturing', False):
                _full_graph = True
        except Exception:
            pass


    # per-request state (persist on the module for the active request)
    global _ACTIVE_REQ
    if _ACTIVE_REQ is None:
        _ACTIVE_REQ = _V4ReqState()

    # multi-request sequence layout (prefill AND decode). The engine runs
    # batched prefill (warmup sends 4 reqs x 1 tok) as well as batched decode,
    # so every path must handle N>1. q_seqlens[r]/kv_seqlens[r] are per-req.
    num_reqs = q_seqlens.numel()
    import os as _os
    _state_paged = _os.environ.get('V4_STATE_PAGED', '0') == '1'
    # FULL-graph steady-replay gate. On a step where ALL of:
    #  (1) _full_graph (post-capture decode replay, see L686-770),
    #  (2) callonce SAS+QLI metadata caches are HIT (populated at capture +
    #      the first step of each batch size -- steady replay reuses them, 0
    #      op calls/step; the MISS paths _build_sas_metadata/_build_qli_metadata
    #      that read query_start_loc_l/seq_lens_l are skipped),
    #  (3) the incremental state tracker is valid (stable decode batch, every
    #      request slot cached -> kv_lens_cpu = _slot_kv[r]+1, no .tolist()
    #      host sync; a slot-recycle / batch-change step is NOT all-hit ->
    #      eager tensors computed, no replay race),
    # the eager q_lens/kv_lens (2 int32->int64 casts) + query_start_loc_l
    # (cumsum+cat+cast) + seq_lens_l (cast) -- 5 aclnn ops/step -- are DEAD:
    # they feed only the callonce MISS paths (skipped), _pin_meta (skipped
    # under _full_graph), and the dsa_inputs dicts (seq_lens/query_start_loc
    # are overridden IN-GRAPH by build_v4_decode_meta_in_graph). Those 5 ops
    # overlapped the graph replay CANN workspace -> 507018/EZ9999 timing race.
    # Skip them on all-hit steps; compute on any other step (capture,
    # batch-change, slot-recycle, prefill, callonce miss) where they are needed
    # and no replay races. Mirrors vllm-ascend FULL graph (0 eager ops between
    # replays).
    _meta_callonce_fg = _os.environ.get('V4_META_CALLONCE', '0') == '1'
    _max_q_chk = getattr(step_context, 'max_q_seqlen', None)
    # FULL-graph steady-replay gate. On a step where ALL of:
    #  (1) _full_graph (post-capture decode replay, see L686-770),
    #  (2) callonce SAS+QLI metadata caches are HIT (populated at capture +
    #      the first step of each batch size -- steady replay reuses them, 0
    #      op calls/step; the MISS paths _build_sas_metadata/_build_qli_metadata
    #      that read query_start_loc_l/seq_lens_l are skipped),
    # the eager q_lens/kv_lens (2 int32->int64 casts) + query_start_loc_l
    # (cumsum+cat+cast) + seq_lens_l (cast) -- 5 aclnn ops/step -- are DEAD:
    # they feed only the callonce MISS paths (skipped), _pin_meta (skipped
    # under _full_graph), and the dsa_inputs dicts (seq_lens/query_start_loc
    # are overridden IN-GRAPH by build_v4_decode_meta_in_graph). state_bt is
    # the slab arange (cached, invariant -- 0 ops steady) under the slab path,
    # so it needs no per-step tracking. Those 5 ops overlapped the graph replay
    # CANN workspace -> 507018/EZ9999 timing race. Skip them on all-hit steps;
    # compute on any other step (capture, batch-change, prefill, callonce miss)
    # where they are needed and no replay races. Mirrors vllm-ascend FULL graph
    # (0 eager ops between replays).
    _meta_all_hit = (
        _full_graph and _meta_callonce_fg and is_decoding
        and _max_q_chk == 1 and num_reqs > 0
        and _os.environ.get('V4_CPU_SEQLENS', '0') == '1'
        and step_context.max_kv_seqlen is not None
        and all((cr, num_reqs) in _SAS_XCACHE for cr in ratios)
        and (num_reqs in _QLI_XCACHE if 4 in ratios else True))
    if _meta_all_hit:
        q_lens = kv_lens = None
    else:
        q_lens = q_seqlens.to(torch.long)        # [N] query tokens per req
        kv_lens = kv_seqlens.to(torch.long)      # [N] total kv tokens per req
    # V4_STATE_PAGED: one CPU sync per step (before the per-layer loop) for the
    # paged state allocator's monotonicity check (kv_len grows during a
    # request's life; a slot whose kv shrank was freed+reused -> free+realloc).
    # Computed once here because build_v4_dsa_inputs runs once per step and
    # loops over layers below reusing kv_lens.
    _state_paged = _os.environ.get('V4_STATE_PAGED', '0') == '1'
    # V4 paged-state allocator needs per-request kv_lens as CPU ints (Python
    # control flow: reuse detection kv<prev + block grow ceil(kv/state_bs)).
    # The device kv_lens.tolist() is a host sync that races the captured graph
    # replay (507018/507035 timing race) -- the LAST host sync on the
    # decode-replay path. Eliminate it via INCREMENTAL tracking:
    #   - during steady DECODE (q_seqlens==1 per req -> max_q_seqlen==1), each
    #     request's kv grows by exactly 1/step. The allocator already caches
    #     _slot_kv[r] (last step's kv). So new_kv[r] = _slot_kv[r] + 1 is EXACT,
    #     computed on CPU from the cached ints -- no device sync.
    #   - this holds for stable-batch replay (the racing path): a batch change
    #     (new/recycled slot, num_reqs change) is NOT the captured graph -> runs
    #     eager (forward_eager / prefill), where a sync is safe (no replay to
    #     race) and re-seeds _slot_kv for the next replay steps.
    #   - prefill (max_q_seqlen>1) or any slot absent from the cache -> fall back
    #     to the device .tolist() (eager, no replay race).
    # _slot_kv lives on each cr's _V4StateAlloc; all cr share the same slots
    # (slot r == request r) and the same kv, so any cr's cache is a valid source.
    _kv_src = 'no_paged'   # which kv_lens_cpu source fired (probe)
    if _state_paged:
        _max_q = getattr(step_context, 'max_q_seqlen', None)
        _inc_alloc = next(iter(_STATE_ALLOCS.values()), None)
        # seq_id-keyed kv cache (the robust path) when the scheduler threaded
        # per-req seq_ids; else fall back to the legacy slot-keyed _slot_kv.
        # The slot index is NOT stable across reindex() compaction, so the
        # sid-keyed map is the source of truth whenever available.
        _seq_ids = getattr(step_context, 'seq_ids', None)
        _use_sid = (_seq_ids is not None and len(_seq_ids) >= num_reqs)
        if (os.environ.get('V4_DEBUG_ALLOC', '0') == '1' and not is_decoding
                and num_reqs <= 4):
            import torch.distributed as _d2c_sid
            _rk_sid = (_d2c_sid.get_rank() if (_d2c_sid.is_available()
                      and _d2c_sid.is_initialized()) else 0)
            if _rk_sid == 0:
                print(f'[V4-SID] rank={_rk_sid} num_reqs={num_reqs} '
                      f'use_sid={_use_sid} seq_ids={_seq_ids} '
                      f'is_decoding={is_decoding}', flush=True)
        if _inc_alloc is not None:
            _slot_kv = (_inc_alloc._req_kv if _use_sid else _inc_alloc._slot_kv)
        else:
            _slot_kv = {}
        _keyof = (lambda r: _seq_ids[r]) if _use_sid else (lambda r: r)
        # CPU-side per-request kv lengths threaded from the scheduler
        # (StepContext.kv_seqlens_cpu, mirrors vllm-ascend _seq_lens_cpu).
        # Eliminates the device kv_seqlens.tolist() D2H sync (the 507035
        # racer) on slot-miss / batch-change steps. None on the
        # prefill->first-decode transition (get_model_inputs_next_decoding /
        # merge_model_inputs do not populate it) -> falls to .tolist()
        # below (those steps are eager/safe, no replay to race).
        _cpu_kv = getattr(step_context, 'kv_seqlens_cpu', None)
        _kv_src = 'none'
        if (is_decoding and _max_q == 1 and num_reqs > 0
                and all(_keyof(r) in _slot_kv for r in range(num_reqs))):
            kv_lens_cpu = [int(_slot_kv[_keyof(r)]) + 1 for r in range(num_reqs)]
            _kv_src = 'sid_incr' if _use_sid else 'slot_incr'
        elif _cpu_kv is not None and len(_cpu_kv) >= num_reqs:
            kv_lens_cpu = [int(_cpu_kv[r]) for r in range(num_reqs)]
            _kv_src = 'cpu_list'
        else:
            # kv_lens may be None here: the callonce all-hit gate above
            # nullifies q_lens/kv_lens when the SAS+QLI caches for this
            # num_reqs are populated (steady replay). Use the always-real
            # source tensor kv_seqlens (device) -- this fallback now fires
            # only on prefill or the prefill->first-decode transition (no
            # captured-graph replay to race). Slot-miss / batch-change
            # decode steps take the CPU-list branch above (0 device sync).
            kv_lens_cpu = kv_seqlens.tolist()
            _kv_src = 'dev_tolist'
    else:
        kv_lens_cpu = None
    if _state_paged:
        _PAGED_BT_CACHE.clear()   # per-step: re-assign slots each step
    # FULL (pre-offset, absolute) device kv lens for the COMPRESSOR path
    # (state_bt + dsa_kv_bt). These pools are ABSOLUTE-position-indexed:
    # _V4StateAlloc's table column j = logical block j from the request's
    # start, with the live window [need-W, need) at the RECENT (high) end
    # and older columns nulled; the kernel indexes state_bt[start_pos//bs].
    # dsa_kv_bt (this change) is the same design (state_window=0, full
    # history). So start_pos / indexer_kvlens MUST be the FULL (absolute)
    # position -- the offset kv_lens (= kv_seqlens - num_ignored_history,
    # window-relative) would index the nulled/early region -> the 0903
    # sliding_window breakage (the state path was offset-broken too, not
    # just compress_kv). kv_lens_cpu is the scheduler-side FULL list
    # (num_all_ids + max_q_seqlen, pre-offset). swa keeps the offset
    # kv_lens / seq_lens_l (windowed swa_block_table + seqused_kv mask).
    # Only compute in the not-all-hit branch: the H2D (torch.tensor from a
    # CPU list) would race the steady-replay path, and under all-hit the
    # in-graph path overrides start_pos/indexer_kvlens anyway (so this is
    # unused there). Eager / V4_FULL_GRAPH_DECODE=0 -> _meta_all_hit False
    # -> fires every step (eager has no replay to race; per-step H2D is
    # acceptable, eager is already off-graph).
    if (not _meta_all_hit and kv_lens_cpu is not None
            and len(kv_lens_cpu) >= num_reqs):
        full_kv_lens = torch.tensor(kv_lens_cpu[:num_reqs], device=device,
                                    dtype=torch.long)
    else:
        full_kv_lens = None
    # Option A signal (batch-change -> eager): would THIS decode step, if
    # REPLAYED, run an eager host op that races the captured graph (507035
    # "vector core abnormal")? Historically the racer was the
    # kv_seqlens.tolist() D2H sync (slot absent from _slot_kv on a
    # batch-change / slot-recycle step). Phase 1 replaced that .tolist()
    # with the scheduler-side CPU list (StepContext.kv_seqlens_cpu, 0 device
    # sync), so the .tolist() racer is now gone on slot-miss decode steps.
    # The OTHER eager ops on a batch-change step -- the callonce MISS path
    # (SAS/QLI metadata op + the q_lens/kv_lens casts, _meta_all_hit False)
    # -- still race a replay. So Phase 1 KEEPS Option A routing batch-change
    # / slot-miss / callonce-miss steps eager (double insurance) until
    # Phase 2 makes callonce HIT on those steps (key -> padding) -> 0 eager
    # ops -> Phase 3 turns Option A off (V4_BATCH_CHANGE_EAGER=0) and
    # batch-change steps replay safely. Steady replay (all-hit + all slots
    # cached) -> False -> replay. Capture (_full_graph False) and prefill
    # (is_decoding False) -> False. Computed here in build_context, which
    # runs eager PRE-__call__ on the worker, so the class-attr is fresh
    # when __call__ reads it. Identical inputs across ranks (ray.put once,
    # q_seqlens not dp-sliced) -> identical _slot_kv / _SAS_XCACHE /
    # num_reqs -> rank-uniform decision (no EP-collective graph-entry
    # mismatch, vllm-ascend "min-mode").
    _v4_slot_all_cached = (
        _state_paged and is_decoding and _max_q_chk == 1 and num_reqs > 0
        and all(_keyof(r) in _slot_kv for r in range(num_reqs)))
    _v4_step_route_eager = bool(
        _full_graph and is_decoding
        and (not _meta_all_hit or not _v4_slot_all_cached))
    if _os.environ.get('V4_BC_EAGER_DEBUG', '0') == '1':
        try:
            import torch.distributed as _d_bc
            _rk_bc = (_d_bc.get_rank() if (_d_bc.is_available()
                       and _d_bc.is_initialized()) else 0)
            if _rk_bc in (0, 8):
                global _V4_BC_STEP
                _V4_BC_STEP = _V4_BC_STEP + 1
                print(f'[V4-BCS] step={_V4_BC_STEP} rank={_rk_bc} '
                      f'route_eager={_v4_step_route_eager} '
                      f'full_graph={_full_graph} meta_all_hit={_meta_all_hit} '
                      f'slot_all_cached={_v4_slot_all_cached} num_reqs={num_reqs} '
                      f'kv_src={_kv_src if _state_paged else "no_paged"} '
                      f'is_decoding={is_decoding}',
                      flush=True)
        except Exception:
            pass
    try:
        from dlinfer.framework.lmdeploy_ext.cudagraph.\
            ascend_cudagraph import AscendGraphRunner as _AGR_BC
        _AGR_BC._v4_step_route_eager = _v4_step_route_eager
    except Exception:
        pass
    # Capture-safe runtime probe: log per-rank num_reqs / total_q on BOTH
    # prefill and decode (build_v4_dsa_inputs runs in the engine pre-step,
    # eager, for capture AND replay -- Python does not re-run on replay, so
    # this log fires on the eager pre-step before each replay too). Settles
    # (a) whether internal DP splits the engine batch on prefill, and (b)
    # whether the single-batch capture fix (get_capture_batch_sizes->[mb])
    # makes decode num_reqs constant. The state pool is sized for
    # max_num_seqs; if num_reqs exceeds it the state_bt
    # (arange(1, num_reqs*m_cr+1)) over-indexes the pool -> aicore OOB.
    # Gated on not-capturing so it cannot break graph capture.
    if _os.environ.get('V4_DEBUG_ALLOC', '0') == '1':
        try:
            from dlinfer.framework.lmdeploy_ext.cudagraph.\
                ascend_cudagraph import AscendGraphRunner as _AGR
            if not getattr(_AGR, 'capturing', False):
                import torch.distributed as _d
                _rk = (_d.get_rank() if (_d.is_available()
                                         and _d.is_initialized()) else 0)
                if _rk in (0, 8):
                    print(f'[V4-NREQ] rank={_rk} num_reqs={num_reqs} '
                          f'total_q={int(q_seqlens.sum())} '
                          f'is_decoding={is_decoding}', flush=True)
        except Exception:
            pass
    # cu_seqlens_q = [0, q0, q0+q1, ...] (valid for prefill and decode).
    # Skipped on _meta_all_hit steps (see the gate above): the callonce SAS/QLI
    # ops are cache-hit (don't read these), _pin_meta is skipped under
    # _full_graph, and the dsa_inputs seq_lens/query_start_loc are overridden
    # in-graph. The cumsum+cat+cast (3 aclnn ops/step) are dead racers on hit
    # steps.
    if _meta_all_hit:
        query_start_loc_l = None
        seq_lens_l = None
    else:
        query_start_loc_l = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            q_lens.cumsum(0)]).to(torch.int32)
        # seqused_kv: feed the sparse-attn op the FULL (pre-offset, absolute)
        # per-req kv length -- NOT the window-OFFSET kv_lens (kv_seqlens -
        # num_ignored_history). The op, when cu_seqlens_cmp_kv=None (prefill
        # always, decode when not provided), derives the COMPRESSED-kv read
        # bound from seqused_kv + cmp_ratio. On chunked prefill
        # (prompt > max_prefill_token_num), chunk 2+ has num_ignored_history>0
        # -> the offset kv_lens shrinks at the swa-evict boundary (e.g.
        # 2304 -> 152) -> the c4/c128 cmp read collapses to ~window/cr
        # (38 / 1 compressed tokens) instead of FULL/cr (582 / 18) -> the
        # model loses chunk-1 compressed history -> pt>max_prefill_token_num
        # needle retrieval fails (confused reasoning, no 4271). Chunk 1
        # (num_ignored=0 -> offset==full) is unaffected, so the boundary is
        # exactly at max_prefill_token_num. This mirrors the Stage 1b
        # in-graph decode override (_shared['seq_lens'] = FULL, proven
        # full+windowed-table safe by the 0/5 marker).
        #
        # SWA ori read safety with FULL seqused_kv: the op reads the window
        # from the windowed swa_block_table's RECENT END (the newest block
        # holds the just-scattered tokens; proven by FULL-graph decode,
        # which runs seqused_kv=FULL + a compacted windowed table with 0
        # contamination), so FULL does NOT OOB the windowed block table --
        # seqused_kv drives the cmp-kv bound + the relative sliding-window
        # mask (ori_win_left, position-relative, correct at either value),
        # NOT absolute block-table indexing. The start_pos / indexer_kvlens
        # / state_bt / dsa_kv_bt absolute consumers already use full_kv_lens
        # (fixed above); this closes the last offset-poisoned consumer.
        if full_kv_lens is not None:
            seq_lens_l = full_kv_lens.to(torch.int32)
        else:
            seq_lens_l = kv_lens.to(torch.int32)
        # V4_DEBUG_WINCHECK (0908): DECODE windowing invariant. Logs, rank0
        # every 50 decode steps, the offset (kv_lens, the SWA op's windowed
        # read bound) vs the FULL (full_kv_lens, absolute) kv length. Under
        # sw=128, kv_lens (offset) should track the WINDOW (~capped) while
        # full_kv_lens grows to the full position. If kv_lens grows unbounded
        # toward full, the SWA op reads past the compacted swa_block_table ->
        # drift (sw=128 port bug). Also logs swa_block_table resident cols.
        if _os.environ.get('V4_DEBUG_WINCHECK', '0') == '1' and is_decoding:
            try:
                import torch.distributed as _wc_d
                _wc_rk = (_wc_d.get_rank() if (_wc_d.is_available()
                           and _wc_d.is_initialized()) else 0)
            except Exception:
                _wc_rk = 0
            # Log on TP-rank-0 of EACH DP group (rank % tp == 0): under DP2 a
            # single request is served by one DP rank; rank 0 alone misses
            # requests routed to DP rank 1. rank%8==0 catches both DP0(rk0)
            # and DP1(rk8) -- the TP-0 worker that runs build_v4_dsa_inputs.
            if _wc_rk % 8 == 0:
                _wc_s = getattr(_DECODE_META, '_wincheck_step', 0)
                if _wc_s % 25 == 0:
                    _kv0 = int(kv_lens[0].item()) if kv_lens is not None and kv_lens.numel() else -1
                    _fk0 = int(full_kv_lens[0].item()) if full_kv_lens is not None and full_kv_lens.numel() else -1
                    _bo = getattr(step_context, 'block_offsets', None)
                    _nnz = int((_bo[0] >= 0).sum().item()) if _bo is not None and _bo.numel() else -1
                    _bo_cols = int(_bo.shape[1]) if _bo is not None and _bo.numel() else -1
                    print(f'[V4-WINC] rk={_wc_rk} step={_wc_s} kv_lens(off)[0]={_kv0} '
                          f'full_kv[0]={_fk0} nih(full-off)={_fk0-_kv0} '
                          f'bt_cols={_bo_cols} bt_nnz_row0={_nnz} '
                          f'read_cols_ceil={(_kv0+MLA_BS-1)//MLA_BS if _kv0>0 else 0}',
                          flush=True)
                _DECODE_META._wincheck_step = _wc_s + 1
        # DEBUG: dump chunk-2 prefill tensors to settle the offset-vs-FULL
        # question for the still-failing pt>max_prefill_token_num needle.
        # Bulletproof: no capturing gate (capturing is decode-only; prefill is
        # never captured), no try/except swallow (so a real error surfaces).
        if _os.environ.get('V4_DEBUG_ALLOC', '0') == '1' and not is_decoding:
            import torch.distributed as _d2c
            _rk2c = (_d2c.get_rank() if (_d2c.is_available()
                                     and _d2c.is_initialized()) else 0)
            if _rk2c == 0:
                _fkl = full_kv_lens.tolist() if full_kv_lens is not None else None
                _kvl = kv_lens.tolist() if kv_lens is not None else None
                _sql = seq_lens_l.tolist() if seq_lens_l is not None else None
                _nih_v = (_fkl[0] - _kvl[0]) if (_fkl and _kvl) else None
                print(f'[V4-CHK] rank={_rk2c} prefill num_reqs={num_reqs} '
                      f'total_q={int(q_seqlens.sum())} '
                      f'q_lens={q_lens.tolist()} '
                      f'kv_lens(off)={_kvl} '
                      f'full_kv_lens={_fkl} '
                      f'seq_lens_l={_sql} '
                      f'nih(full-off)={_nih_v} '
                      f'kv_lens_cpu={kv_lens_cpu} '
                      f'meta_all_hit={_meta_all_hit}', flush=True)
    # seq_len/kv_len from CPU-side scheduler ints when V4_CPU_SEQLENS=1: avoids
    # the 2 `.max().item()` host syncs that stall host dispatch behind the prior
    # NPU replay (build_v4_dsa_inputs runs in the eager pre-step). q_seqlens is
    # uniform (= max_q_seqlen per seq) so max(q)=max_q_seqlen; the scheduler
    # tracks kv max as max_kv_seqlen. Fall back to .item() if the CPU ints are
    # not threaded through (older step_context) for safety.
    if _os.environ.get('V4_CPU_SEQLENS', '0') == '1' and \
            getattr(step_context, 'max_q_seqlen', None) is not None and \
            step_context.max_kv_seqlen is not None:
        seq_len = int(step_context.max_q_seqlen)
        kv_len = int(step_context.max_kv_seqlen)
    else:
        seq_len = int(q_lens.max().item())
        kv_len = int(kv_lens.max().item())

    # main RoPE cos/sin over the *current* tokens (positions = position_ids)
    # For prefill, positions are 0..seq_len-1 (single req, no history) or
    # history..history+seq_len-1.  Build cos/sin over the max position covered.
    rotary_dim = hf_config.qk_rope_head_dim
    max_pos_emb = hf_config.rope_scaling['original_max_position_embeddings']
    rope_base = hf_config.rope_theta
    rope_factor = hf_config.rope_scaling['factor']
    beta_fast = hf_config.rope_scaling['beta_fast']
    beta_slow = hf_config.rope_scaling['beta_slow']

    # positions covered by current tokens (absolute)
    cur_positions = position_ids.view(-1)
    nt = cur_positions.numel()
    # V4: c4/c128 layers (compress_ratio > 1) use compress_rope_theta (160000)
    # for the MAIN MLA q/kv/o_proj RoPE too -- NOT rope_theta (10000).
    compress_rope_base = hf_config.compress_rope_theta
    _precompute_rope = (_os.environ.get('V4_PRECOMPUTE_ROPE', '0') == '1')
    if is_decoding and _precompute_rope:
        # Persistent full main-RoPE cos/sin tables, built once on the first
        # decode step and indexed by absolute position each step. Eliminates
        # the per-step build_cos_sin (arange+einsum+cos+sin+repeat_interleave,
        # ~5 ops x2 variants) AND the pos_max .item() host sync. The YaRN
        # inv_freq depends only on original_max_position_embeddings (fixed),
        # so cos[i]=cos(i*inv_freq) is position-deterministic -> a precomputed
        # [ceiling, ...] table indexed by position_ids is identical to building
        # [pos_max] and indexing. Fixed-shape table is also graph-capturable.
        # +1 boundary margin: the in-graph cos gather ``main_cos[pos]`` is
        # bounds-checked by aclnn IndexCheck. The table must cover every
        # absolute position the engine can produce. The engine caps total
        # tokens (prompt+decode) at the *effective* session_len
        # (engine._get_max_session_len = min(configured, GPU-block budget)),
        # so positions never exceed session_len-1; sizing the table to
        # session_len+1 makes position==session_len a valid index (its
        # cos/sin is well-defined), eliminating the off-by-one.
        #
        # ROOT CAUSE this guards: ``cache_config.session_len`` is plumbed
        # by the engine (CacheConfig.session_len). Before that plumbing
        # existed, this ``getattr`` fell back to a HARDCODED 8192 -- so a
        # serve configured with session_len=65536 still sized main_cos to
        # 8192, and a long decode whose position exceeded 8192 OOB'd the
        # gather: plog "Index out of range in dimension 0: index value
        # 8192 exceeds bounds 8192" -> 507011 aicore crash -> whole serve
        # down (the GPQA long-decode crash). Now sized to the real cap.
        #
        # Fallback when session_len isn't plumbed (None, e.g. a config path
        # that bypasses the engine): the model's YaRN original_max_position
        # embeddings -- a bounded model ceiling, never the 8192 undersize.
        _cfg_sess = getattr(step_context.cache_config, 'session_len', None)
        if _cfg_sess:
            _sess = int(_cfg_sess)
        else:
            _sess = int(max_pos_emb) or 8192
        _sess = _sess + 1
        if _DECODE_META.main_rope_len < _sess or _DECODE_META.main_cos is None:
            _DECODE_META.main_cos, _DECODE_META.main_sin = build_cos_sin(
                _sess, rotary_dim, max_pos_emb, rope_base,
                rope_factor, beta_fast, beta_slow, device, dtype)
            _DECODE_META.main_cos_cmp, _DECODE_META.main_sin_cmp = build_cos_sin(
                _sess, rotary_dim, max_pos_emb, compress_rope_base,
                rope_factor, beta_fast, beta_slow, device, dtype)
            _DECODE_META.main_rope_len = _sess
        if _full_graph:
            # cos/sin gather moved IN-GRAPH (build_v4_decode_meta_in_graph
            # in DeepseekV4Model.forward); leave None sentinels for the model
            # forward to fill. The persistent main_cos/sin tables above are
            # still built once (one-time, not a steady-state racer) -- the
            # in-graph gather indexes them.
            cos_cur = sin_cur = cos_cur_cmp = sin_cur_cmp = None
        else:
            # Clamp to table bound -- defense-in-depth so an OOB position
            # never reaches aclnn IndexCheck (see the FULL-graph clamp in
            # build_v4_decode_meta_in_graph for rationale). The engine cap
            # + +1 table sizing already bound pos in normal operation.
            _cp = cur_positions.long()
            _cp = _cp.clamp(min=0, max=_DECODE_META.main_cos.shape[0] - 1)
            cos_cur = _DECODE_META.main_cos[_cp].view(nt, 1, 1, rotary_dim)
            sin_cur = _DECODE_META.main_sin[_cp].view(nt, 1, 1, rotary_dim)
            cos_cur_cmp = _DECODE_META.main_cos_cmp[_cp].view(nt, 1, 1, rotary_dim)
            sin_cur_cmp = _DECODE_META.main_sin_cmp[_cp].view(nt, 1, 1, rotary_dim)
    else:
        pos_max = int(cur_positions.max().item()) + 1 if cur_positions.numel() else seq_len
        pos_max = max(pos_max, seq_len, 16)
        cos, sin = build_cos_sin(pos_max, rotary_dim, max_pos_emb, rope_base,
                                 rope_factor, beta_fast, beta_slow, device, dtype)
        # slice to current token positions (cos/sin indexed by absolute position)
        # size the cos/sin view by the actual number of current tokens — during
        # batched decoding cur_positions holds one entry per request (batch), not 1,
        # so seq_len/q_seqlens[0] (both =1 per decoding seq) would under-count.
        cos_cur = cos[cur_positions.long()].view(nt, 1, 1, rotary_dim)
        sin_cur = sin[cur_positions.long()].view(nt, 1, 1, rotary_dim)

        # V4: c4/c128 layers (compress_ratio > 1) use compress_rope_theta (160000)
        # for the MAIN MLA q/kv/o_proj RoPE too -- NOT rope_theta (10000).
        # vllm-ascend deepseek_v4.py:789-795 sets
        #   config.rope_parameters["rope_theta"] = config.compress_rope_theta
        #   rope_groups = ["default", f"c{compress_ratio}"]
        # for compress_ratio>1, so the "default" (main MLA) group is registered
        # under base=compress_rope_theta and shares that full_rope_cache. SWA
        # layers (ratio 0) keep rope_theta. Using 10000 on c4/c128 layers causes
        # a zero-mean rotary-angle drift that starts at layer 2 (first compressor)
        # and accumulates -- the ~0.75-logit near-tie bug. Build a second main
        # cos/sin with the compress theta; same YaRN scaling (factor/original_max/
        # beta unchanged -- only base differs), dispatch per layer below.
        cos_cmp, sin_cmp = build_cos_sin(pos_max, rotary_dim, max_pos_emb,
                                         compress_rope_base, rope_factor,
                                         beta_fast, beta_slow, device, dtype)
        cos_cur_cmp = cos_cmp[cur_positions.long()].view(nt, 1, 1, rotary_dim)
        sin_cur_cmp = sin_cmp[cur_positions.long()].view(nt, 1, 1, rotary_dim)

    # swa block table + slot mapping (multi-request, prefill AND decode).
    # block_offsets is [num_reqs, max_blocks] (the engine per-request block
    # table). For each flattened query token t in request r, its absolute kv
    # position is (kv_lens[r]-q_lens[r]) + local_idx, and its physical slot is
    # block_table[r][pos//bs]*bs + pos%bs. This single construction covers
    # batched prefill (>=1 new tok/req) and batched decode (exactly 1 tok/req).
    # The swa_kv cache is the engine's paged k_cache; its block_size is the
    # engine's own (== cache.shape[1]), which is NOT guaranteed to equal
    # SWA_BS=MLA_BS. Read it from the actual cache so the slot mapping, the
    # block table, and the op's block indexing all agree on the same
    # granularity -- a mismatch (e.g. 128 here vs 32 in the cache) makes the op
    # read empty blocks and return exact zeros.
    swa_cache = step_context.kv_caches[0][0]
    block_size = int(swa_cache.shape[1])
    # total_q from CPU (q_seqlens uniform = seq_len each -> sum = num_reqs*seq_len)
    # avoids the `.sum().item()` host sync. seq_len is already the CPU
    # max_q_seqlen int under V4_CPU_SEQLENS=1; fall back to NPU sum otherwise.
    # NOTE: the num_reqs*seq_len shortcut is only exact for UNIFORM q_lens
    # (decode: all 1, seq_len=1). Batched prefill with variable q_lens needs
    # q_lens.sum() -- using num_reqs*seq_len there gives num_reqs*max_q_seqlen !=
    # sum(q_lens), which crashes the repeat_interleave/arange indexing below with a
    # dim-0 size mismatch (e.g. 308 vs 299 for q_lens=[154,145]). This block (the
    # non-full-graph branch) only runs for EAGER steps (prefill / first-decode /
    # batch-change-eager), never for full-graph decode replay, so the q_lens.sum()
    # sync here cannot race a captured collective. Guard the shortcut to
    # is_decoding (uniform q_lens=1, exact) so prefill takes the exact sum.
    if (is_decoding
            and _os.environ.get('V4_CPU_SEQLENS', '0') == '1'
            and getattr(step_context, 'max_q_seqlen', None) is not None):
        total_q = num_reqs * seq_len
    else:
        total_q = int(q_lens.sum())
    if _full_graph:
        # slot_mapping / swa_block_table / start_pos moved IN-GRAPH
        # (build_v4_decode_meta_in_graph in DeepseekV4Model.forward); leave
        # None sentinels for the model forward to fill. The decode graph
        # computes these from kv_seqlens + block_offsets (device graph
        # inputs) at fixed max_batches shape, matching the eager shapes the
        # attention op recorded at capture.
        swa_slot_mapping = None
        swa_block_table = None
        start_pos = None
    else:
        req_ar = torch.repeat_interleave(
            torch.arange(num_reqs, device=device), q_lens)            # [total_q]
        q_cum = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                          q_lens.cumsum(0)[:-1]])                    # [N] q starts
        local_idx = torch.arange(total_q, device=device) - q_cum[req_ar]  # [total_q]
        kv_start = (kv_lens - q_lens).clamp(min=0)                   # [N] OFFSET (swa window)
        # compressor start_pos uses the FULL (absolute) kv lens -- state_bt
        # + dsa_kv_bt are absolute-position-indexed (live window at the high
        # end; offset would index the nulled/early region). swa_slot_mapping
        # below keeps the offset kv_start (windowed swa_block_table). This
        # branch only fires when NOT _full_graph -> _meta_all_hit False ->
        # full_kv_lens is computed (not None) above.
        if full_kv_lens is not None:
            full_start = (full_kv_lens - q_lens).clamp(min=0)        # [N] FULL
        else:
            full_start = kv_start
        # compressor start_pos is per-request (size num_reqs): the KV position
        # where compression begins (vllm-ascend: start_pos = seq_lens - q_lens).
        # 0 for a fresh prefill; the current decode position otherwise.
        start_pos = full_start.to(torch.int32)
        kv_pos = kv_start[req_ar] + local_idx                        # [total_q] OFFSET (swa)
        bnum = kv_pos // block_size
        bidx = kv_pos % block_size
        blocks = block_offsets[req_ar, bnum]                         # [total_q]
        swa_slot_mapping = (blocks.to(torch.int64) * block_size + bidx).to(torch.int32)
        # block table for the attention op: full 2D [num_reqs, max_blocks]
        swa_block_table = block_offsets.to(torch.int32)
        # [V4-SWA] swa block-table bounds probe (chunked-prefill crash hunt)
        if (_os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and not is_decoding and num_reqs <= 4
                and (int(q_lens.max().item()) > 1000
                     or (full_kv_lens is not None
                         and int(max(full_kv_lens)) > 1000))):
            import torch.distributed as _d2c_swa
            _rk_swa = (_d2c_swa.get_rank() if (_d2c_swa.is_available()
                       and _d2c_swa.is_initialized()) else 0)
            if _rk_swa == 0:
                _bo_shape = tuple(block_offsets.shape)
                _bnum_max = int(bnum.max().item())
                _sm_min = int(swa_slot_mapping.min().item())
                _sm_max = int(swa_slot_mapping.max().item())
                _bt_max = int(swa_block_table.max().item())
                # how many columns actually hold a real block (< pool size)?
                _pool_nblk = swa_cache.shape[0]
                _r0 = swa_block_table[0]
                _n_real = int(((_r0 >= 0) & (_r0 < _pool_nblk)).sum().item())
                _r0_first = _r0[:min(_bnum_max + 2, _bo_shape[1])].tolist()
                _kv_lens = kv_lens.tolist()
                _q_lens = q_lens.tolist()
                _fkl = (full_kv_lens.tolist() if full_kv_lens is not None else None)
                _ign = ((full_kv_lens - kv_lens).tolist()
                        if full_kv_lens is not None else None)
                print(f'[V4-SWA] nr={num_reqs} q={_q_lens} kv(off)={_kv_lens} '
                      f'fkl={_fkl} ign={_ign} bo_shape={_bo_shape} pool_nblk={_pool_nblk} '
                      f'bnum_max={_bnum_max} bt_max={_bt_max} sm=[{_sm_min},{_sm_max}] '
                      f'n_real_col0={_n_real} r0[:bmax+2]={_r0_first}', flush=True)

    # ---- decode graph-capture: pin per-step metadata into persistent,
    # address-stable buffers (mirror vllm-ascend dsa_v1.py:463-472 +
    # build() in-place refresh). At capture the NPU graph records these
    # buffers' addresses; at replay build_v4_dsa_inputs re-runs in the
    # engine pre-step and copy_'s refreshed contents into the SAME
    # buffers, so replay reads new values from stable addresses (instead
    # of the freed/stale addresses of fresh-per-step tensors -> aicore
    # OOB). Prefill (not is_decoding) keeps fresh tensors -- it is never
    # captured. cos_cur/sin_cur are views into the fresh cos/sin; _pin
    # clones them on first call (standalone buffer) and copy_s from the
    # transient view afterwards.
    if is_decoding:
        # per-request fields: max-size prefix-slice so the captured graph's
        # recorded address stays stable across all decode batch sizes (see
        # _pin). max_dim0 = per-rank max_num_seqs (the capture ceiling);
        # query_start_loc has one extra row (cumsum -> N+1).
        _mr = _DECODE_META.max_num_seqs
        # swa_block_table columns (max blocks per request) GROW with request
        # length (warmup ~1 block, real decode after a 2k prefill ~45). Pin
        # both dims so the captured graph's address + column count stay stable
        # (else the dim1 shape guard reallocates -> stale addr -> OOB). Ceiling
        # = session_len // swa_block_size.
        _sess = int(getattr(step_context.cache_config, 'session_len', 8192))
        _mblk = (_sess + block_size - 1) // block_size
        if not _full_graph:
            # _pin_meta copy_ ops are steady-state racers (overlap the graph
            # replay CANN workspace -> 507018). Under FULL-graph the SHARED
            # metadata is computed IN-GRAPH (graph-pool stable addresses, no
            # pin needed), so skip the per-step pin entirely. query_start_loc/
            # seq_lens are still computed above (cheap, 2 ops) -- needed by the
            # callonce SAS/QLI OP miss at capture; at replay (callonce hit)
            # they are unused (the replayed graph reads the in-graph values).
            query_start_loc_l = _pin_meta('query_start_loc', query_start_loc_l,
                                          max_dim0=(_mr + 1) if _mr else None)
            seq_lens_l = _pin_meta('seq_lens', seq_lens_l, max_dim0=_mr)
            cos_cur = _pin_meta('cos_cur', cos_cur, max_dim0=_mr)
            sin_cur = _pin_meta('sin_cur', sin_cur, max_dim0=_mr)
            cos_cur_cmp = _pin_meta('cos_cur_cmp', cos_cur_cmp, max_dim0=_mr)
            sin_cur_cmp = _pin_meta('sin_cur_cmp', sin_cur_cmp, max_dim0=_mr)
            swa_slot_mapping = _pin_meta('swa_slot_mapping', swa_slot_mapping,
                                         max_dim0=_mr)
            swa_block_table = _pin_meta('swa_block_table', swa_block_table,
                                        max_dim0=_mr, max_dim1=_mblk)
            start_pos = _pin_meta('start_pos', start_pos, max_dim0=_mr)
        if _DECODE_META.sas_metadata is None:
            _DECODE_META.sas_metadata = [None] * len(ratios)
            _DECODE_META.qli_metadata = [None] * len(ratios)
        # TEMP decode probe: log PINNED data_ptr (must be stable across steps)
        # + range values to find which address/value strays at the step-2
        # SparseAttnSharedkv OOB. Gated not-capturing (real decode replay only).
        if __import__('os').environ.get('V4_DEBUG_ALLOC', '0') == '1':
            try:
                from dlinfer.framework.lmdeploy_ext.cudagraph.\
                    ascend_cudagraph import AscendGraphRunner as _AGR
                if not getattr(_AGR, 'capturing', False):
                    import torch.distributed as _d
                    if (not _d.is_available() or not _d.is_initialized()
                            or _d.get_rank() == 0):
                        _smx = int(swa_slot_mapping.max().item()) if swa_slot_mapping.numel() else -1
                        _bmx = int(swa_block_table.max().item()) if swa_block_table.numel() else -1
                        print(f'[V4-DEC] mr={_mr} nr={num_reqs} tq={total_q} bs={block_size} '
                              f'slot_max={_smx} bt_max={_bmx} '
                              f'sl_ptr={swa_slot_mapping.data_ptr()} '
                              f'bt_ptr={swa_block_table.data_ptr()} '
                              f'ql_ptr={query_start_loc_l.data_ptr()} '
                              f'slptr={seq_lens_l.data_ptr()} '
                              f'sp={start_pos.tolist() if start_pos.numel()<=8 else tuple(start_pos.shape)} '
                              f'sl={seq_lens_l.tolist() if seq_lens_l.numel()<=8 else tuple(seq_lens_l.shape)} '
                              f'qsl={query_start_loc_l.tolist() if query_start_loc_l.numel()<=9 else tuple(query_start_loc_l.shape)}',
                              flush=True)
            except Exception:
                pass
        # READ-ONLY tail-stale probe (mode-(b) backings use torch.empty and do
        # NOT zero the tail beyond the active prefix -- unlike mode-(a) which
        # zeroes the padded tail. If the captured graph baked extent=max_num_seqs
        # while replay uses fewer rows, the kernel may read these stale tail
        # values. This probe prints the tail max vs active max to confirm/refute
        # a stale-tail OOB hazard correlating with crashes. Pure observation:
        # no computation change.
        if __import__('os').environ.get('V4_DEBUG_ALLOC', '0') == '1':
            try:
                from dlinfer.framework.lmdeploy_ext.cudagraph.\
                    ascend_cudagraph import AscendGraphRunner as _AGR2
                if not getattr(_AGR2, 'capturing', False):
                    import torch.distributed as _d2
                    if (not _d2.is_available() or not _d2.is_initialized()
                            or _d2.get_rank() == 0):
                        _probe = []
                        for _nm, _act in (('swa_slot_mapping', total_q),
                                          ('seq_lens', num_reqs),
                                          ('query_start_loc', num_reqs + 1),
                                          ('start_pos', num_reqs)):
                            _bk = getattr(_DECODE_META, _nm, None)
                            if _bk is not None and _bk.numel() > _act:
                                _amax = int(_bk[:_act].max().item()) \
                                    if _act > 0 else -1
                                _tmax = int(_bk[_act:].max().item()) \
                                    if _bk[_act:].numel() else -1
                                _probe.append(
                                    f"{_nm}:act_max={_amax} tail_max={_tmax}"
                                    f" (tail_len={_bk[_act:].numel()})")
                        if _probe:
                            print('[V4-TAIL] ' + ' | '.join(_probe), flush=True)
            except Exception:
                pass

    # Per-request state block tables for the recurrent state caches are built
    # PER LAYER below (compressor/indexer state -- separate per-cr pools, block
    # size C4/C128_STATE_BS, NOT the main block table; compress_kv/indexer_k use
    # swa_block_table above). Each request r owns contiguous state blocks
    # [r*M_cr, (r+1)*M_cr) in its layer-type pool; the op indexes
    # state_block_table[r*stride + j] = r*M_cr + j, so the table is
    # arange(num_reqs*M_cr).view(num_reqs, M_cr). M_cr = cdiv(max_pt//cr, BS) is
    # per-cr: c4 densest (M=72), c128 ~1. Sizing c128 at c4's M (old shared
    # table) wasted ~5.5GB. Built every step (prefill AND decode) so dim0 always
    # equals the current num_reqs (compressor tiling checks
    # state_block_table.dim0 == batchSize). _pin_meta keeps the decode-graph
    # buffers address-stable, keyed per-cr (state_bt_c4 / state_bt_c128).
    # NOTE: this couples a request's state to its batch SLOT, not its identity
    # -- correct while batch composition is stable (the profiling workload: 30
    # reqs prefill together then decode together). General churn needs an
    # engine-style per-request state allocator (follow-up).
    max_pt = int(getattr(step_context.cache_config,
                         'max_prefill_token_num', 2048))

    # FULL-graph precondition (C1 step1): bake max_seqlen_kv to a session_len
    # ceiling when V4_META_CEILING=1 + decoding. vllm-ascend dsa_v1.py bakes
    # max_seqlen_kv at capture and never updates it (update_graph_params is a
    # no-op); the per-step actual kv length flows through the seqused_kv TENSOR
    # input (kv_slice, address-stable via _pin), which the captured metadata
    # kernel re-reads each replay. So max_seqlen_kv must be a FIXED ceiling >=
    # any real kv_len (session_len) for FULL capture -- it may only drive
    # workspace/output sizing, not actual compute ranges. This probe verifies
    # baking does not corrupt output (Paris/Rome/Madrid): if correct, the op
    # treats max_seqlen_kv as pure sizing -> FULL can bake it; if corrupt, the
    # op uses it for actual ranges -> FULL needs per-replay host-arg patching
    # (NPUTaskGroupHandle, which dlinfer already has) instead. max_seqlen_q for
    # decode is always 1 (constant), so it needs no baking.
    _meta_ceiling = (_os.environ.get('V4_META_CEILING', '0') == '1')
    if is_decoding and _meta_ceiling:
        _eff_max_kv = int(getattr(step_context.cache_config, 'session_len', 8192))
    else:
        _eff_max_kv = kv_len
    # FULL-graph C1 step2b: stash the ceiling so the captured attention forward
    # (which builds the metadata op in-graph under V4_META_IN_GRAPH=1) reads
    # the SAME baked ceiling without param threading.
    if is_decoding:
        _DECODE_META.max_kv_ceiling = _eff_max_kv

    # FULL-graph C1 step2a: dedup the per-layer metadata op calls BY
    # compress_ratio WITHIN one step. The SAS metadata op output depends only
    # on (cr, query_start_loc, seq_lens, max_seqlen_q/kv, cfg) -- all
    # LAYER-INDEPENDENT (same across same-cr layers; vllm-ascend caches by
    # layer_name=f"c{compress_ratio}" for the same reason). The 43-layer loop
    # otherwise re-launches the op 43x/step (the dominant eager gap cluster:
    # SparseAttnSharedkvMetadata 43 + VllmQuantLightningIndexerMetadata 21,
    # ~64 ops/step). Caching per-cr per-step (cleared each step, so per-step
    # seqused_kv changes are picked up -- this is NOT a cross-step cache, which
    # would stale the metadata) cuts SAS 43-><=3 and qli ~21->1 calls/step.
    # _pin_list still COPIES the cached output into each layer's address-stable
    # slot. Step2b will move the op itself into the captured forward (0 host
    # dispatch); this step2a keeps it eager but cheap.
    _sas_cache = {}
    _qli_cache = None
    # indexer_kvlens pin is shared across all c4 layers (_pin_meta uses ONE
    # _DECODE_META.indexer_kvlens buffer); the prior per-c4-layer call copied
    # the same value into the same shared buffer ~24x/fwd (idempotent waste).
    # Compute it ONCE per step (first c4 layer) and reuse the view for every
    # c4 layer -- byte-identical to before (all layers already aliased the
    # same shared view; only the redundant re-copies are removed).
    _indexer_kvlens_view = None
    # FULL-graph C1 step2b: when V4_META_IN_GRAPH=1 + decoding, DEFER the
    # metadata op calls to the captured attention forward (build them
    # in-graph -> 0 host dispatch). build_v4_dsa_inputs (eager pre-step)
    # still builds the pinned INPUT buffers (query_start_loc_l, seq_lens_l,
    # block tables) and passes max_kv_ceiling; it just skips the op itself,
    # leaving sas_metadata/qli_metadata=None as a sentinel. The attention
    # forward sees None -> calls the op in-graph (recorded at capture, replayed
    # each step reading the pinned, refreshed inputs). At capture the op is
    # recorded once per call site; at replay Python is frozen so the None
    # sentinel is irrelevant -- the recorded op output is used. V4_META_IN_GRAPH
    # =0 (or prefill) keeps the eager 2a cache-by-cr path below.
    _meta_in_graph = (_os.environ.get('V4_META_IN_GRAPH', '0') == '1')
    # Isolation gate: V4_QLI_IN_GRAPH (default = V4_META_IN_GRAPH). When 0 the
    # lightning-indexer metadata op stays EAGER (built+pinned here) while the
    # SAS op can still be in-graph -- to isolate which aclnn metadata op
    # causes the 2nd-replay 507011. Set =V4_META_IN_GRAPH once isolated.
    _qli_in_graph = (_os.environ.get(
        'V4_QLI_IN_GRAPH', '1' if _meta_in_graph else '0') == '1')
    # Cross-step call-once cache for the SAS AND QLI metadata OPs (mirrors
    # vllm-ascend decode_ratio_to_sas_metadata: compute once per
    # (cr, num_reqs) [SAS] / num_reqs [QLI], reuse across decode steps).
    # Decode-only; prefill inputs vary per step so the ops must run. Both SAS
    # and QLI invariance are on-device verified (vllm caches both in the same
    # call-once dict; see _SAS_XCACHE / _QLI_XCACHE docs).
    _meta_callonce = (_os.environ.get('V4_META_CALLONCE', '0') == '1'
                      and is_decoding and not _meta_in_graph)

    # FULL-graph C1 step2b fix: build the SAS op's SEPARATE pinned seqused_kv
    # buffer (consumed by _build_sas_metadata when in_graph). The SAS op sizes
    # its INTERNAL executor temps by the capture-time seqused_kv VALUE, so
    # capturing at the decode-warmup kv_len (=1) makes temps sized for 1 ->
    # aicore OOB at the 1st replay (real kv_len > 1). vllm-ascend captures with
    # seqused_kv = SEQ_LEN_WITH_MAX_PA_WORKSPACE (=6144) to max-size the temps.
    # Here: =ceiling at graph capture (V4_GRAPH_CAPTURING, set by the
    # model-agent decode-warmup loop -- the only capture site; build_context
    # runs BEFORE graph_runner.capture flips AscendGraphRunner.capturing, so
    # that flag can't gate here), =real per-req kv_len at replay (copied from
    # the seq_lens backing so the padded slots MATCH what the attention op
    # reads -- the SAS metadata is kv_len-invariant so the capture-time ceiling
    # value is discarded, only the temp sizing matters). The attention op is
    # untouched: it keeps reading seq_lens_l (sized by max_seqlen_kv ceiling,
    # replay-safe per the 2a eager-SAS path); inflating seq_lens_l would OOB
    # the attention op's KV read at capture.
    if is_decoding and _meta_in_graph:
        _sas_capturing = _os.environ.get('V4_GRAPH_CAPTURING', '0') == '1'
        # V4_SAS_FREEZE=1 (diagnostic): feed the ceiling at EVERY replay, not
        # just capture. Tests the hypothesis that the aclnn SAS op sizes its
        # internal UB/temps at the FIRST REPLAY's seqused_kv (=6 for step1),
        # not at capture (=8192) -> step1 (6) fits, step2 (7>6) OOBs the UB
        # ("D-cache/UB bus error"). Freezing to ceiling makes the op size for
        # 8192 at every replay -> 7<=8192 -> no OOB. Correctness hold only if
        # the SAS metadata is kv_len-INVARIANT (the op treats seqused_kv as a
        # pure sizing/bound; the real per-req kv_len flows through the attention
        # op's seq_lens, which is NOT frozen). vllm sizes at capture (6144) and
        # re-runs with real; dlinfer appears to size at first replay, so freeze
        # is the dlinfer-portable equivalent of vllm's capture-time max-sizing.
        _sas_freeze = _os.environ.get('V4_SAS_FREEZE', '0') == '1'
        if _sas_capturing or _sas_freeze:
            _sas_sv = torch.full((_mr,), _eff_max_kv,
                                 dtype=torch.int32, device=device)
        else:
            _sas_sv = _DECODE_META.seq_lens[:_mr].to(torch.int32)
        _DECODE_META.sas_seqused_kv_view = _pin_meta(
            'sas_seqused_kv', _sas_sv, max_dim0=_mr)

    dsa_inputs = []
    for layer_idx, cr in enumerate(ratios):
        # swa_kv = engine paged k cache for this layer
        swa_kv_cache = step_context.kv_caches[layer_idx][0]

        if _meta_in_graph and is_decoding:
            # op deferred to the captured attention forward (step2b).
            sas_metadata = None
        else:
            _sas_key = (cr, num_reqs)
            if _meta_callonce:
                # Cross-step call-once (mirrors vllm-ascend
                # decode_ratio_to_sas_metadata): the SAS metadata is
                # kv_len-invariant (on-device verified), so the op output
                # computed at the first decode step of this (cr, num_reqs)
                # batch is reused for every subsequent step -- 0 op calls/step
                # after the first. Keyed by (cr, num_reqs) -> self-invalidates
                # on batch change, consistent with _SAS_PIN_CACHE below.
                if _sas_key in _SAS_XCACHE:
                    sas_metadata = _SAS_XCACHE[_sas_key]
                    _SAS_XC_STATS['hit'] += 1
                else:
                    sas_metadata = _build_sas_metadata(
                        hf_config, cr, seq_len, query_start_loc_l, seq_lens_l,
                        kv_len, device, is_decoding=is_decoding,
                        max_seqlen_q=seq_len, max_seqlen_kv=_eff_max_kv)
                    _SAS_XCACHE[_sas_key] = sas_metadata
                    _SAS_XC_STATS['miss'] += 1
            else:
                # per-step within-step dedup by cr (3 op calls/step, once per
                # unique cr); the op still runs every step.
                if cr not in _sas_cache:
                    _sas_cache[cr] = _build_sas_metadata(
                        hf_config, cr, seq_len, query_start_loc_l, seq_lens_l,
                        kv_len, device, is_decoding=is_decoding,
                        max_seqlen_q=seq_len, max_seqlen_kv=_eff_max_kv)
                sas_metadata = _sas_cache[cr]
            if is_decoding:
                # sas_metadata is step-INVARIANT (kv_len-invariant op output,
                # see _SAS_PIN_CACHE doc). Skip the per-layer _pin_list copy
                # when (num_reqs, cr) is unchanged: the per-layer backing
                # (_DECODE_META.sas_metadata[layer_idx], set at capture, stable
                # address) already holds the byte-identical value. On cache
                # miss (first step / batch change) fall through to _pin_list.
                _sc = _SAS_PIN_CACHE.get(layer_idx)
                if (_sc is not None and _sc[0] == num_reqs
                        and _sc[1] == cr
                        and _DECODE_META.sas_metadata[layer_idx] is not None):
                    sas_metadata = _DECODE_META.sas_metadata[layer_idx]
                else:
                    sas_metadata = _pin_list(_DECODE_META.sas_metadata,
                                             layer_idx, sas_metadata)
                    _SAS_PIN_CACHE[layer_idx] = (num_reqs, cr)

        # 0904 Task #22 root cause + fix: the SWA sparse_attn_sharedkv op
        # indexes the (windowed) swa_block_table by ABSOLUTE block position
        # (col = floor(seqused_kv/block_size)). Under sliding_window>0, chunk-2+
        # prefill's swa_block_table is compacted to the window only (e.g. 6
        # cols / 152 tok for q=24 + history - num_ignored), so seqused_kv MUST
        # be the OFFSET (window-relative) kv_lens. FULL (absolute, =2328)
        # makes the op read bt[floor(2328/32)=72] -> OOB on the 6-col table ->
        # aicore 507057 (GPQA 0904 crash, cr=1 at deepseek_v4.py L680 + cr>1 at
        # L1402). Under sw=-1 (full table, absolute indexing in-bounds) and
        # decode (PA_ND paged, relative, proven 0/5 marker) keep FULL via the
        # `seqused_kv = kv_slice` decode branch (L921). chunk 1 (num_ignored=0
        # -> offset==full) is unchanged. Defined ONCE here in the shared
        # section so BOTH cr<=1 (SWA/dense) and cr>1 (c4/c128) dicts use it;
        # the cr>1 dict passes the same _swa_sl as seqused_kv (its ori read
        # uses the same windowed swa_block_table). NOTE: the cr>1 cmp
        # (compress_kv) read is bounded by seqused_kv/cmp_ratio when
        # cu_seqlens_cmp_kv=None (prefill) -- so OFFSET may starve the cmp
        # read of chunk-1 compressed history (cr=128: 152/128~1 block). If the
        # chunked needle returns a WRONG answer (no crash) under this fix,
        # the cmp bound must be decoupled: keep _swa_sl=OFFSET for the ori
        # read and set dsa_meta.cu_seqlens_cmp_kv to the FULL compressed
        # cumsum so the op bounds the cmp read from there, not seqused_kv.
        _swa_sl = seq_lens_l
        if (not is_decoding and kv_lens is not None
                and getattr(hf_config, 'sliding_window', -1) > 0):
            _swa_sl = kv_lens.to(torch.int32)
        if cr <= 1:
            dsa_inputs.append(dict(
                cos=cos_cur, sin=sin_cur,
                swa_kv_cache=swa_kv_cache, slot_mapping=swa_slot_mapping,
                block_table=swa_block_table,
                seq_lens=_swa_sl, query_start_loc=query_start_loc_l,
                sas_metadata=sas_metadata,
                dsa_caches=None, dsa_meta=None, is_decoding=is_decoding))
            continue

        # ---- c4 / c128: build/refresh DSA block tables on prefill ----
        # 6-tuple: (compress_kv, swa_kv, state, indexer_state, indexer_k, indexer_scale)
        compress_kv, _swa, state, indexer_state, indexer_k, indexer_scale = \
            v4_caches[layer_idx]
        if not is_decoding:
            _ACTIVE_REQ.prefill_kv_len = kv_len
            # Compressor RoPE cos/sin: request-INDEPENDENT, config-driven, so
            # allocate once on the first c4/c128 prefill and reuse for the
            # graph's lifetime (the captured decode graph is reused across
            # requests and reads these on-NPU). The per-request block tables
            # -- compress_kv/indexer_k (= swa_block_table) and state/
            # indexer_state (= state_bt, built per-step in the shared section
            # above) -- are NOT allocated here; they are request-aware and
            # rebuilt every step.
            if _DECODE_META.compress_cos is None:
                max_pt = getattr(step_context.cache_config,
                                 'max_prefill_token_num', 2048)
                full_pos = max(1024, int(max_pt) * 2)
                _DECODE_META.compress_cos, _DECODE_META.compress_sin = \
                    _build_compress_cos_sin_for_len(
                        full_pos, hf_config, device, dtype)
            _ACTIVE_REQ.compress_cos = _DECODE_META.compress_cos
            _ACTIVE_REQ.compress_sin = _DECODE_META.compress_sin

        # Per-cr state block table for THIS layer's recurrent state pool.
        # The compressor kernel indexes state_bt[batchIdx*maxBlockNumPerBatch
        # + curSeqIdx/blockSize] where curSeqIdx = bStartPos + sIdx and
        # bStartPos = start_pos[bIdx] is in ORIGINAL token units -- so the
        # column index reaches up to (max token position)/state_bs, covering
        # the FULL sequence length, NOT (max pos)/cr. The earlier contiguous
        # arange(1, num_reqs*M+1).view(num_reqs, M) scheme capped M at
        # (max_pt+margin)//state_bs (c4: 1216 cols -> 2432-token ceiling), so
        # multi-chunk prefill chunk2 (bStartPos=2304) or long decode OOB-wrote
        # past column 1216 -> aicore MTE (EZ9999), NOT OOM.
        #
        # V4_STATE_PAGED=1: the table is [num_reqs, session_len//state_bs]
        # (cheap int32), backed by a SHARED-BUDGET pool (max_num_seqs*m_cr
        # blocks, NOT num_reqs*session_len -- a single long request reuses the
        # over-provisioned pool without HBM growth; idle HBM ~6.5GB, full
        # session_len per-request sizing = ~36GB infeasible). Assignment is
        # slot-stable: a request's logical blocks keep the same pool ids for
        # its life (recurrent state MUST live at a stable address); new blocks
        # append as kv grows; completion detected via kv_len monotonicity (a
        # slot whose kv shrank was freed+reused) -- no engine free hook. The
        # pinned [max_num_seqs, max_cols] backing is mode-(a) _pin
        # graph-stable, updated each eager pre-step before replay (exactly how
        # swa_block_table already works in the graph).
        #
        # Default (V4_STATE_PAGED=0): the legacy contiguous arange scheme --
        # correct for prompts <= (max_pt+margin) tokens (the 2k-fix ceiling).
        _max_batch = getattr(step_context.cache_config, 'max_batches',
                             max(num_reqs, 1))
        _state_bs = C4_STATE_BS if cr == 4 else C128_STATE_BS
        _m_cr = (max_pt + V4_STATE_DECODE_MARGIN) // _state_bs
        indexer_state_bt = None
        if _state_paged:
            # Paged table width = session_len//state_bs (covers full sequence).
            _sess = int(getattr(step_context.cache_config, 'session_len', 8192))
            _max_cols = _cdiv(_sess, _state_bs)
            # Pool budget matches allocate_v4_caches (max_num_seqs*m_cr+1).
            # _DECODE_META.max_num_seqs is the pool-sizing ceiling (set in
            # allocate_v4_caches); use it directly because _mr (the local
            # alias) is only bound inside the `if is_decoding:` block above,
            # but this paged path runs for prefill too.
            _mrns = _DECODE_META.max_num_seqs or max(num_reqs, 1)
            _pool = _state_pool_size(cr, _mrns, _m_cr)
            _alloc = _get_state_alloc(cr, _state_bs, _max_cols, _pool,
                                      _mrns, device)
            # Call assign once per cr per step; per-layer reuse via the cache.
            _view = _PAGED_BT_CACHE.get(cr)
            if _view is None:
                _view = _alloc.assign(num_reqs, kv_lens_cpu, is_decoding,
                                      max_pt, seq_ids=getattr(
                                          step_context, 'seq_ids', None))
                _PAGED_BT_CACHE[cr] = _view
            state_bt = _view
            # indexer_state lives in its own pool (same c4 scheme); c128 has
            # no indexer. Guard to c4.
            if cr == 4:
                _ix_key = ('indexer', cr)
                _ix_view = _PAGED_BT_CACHE.get(_ix_key)
                if _ix_view is None:
                    _ix_alloc = _get_state_alloc(_ix_key, _state_bs, _max_cols,
                                                 _pool, _mrns, device)
                    _ix_view = _ix_alloc.assign(num_reqs, kv_lens_cpu,
                                                is_decoding, max_pt,
                                                seq_ids=getattr(
                                                    step_context, 'seq_ids',
                                                    None))
                    _PAGED_BT_CACHE[_ix_key] = _ix_view
                indexer_state_bt = _ix_view
        elif is_decoding:
            _pin_name = 'state_bt_c4' if cr == 4 else 'state_bt_c128'
            # state_bt = arange(1, num_reqs*M_cr+1).view(num_reqs, M_cr) depends
            # only on (num_reqs, M_cr) -- identical across layers of the same cr
            # and across decode steps (fixed batch). The attention op only READS
            # it, so when (num_reqs, M_cr) is unchanged the pinned backing already
            # holds the byte-identical value: reuse the cached view and SKIP the
            # per-layer arange rebuild + the _pin_meta copy_ (~76 copy_ + ~58
            # arange per forward eliminated). data_ptr is unchanged -> graph-safe.
            _cached = _STATE_BT_CACHE.get(_pin_name)
            if (_cached is not None and _cached[0] == num_reqs
                    and _cached[1] == _m_cr):
                state_bt = _cached[2]
            else:
                state_bt = (torch.arange(1, num_reqs * _m_cr + 1,
                            device=device).view(num_reqs, _m_cr).to(torch.int32))
                state_bt = _pin_meta(_pin_name, state_bt, max_dim0=_mr)
                _STATE_BT_CACHE[_pin_name] = (num_reqs, _m_cr, state_bt)
        else:
            state_bt = (torch.arange(1, num_reqs * _m_cr + 1, device=device)
                        .view(num_reqs, _m_cr).to(torch.int32))
        # Legacy (non-paged) indexer_state_bt pinning: c4 reuses the same
        # table in its own pool. Guarded to c4 and to non-paged (the paged
        # branch built indexer_state_bt above).
        if cr == 4 and not _state_paged:
            indexer_state_bt = state_bt
            if is_decoding:
                _cached_ix = _STATE_BT_CACHE.get('indexer_state_bt')
                if (_cached_ix is not None and _cached_ix[0] == num_reqs
                        and _cached_ix[1] == _m_cr):
                    indexer_state_bt = _cached_ix[2]
                else:
                    indexer_state_bt = _pin_meta('indexer_state_bt',
                                                 indexer_state_bt, max_dim0=_mr)
                    _STATE_BT_CACHE['indexer_state_bt'] = (
                        num_reqs, _m_cr, indexer_state_bt)

        # dsa_kv_bt: SHARED full-history block table for compress_kv +
        # indexer_k (own pool, append-only, MLA_BS-paged). Replaces the old
        # swa_block_table alias (L1859/1867 below) that cross-request-
        # contaminated under swa eviction -- the 0903 structural bug. Built
        # per-cr (cached in _PAGED_BT_CACHE), used by BOTH compress_kv (all
        # cr>1) and indexer_k (cr==4). Pool/cols from _DECODE_META (set in
        # allocate_v4_caches); kv_lens_cpu is the FULL pre-offset per-req kv
        # length (same source the state alloc uses) so the table grows to
        # FULL history, not the swa window. swa_kv keeps swa_block_table
        # (windowed/evicted) -- SWA only needs the window.
        if cr > 1:
            _dk_key = ('dsa_kv', cr)
            _dk_view = _PAGED_BT_CACHE.get(_dk_key)
            if _dk_view is None:
                _dk_sess = int(getattr(step_context.cache_config,
                                       'session_len', 8192))
                _dk_mrns = _DECODE_META.max_num_seqs or max(num_reqs, 1)
                _dk_pool = (_DECODE_META.dsa_kv_pool
                            or _dsa_kv_pool_size(_dk_mrns, _dk_sess))
                _dk_cols = (_DECODE_META.dsa_kv_max_cols
                            or _cdiv(max(1, _dk_sess), MLA_BS))
                _dk_alloc = _get_dsa_kv_alloc(cr, _dk_cols, _dk_pool,
                                              _dk_mrns, device)
                _dk_view = _dk_alloc.assign(num_reqs, kv_lens_cpu,
                                            is_decoding, max_pt,
                                            seq_ids=getattr(
                                                step_context, 'seq_ids', None))
                _PAGED_BT_CACHE[_dk_key] = _dk_view
            dsa_kv_bt = _dk_view
        else:
            dsa_kv_bt = None

        # num_compressed_tokens = min(tokenSize, tokenSize//cr + batchSize)
        # (vllm-ascend dsa_v1.py _num_compressor_metadata_rows). tokenSize is
        # the current forward's total token count (total_q = sum q_lens), and
        # batchSize is num_reqs -- NOT the per-seq kv_len and NOT a hardcoded
        # 1. The single-request assumption (+1) trips the compressor tiling
        # check ("ropeSin ... should be equal to min(tokenSize, ...)") on any
        # batched prefill/decode with >1 request. Computed per layer (cr
        # varies) and for BOTH prefill and decode (decode skips the prefill
        # block-table build above but still needs a fresh, request-aware
        # num_cmp -- it cannot reuse a stale per-layer _ACTIVE_REQ value).
        num_cmp = max(min(total_q, total_q // cr + num_reqs), 1)
        _ACTIVE_REQ.num_cmp = num_cmp

        dsa_caches = (compress_kv, swa_kv_cache, state,
                      indexer_state, indexer_k, indexer_scale)

        dsa_meta = SimpleNamespace(
            full_compress_cos=_ACTIVE_REQ.compress_cos,
            full_compress_sin=_ACTIVE_REQ.compress_sin,
            start_pos=start_pos,
            num_compressed_tokens=num_cmp,
            num_reqs_actual=num_reqs,
            compress_kv_block_table=dsa_kv_bt,
            compress_kv_block_size=MLA_BS,
            state_block_table=state_bt,
            sas_metadata=sas_metadata,
            cu_seqlens_cmp_kv=None,
        )

        if cr == 4:
            dsa_meta.indexer_k_block_table = dsa_kv_bt
            dsa_meta.indexer_k_block_size = MLA_BS
            dsa_meta.indexer_state_block_table = indexer_state_bt
            # The lightning indexer's key cache holds the *full* (quantized) kv
            # — it is paged by MLA_BS=128 to cover the whole kv, not the
            # compressed num_cmp. So per-req kvlens + the metadata op's
            # actual_seq_lengths_key / max_seqlen_k are the ORIGINAL kv seq
            # lens, mirroring vllm-ascend dsa_v1.py. batch_size / max_seqlen_q
            # track the real request set (warmup runs batched decode, so a
            # hardcoded batch_size=1 / first-seq lens trips AICPU validation
            # retCode 0x2a).
            # query_start_loc has num_reqs+1 rows (cat([0], cumsum)) so
            # numel()-1 == num_reqs; use num_reqs directly so this doesn't
            # depend on the (None under _meta_all_hit) eager tensor.
            _nr = max(num_reqs, 1)
            # _kv is a slice of seq_lens_l (= kv_lens.to(int32), see L532) ->
            # already int32 (the prior .to(int32) was a no-op). _pin_meta copies
            # the source into a persistent backing internally, so the prior
            # .clone() was a redundant extra copy. AND indexer_kvlens is a SHARED
            # buffer (one _DECODE_META.indexer_kvlens for all c4 layers), so the
            # prior per-c4-layer _pin_meta re-copied the same value into the same
            # buffer ~24x/fwd (idempotent waste). Hoist to once-per-step: fill
            # _indexer_kvlens_view on the first c4 layer, reuse for the rest.
            # Byte-identical to before (all c4 layers already aliased the same
            # shared view; only the redundant re-copies are removed).
            if _full_graph:
                # indexer_kvlens moved IN-GRAPH (build_v4_decode_meta_in_graph
                # fills dsa_meta.indexer_kvlens = the in-graph FULL kv lens
                # derived from position_ids; Stage 1b -- NOT the offset
                # seq_lens, which would read wrong dsa_kv slots); skip the
                # per-step _pin_meta copy_ racer.
                dsa_meta.indexer_kvlens = None
            elif is_decoding:
                if _indexer_kvlens_view is None:
                    # FULL (absolute) kv lens: indexer_k is paged by MLA_BS
                    # over FULL history (dsa_kv_bt, absolute 0-based), so
                    # actual_seq_lengths_key must be the ORIGINAL (pre-offset)
                    # length -- NOT the offset seq_lens_l (window-relative),
                    # which would read only the recent window / wrong blocks.
                    # (This branch is eager: not _full_graph -> full_kv_lens is
                    #  computed above; the seq_lens_l fallback is defensive.)
                    if full_kv_lens is not None:
                        _kv = full_kv_lens.to(torch.int32)[:_nr]
                    else:
                        _kv = (seq_lens_l[:_nr] if seq_lens_l.numel() >= _nr
                               else seq_lens_l)
                    _indexer_kvlens_view = _pin_meta(
                        'indexer_kvlens', _kv, max_dim0=_mr)
                dsa_meta.indexer_kvlens = _indexer_kvlens_view
            else:
                if full_kv_lens is not None:
                    _kv = full_kv_lens.to(torch.int32)[:_nr]
                else:
                    _kv = (seq_lens_l[:_nr] if seq_lens_l.numel() >= _nr
                           else seq_lens_l)
                dsa_meta.indexer_kvlens = _kv.clone()
            # FULL-graph C1 step2b: when V4_META_IN_GRAPH=1 + decoding, defer the
            # lightning-indexer metadata op to the captured attention forward
            # (set qli_metadata=None sentinel; op built in-graph). Otherwise
            # dedup across c4 layers via the per-step _qli_cache (step2a) and
            # bake max_seqlen_k to the session_len ceiling (sizing-only; actual
            # per-step lengths flow through actual_seq_lengths_key tensor). The
            # _build_qli_metadata helper is shared by both the eager path here
            # and the in-graph path in the attention forward.
            if _qli_in_graph and is_decoding:
                dsa_meta.qli_metadata = None
            else:
                if _meta_callonce:
                    # cross-step call-once (mirrors vllm
                    # decode_ratio_to_sas_metadata["qli"]; see _QLI_XCACHE).
                    _qli_key = num_reqs
                    if _qli_key in _QLI_XCACHE:
                        _qli_cache = _QLI_XCACHE[_qli_key]
                        _QLI_XC_STATS['hit'] += 1
                    else:
                        _qli_cache = _build_qli_metadata(
                            hf_config, query_start_loc_l, seq_lens_l,
                            seq_len, _eff_max_kv, device)
                        _QLI_XCACHE[_qli_key] = _qli_cache
                        _QLI_XC_STATS['miss'] += 1
                elif _qli_cache is None:
                    # per-step within-step dedup across c4 layers (step2a)
                    _qli_cache = _build_qli_metadata(
                        hf_config, query_start_loc_l, seq_lens_l,
                        seq_len, _eff_max_kv, device)
                dsa_meta.qli_metadata = _qli_cache
                if is_decoding:
                    dsa_meta.qli_metadata = _pin_list(
                        _DECODE_META.qli_metadata, layer_idx,
                        dsa_meta.qli_metadata)

        dsa_inputs.append(dict(
            cos=cos_cur_cmp, sin=sin_cur_cmp,
            swa_kv_cache=swa_kv_cache, slot_mapping=swa_slot_mapping,
            block_table=swa_block_table,
            seq_lens=_swa_sl, query_start_loc=query_start_loc_l,
            sas_metadata=sas_metadata,
            dsa_caches=dsa_caches, dsa_meta=dsa_meta, is_decoding=is_decoding))

    return dsa_inputs


def build_v4_decode_meta_in_graph(position_ids, attn_metadata, hf_config,
                                  block_size, device, dtype):
    """FULL-graph (V4_FULL_GRAPH_DECODE=1): compute the SHARED decode metadata
    IN-GRAPH (inside the captured ``DeepseekV4Model.forward``) from the device
    graph-input buffers refreshed each step by ``fill_buffers_cudagraph`` --
    eliminating the ~31 per-step eager ops (slot_mapping construction + cos
    gather + ``_pin_meta`` copies) that race the graph replay (507018).

    Mirrors ``build_v4_dsa_inputs`` L709-840 but reads from graph inputs
    (``attn_metadata.kv_seqlens`` / ``.block_offsets`` + the ``position_ids``
    forward arg, all device + address-stable) and uses NO ``.item`` / NO
    ``_pin`` -- in-graph fresh tensors land in the captured graph's private
    pool at stable addresses, so no host-side pin/copy is needed.

    Decode-only: q_lens==1 per req (constant), so query_start_loc = arange
    (constant), start_pos = (kv_seqlens - 1).clamp(min=0), and
    swa_slot_mapping derives from kv_seqlens + block_offsets alone. Requires
    V4_PRECOMPUTE_ROPE=1 (the persistent main_cos/sin tables). For the decode
    graph max_tokens == max_batches (1 tok/req), so position_ids.view(-1) [nt]
    and kv_seqlens [N] are consistent (asserted). Padded slots (kv_seqlens=0
    -> slot=0) are bounded out by cu_seqlens_q (the op never reads them).

    Returns the shared dict (query_start_loc/seq_lens/cos_cur/sin_cur/
    cos_cur_cmp/sin_cur_cmp/swa_slot_mapping/swa_block_table/start_pos) the
    model forward merges into every layer's dsa_inputs entry (overriding the
    None sentinels build_v4_dsa_inputs left under V4_FULL_GRAPH_DECODE).
    """
    kv_seqlens = attn_metadata.kv_seqlens        # device int32 [N], refreshed
    block_offsets = attn_metadata.block_offsets  # device int32 [N, num_blocks]
    pos = position_ids.view(-1).long()           # [nt], device
    nt = pos.shape[0]
    N = kv_seqlens.shape[0]
    assert N == nt, (
        f"V4_FULL_GRAPH_DECODE: decode graph shape mismatch N={N} nt={nt} "
        f"(expected max_tokens==max_batches for the 1-tok/req decode graph)")
    rotary_dim = hf_config.qk_rope_head_dim

    # query_start_loc = arange(N+1) (decode q_lens=1 -> cu_seqlens_q=[0,1,..,N])
    query_start_loc = torch.arange(N + 1, dtype=torch.int32, device=device)
    # sw=128 NULL-PAD fix (match vllm-ascend; see
    # dsv4-sw128-compaction-vs-nullpad-rootcause). The engine's
    # block_offsets is COMPACTED (WindowBlockManager drops evicted logical
    # blocks -> window-only table), and kv_seqlens is the OFFSET
    # (window-relative) length. The opaque npu_sparse_attn_sharedkv kernel
    # reads the ori (swa) block_table by ABSOLUTE block index
    # [0, seqused_kv/block_size] and shares ONE seqused_kv to bound BOTH the
    # ori read and (when cu_seqlens_cmp_kv is empty/None) the cmp
    # (compress_kv) read (=seqused_kv/cmp_ratio). OFFSET seqused_kv starves
    # the cmp bound -> the question scrolls out of the SWA window (~128
    # decode tok) and HCA cannot recover its compressed block -> drift
    # (GSM8K ~83% vs sw=-1 98%; Candidate B's cu_seqlens_cmp_kv decouple
    # FAILED -- the kernel ignores it in PA_ND decode).
    #
    # Fix: reconstruct a FULL-EXTENT swa block_table IN-GRAPH (so the
    # absolute-index ori read stays in-bounds) with evicted slots -> null_id
    # (a valid physical block; the evicted region is masked out of attention
    # by ori_win_left = window-1, exactly as vllm-ascend does with its
    # null_block), and set seqused_kv = FULL (= position_id+1) so the cmp
    # bound = FULL/cmp_ratio covers the question's compressed block. The swa
    # POOL still evicts (frees physical blocks -> memory saved); only the
    # cheap int32 table grows to full-extent. Decode-only; prefill has
    # num_ignored_history=0 -> offset==full -> unaffected.
    _full_pos = position_ids.view(-1).long()              # [N] ABSOLUTE (decode: full_kv-1)
    seq_lens = (_full_pos + 1).to(torch.int32)            # [N] FULL seqused_kv

    # full-extent swa block_table reconstruction. num_evicted_blocks =
    # (full_kv - offset_kv) // block_size. For each full-extent column j,
    # slot j is REAL iff j in [num_evicted, full_kv_blocks); the real physical
    # id = block_offsets[req, j - num_evicted] (the compacted window table).
    # Evicted / padding slots -> null_id (=0; masked by ori_win_left).
    _full_kv = _full_pos + 1                              # [N] FULL kv lens
    _full_kv_blocks = ((_full_kv + block_size - 1) // block_size).to(torch.int32)  # [N] ceil
    _n_evict = (((_full_kv - kv_seqlens.long()).clamp(min=0)
                 + block_size - 1) // block_size).to(torch.int32)  # [N] evicted blocks
    _max_full_cols = _cdiv(getattr(_DECODE_META, 'session_len', 0)
                          or 65536, block_size)
    _j = torch.arange(_max_full_cols, device=device, dtype=torch.int32)  # [C]
    _bo_cols = block_offsets.shape[1]
    _gather_idx = (_j[None, :] - _n_evict[:, None]).clamp(
        min=0, max=max(0, _bo_cols - 1)).long()           # [N, C]
    _real = torch.gather(block_offsets.to(torch.int32), 1, _gather_idx)  # [N, C]
    _is_real = ((_j[None, :] >= _n_evict[:, None])
                & (_j[None, :] < _full_kv_blocks[:, None]))  # [N, C]
    _NULL_ID = 0  # masked out by ori_win_left (evicted region not attended)
    swa_block_table = torch.where(_is_real, _real,
                                  torch.full_like(_real, _NULL_ID))

    # cos/sin gathered from the persistent precomputed tables (address-stable
    # _DECODE_META buffers; the gather itself is captured in-graph).
    # Defense-in-depth: clamp position to the table bound so an out-of-range
    # position NEVER reaches aclnn IndexCheck. The engine cap + the +1 table
    # sizing already guarantee pos < main_cos.shape[0] in normal operation;
    # this clamp only engages if some path bypasses the cap (a future bug,
    # mis-config, or a request whose prompt alone exceeds session_len). It
    # turns a guaranteed aicore crash (507011 -> whole-serve-down) into
    # graceful degradation of the offending token only. Zero host sync -- a
    # capturable in-graph op, no per-step .item()/cpu() racer.
    _rope_bound = _DECODE_META.main_cos.shape[0] - 1
    pos = pos.clamp(min=0, max=_rope_bound)
    cos_cur = _DECODE_META.main_cos[pos].view(nt, 1, 1, rotary_dim)
    sin_cur = _DECODE_META.main_sin[pos].view(nt, 1, 1, rotary_dim)
    cos_cur_cmp = _DECODE_META.main_cos_cmp[pos].view(nt, 1, 1, rotary_dim)
    sin_cur_cmp = _DECODE_META.main_sin_cmp[pos].view(nt, 1, 1, rotary_dim)

    # start_pos / indexer_kvlens for the COMPRESSOR path: FULL (absolute),
    # derived in-graph from position_ids (no new graph input). _full_pos is
    # computed above (NULL-PAD block); reuse it. dsa_kv_bt + state_bt are
    # absolute-position-indexed; the compressor RoPE theta=160000 over the
    # full position. For decode (q_lens=1): start_pos = full_kv-1 =
    # position_id, indexer_kvlens = full_kv = position_id+1.
    start_pos = _full_pos.to(torch.int32)                 # = full_kv - 1
    indexer_kvlens = (_full_pos + 1).to(torch.int32)      # [N] FULL kv lens

    # swa_slot_mapping (decode): the NEW token's slot. kv_pos = full_kv-1
    # (= position_id, ABSOLUTE/FULL) -- the just-decoded token lives at the
    # recent (non-evicted) end of the full-extent swa_block_table, so it
    # gathers a REAL physical block (not the null_id). bnum = kv_pos//bs is
    # an absolute column index into the full-extent swa_block_table; clamp
    # to [0, max_full_cols-1] for graph safety. req_ar = arange(N). All
    # in-graph, fixed shape.
    req_ar = torch.arange(N, device=device)
    kv_pos = _full_pos.clamp(min=0)                      # [N] FULL (full_kv-1)
    bnum = (kv_pos // block_size).clamp(min=0, max=_max_full_cols - 1)  # [N]
    bidx = kv_pos % block_size                           # [N]
    blocks = swa_block_table[req_ar, bnum]               # [N] (full-extent table)
    swa_slot_mapping = (blocks.to(torch.int64) * block_size + bidx).to(torch.int32)

    return dict(query_start_loc=query_start_loc, seq_lens=seq_lens,
                cos_cur=cos_cur, sin_cur=sin_cur,
                cos_cur_cmp=cos_cur_cmp, sin_cur_cmp=sin_cur_cmp,
                swa_slot_mapping=swa_slot_mapping, swa_block_table=swa_block_table,
                start_pos=start_pos, indexer_kvlens=indexer_kvlens)


def reset_active_request():
    """Reset the per-request DSA state (call on a new prefill / session)."""
    global _ACTIVE_REQ
    _ACTIVE_REQ = None


_ACTIVE_REQ = None


# ---------------------------------------------------------------------------
# Decode graph-capture persistent buffers
# ---------------------------------------------------------------------------
class _V4DecodeMeta:
    """Persistent, address-stable buffers for the decode graph-capture path.

    Mirrors vllm-ascend's ``AscendDSAMetadataBuilder`` (dsa_v1.py:463-472):
    allocate once, refresh contents in-place each step. At graph capture the
    NPU records these buffers' addresses; at replay ``build_v4_dsa_inputs``
    re-runs in the engine pre-step and ``copy_``'s the new values into the
    SAME buffers, so the graph reads refreshed contents from stable
    addresses. Without this, fresh-per-step tensors get freed between steps
    and replay reads stale/freed addresses -> aicore OOB ("MTE DDR address
    out of range" / error 507011).

    These buffers are NOT cleared by ``reset_active_request`` -- the captured
    graph is reused across requests (graph_key is stable), so the buffer
    addresses must stay valid for the graph's whole lifetime. Prefill (not
    is_decoding) bypasses the per-step buffers and uses fresh tensors (prefill
    is never captured). The block tables / compress cos-sin ARE shared with
    prefill (request-independent values) and allocated here once.
    """
    # shared (cross-layer) per-step buffers, refreshed each decode step
    query_start_loc = None   # [N+1] int32
    seq_lens = None          # [N] int32
    cos_cur = None           # [nt,1,1,rotary_dim] (model dtype)
    sin_cur = None
    cos_cur_cmp = None       # c4/c128 layers: compress_rope_theta variant
    sin_cur_cmp = None
    swa_slot_mapping = None  # [total_q] int32
    swa_block_table = None   # [N, max_blocks] int32
    start_pos = None         # [N] int32
    indexer_kvlens = None    # [nr] int32 (shared across c4 layers; same value)
    # per-layer op-output metadata (cr differs per layer -> per-layer buffers)
    sas_metadata = None      # list[tensor|None], len=num_layers
    qli_metadata = None      # list[tensor|None], len=num_layers (c4 layers only)
    # FULL-graph C1 step2b: baked session_len ceiling for the metadata ops'
    # max_seqlen_kv/max_seqlen_k sizing scalar. Read by the captured attention
    # forward when it builds the metadata op IN-GRAPH (V4_META_IN_GRAPH=1).
    max_kv_ceiling = None
    # FULL-graph C1 step2b fix: SEPARATE pinned seqused_kv for the SAS op,
    # =ceiling at capture (temp sizing) / =real at replay (compute). The
    # attention op keeps reading seq_lens (above) -- only the SAS op uses
    # this. `sas_seqused_kv` is the backing buffer; `sas_seqused_kv_view` is
    # the active view read by _build_sas_metadata(in_graph=True).
    sas_seqused_kv = None
    sas_seqused_kv_view = None
    sas_cu_seqlens = None     # persistent external 1-elem backing for the in-graph decode cu_seqlens_ori_kv 0-view
    # request-INDEPENDENT persistent tensors (allocated on first c4/c128
    # prefill, reused across all requests for the graph's lifetime)
    compress_bt = None
    indexer_k_bt = None
    # per-cr per-request state block tables (M differs: c4=72, c128=1); each
    # layer type gets its own pinned buffer so shapes never collide.
    state_bt_c4 = None
    state_bt_c128 = None
    indexer_state_bt = None
    compress_cos = None
    compress_sin = None
    # Persistent full main-RoPE cos/sin tables [max_pos, 1, 1, rotary_dim],
    # precomputed once on the first decode step (V4_PRECOMPUTE_ROPE=1).
    # Indexed by absolute position each step instead of rebuilding a
    # pos_max-sized table + a pos_max .item() host sync every step. The
    # YaRN inv_freq depends only on original_max_position_embeddings (fixed
    # config), so cos[i] == cos(i*inv_freq) is position-deterministic; a
    # precomputed [max_pos, ...] table indexed by position_ids is identical
    # to building [pos_max] and indexing -- but with zero per-step rebuild
    # and no host sync, and the fixed-shape table is graph-capturable.
    main_cos = None
    main_sin = None
    main_cos_cmp = None       # compress_rope_theta (160000) variant
    main_sin_cmp = None
    main_rope_len = 0
    # per-rank max concurrent seqs (pool-sizing ceiling, set in
    # allocate_v4_caches). Decode pads every step up to this size so the
    # single captured decode graph sees a constant batch.
    max_num_seqs = None
    # dsa_kv (compress_kv+indexer_k) own-pool sizing (set in
    # allocate_v4_caches). dsa_kv_pool = block count of the separate
    # full-history pool; dsa_kv_max_cols = table column count
    # (cdiv(session_len, MLA_BS)). Read by _get_dsa_kv_alloc.
    dsa_kv_pool = None
    dsa_kv_max_cols = None
    session_len = None  # for in-graph swa full-extent table reconstruction


_DECODE_META = _V4DecodeMeta()

# state_bt redundancy cache: state_bt = arange(1, num_reqs*M_cr+1).view(num_reqs,
# M_cr) depends ONLY on (num_reqs, M_cr) -- identical across all layers of the same
# compress_ratio AND across decode steps (for a fixed batch). It is built + _pin_meta
# copy_'d per layer (61x/forward) in build_v4_dsa_inputs, but the attention op only
# READS it (never writes), so once the pinned backing holds the right value for a
# given num_reqs, every subsequent layer/step with the same num_reqs can reuse the
# pinned view WITHOUT rebuilding the arange or re-copying (the backing still holds
# the byte-identical value -> graph reads the correct data from the stable address).
# Keyed by pin_name -> (num_reqs, m_cr, view). Invalidated on num_reqs/m_cr change.
_STATE_BT_CACHE = {}

# Per-layer sas_metadata pin skip: sas_metadata is the kv_len-INVARIANT output
# of sparse_attn_sharedkv_metadata (the op uses seqused_kv only as a sizing bound;
# the real per-step kv_len flows through a SEPARATE pinned tensor, see
# dsv4-2b-in-graph-sas-blocked; on-device fingerprinted 2026-08-20: valid prefix
# constant across kv 6..9). Cached in _SAS_XCACHE[(cr, num_reqs)] and computed
# ONCE per (cr, num_reqs), so its object/data_ptr/content is byte-stable across
# decode steps for a fixed batch. The per-layer _pin_list re-copies the SAME value
# into the same per-layer backing every step (~43 copy_/fwd pure waste -- the
# biggest per-layer pin cluster after state_bt). Skipping the copy when (num_reqs,
# cr) is unchanged is byte-identical in effect (the backing already holds the
# value); keying on (num_reqs, cr) self-invalidates on batch / capture-size
# change (mirrors _STATE_BT_CACHE). The op-output shape is fixed (independent of
# num_reqs -- else the existing per-step _pin_list would itself reallocate ->
# stale-addr OOB), so mode-(c) never reallocates and the capture-time address
# returned on skip is the stable graph-read address.
_SAS_PIN_CACHE = {}

# Cross-step call-once cache for the SAS metadata OP itself (mirrors vllm-ascend
# dsa_v1.py decode_ratio_to_sas_metadata: the op runs ONCE per (cr, num_reqs) at
# the first decode step, then is reused across steps -- the output is
# kv_len-invariant, so no recompute is needed as kv grows). Keyed by
# (cr, num_reqs) so it self-invalidates on batch-composition change, exactly like
# _SAS_PIN_CACHE. Without this, the per-step local _sas_cache (rebuilt each call
# to build_v4_dsa_inputs) re-launches the op 3x/step (once per cr) even though
# _SAS_PIN_CACHE already skips the per-layer COPY -- the op host-dispatch is the
# remaining eager cost in build_step_context. Gated by V4_META_CALLONCE.
# QLI metadata is ALSO cross-step cached in _QLI_XCACHE (keyed by num_reqs --
# QLI is layer-INdependent, single cr=4). vllm-ascend dsa_v1.py caches QLI in
# the SAME decode_ratio_to_sas_metadata call-once dict as SAS (L1056-1078),
# with the identical per-step input actual_seq_lengths_key=self.seq_lens[..]
# that we pass. The op OUTPUT is a sizing/layout descriptor (a 1024-int32
# buffer); the real per-step kv_len flows through the SEPARATE KV tensors +
# block_table fed to the npu_vllm_quant_lightning_indexer indexer call (in the
# captured forward), NOT through this metadata. So the cached output is
# byte-valid across decode steps for a fixed batch -- same invariance class as
# SAS. Correctness is the on-device verification (Paris/Madrid/Rome match).
_SAS_XCACHE = {}
# Cumulative hit/miss counters for _SAS_XCACHE (read by the agent V4-STEP print
# to confirm the call-once cache fires in steady state). 'miss' = op ran;
# 'hit' = op skipped. Steady-state decode of a fixed batch should be ~all hits
# after the first step per (cr, num_reqs).
_SAS_XC_STATS = {'hit': 0, 'miss': 0}
# Cross-step call-once cache for the QLI metadata OP (mirrors vllm-ascend
# decode_ratio_to_sas_metadata["qli"], same call-once block as SAS). Keyed by
# num_reqs (QLI is cr=4 only, layer-independent -> one op/step in the eager
# path, now 0/step after the first). Without this the per-step _qli_cache
# (rebuilt each build_v4_dsa_inputs call) re-launches the op 1x/step. Gated by
# V4_META_CALLONCE (same flag as SAS -- vllm caches both in one dict).
_QLI_XCACHE = {}
_QLI_XC_STATS = {'hit': 0, 'miss': 0}


def sas_xcache_stats():
    """Return (hit, miss) cumulative SAS call-once cache counters."""
    return _SAS_XC_STATS['hit'], _SAS_XC_STATS['miss']


def qli_xcache_stats():
    """Return (hit, miss) cumulative QLI call-once cache counters."""
    return _QLI_XC_STATS['hit'], _QLI_XC_STATS['miss']


def _pin(dst, src, max_dim0=None, max_dim1=None):
    """Return ``(view, backing)`` for ``src`` -- graph-stable persistent pin.

    ``view`` is what the caller uses this step (the active prefix); ``backing``
    is the FULL-SIZE buffer that must be stored for the next call (so the
    shape guard sees the max dims, not the active prefix -- otherwise the next
    call sees a smaller view, fails the shape guard, REALLOCATES, and moves the
    buffer to a new address -> the captured NPU graph reads a stale/freed
    address -> aicore OOB 507011/507057 at the next decode step. Storing the
    view (old code) was exactly that bug.)

    Three modes:

    (a) ``max_dim0`` + ``max_dim1`` both given (2D block tables) -- MAX-SIZE
        2D PREFIX. Allocate ``backing`` ONCE at ``(max_dim0, max_dim1)`` and
        return the full-width prefix view ``backing[:s0]`` (== ``backing[:s0,
        :max_dim1]``), whose ``data_ptr`` == ``backing.data_ptr()`` for any
        ``s0``. The active sub-rectangle ``[:s0, :s1]`` is ``copy_``'d each
        step; the padded tail ``[:s0, s1:max_dim1]`` is zeroed so no stale
        block id from a prior (longer) request can be followed. The op bounds
        block-table reads by ``seqused_kv`` (actual kv length), so padded
        columns are never accessed -- zeroing is defensive. ``s1`` (block
        count per request) GROWS with request length (warmup ~1 block, real
        decode after a 2k prefill ~45 blocks); without max_dim1 the old shape
        guard saw ``dst.shape[1] != s1`` and reallocated -> stale addr -> OOB
        on the first decode step after a long prefill. Used for
        swa_block_table (swa-only -- compress_kv_block_table /
        indexer_k_block_table now use the separate dsa_kv_bt backing from
        _V4StateAlloc, not this pin; the old aliasing was the 0903
        cross-request contamination bug).

    (b) ``max_dim0`` only given and > ``src.shape[0]`` -- MAX-SIZE 1D PREFIX.
        Allocate ``backing`` ONCE at ``max_dim0`` rows; each call ``copy_``'s
        the active prefix ``[:s0]`` in place and returns that prefix VIEW.
        ``view.data_ptr`` == ``backing.data_ptr()`` for ANY prefix length ->
        every captured graph size (1/2/4/8/15) AND every replay step share ONE
        data_ptr. Used for per-request fields whose dim0 == num_reqs.

    (c) ``max_dim0`` None or == src.shape[0] -- FIXED-SHAPE pin (op outputs
        whose shape is constant across batch sizes, e.g. sas/qli metadata =
        ``(1024,)``). Clone on first call / shape change, else ``copy_`` in
        place (address unchanged). ``view == backing`` here."""
    # (a) 2D block-table: both dims maxed, address stable across (s0, s1).
    if (max_dim0 is not None and max_dim1 is not None and src.dim() == 2
            and (max_dim0 > src.shape[0] or max_dim1 > src.shape[1])):
        s0, s1 = src.shape[0], src.shape[1]
        if (dst is None or dst.shape[0] != max_dim0 or dst.shape[1] != max_dim1
                or dst.dtype != src.dtype):
            dst = torch.zeros((max_dim0, max_dim1), dtype=src.dtype,
                              device=src.device)
        dst[:s0, :s1].copy_(src)
        if s1 < max_dim1:
            dst[:s0, s1:max_dim1].zero_()
        # Zero the FREED-SLOT rows (s0:max_dim0) -- when num_reqs < the
        # capture/padding batch (batch-change 2->1, slot recycle), these rows
        # still hold the prior request's block ids. The captured attention op
        # baked its read extent at max_num_seqs, so it reads these stale block
        # ids -> invalid GM address -> 507035 "cross-device memory access
        # timeout" (EZ9999 vector-core). Mirrors vllm-ascend dsa_v1.py
        # `block_table[num_reqs_actual:].fill_(0)` (L935-936). Without this the
        # freed slot's seq_lens (mode-b, torch.empty -> uninitialized garbage)
        # drives a huge block-table read -> OOB. Zeroing is defensive+correct:
        # the op bounds reads by seqused_kv, so a zeroed freed slot (seq_len=0,
        # block_table=0) is never accessed. Cost: <=(max_batch-1) rows/step.
        if s0 < max_dim0:
            dst[s0:max_dim0].zero_()
        return dst[:s0], dst
    # (b) 1D max-size prefix.
    if max_dim0 is not None and max_dim0 > src.shape[0]:
        rest = src.shape[1:]
        if (dst is None or dst.shape[0] != max_dim0
                or dst.shape[1:] != rest or dst.dtype != src.dtype):
            dst = torch.empty((max_dim0, *rest), dtype=src.dtype,
                              device=src.device)
        dst[:src.shape[0]].copy_(src)
        # Zero the FREED-SLOT tail (src.shape[0]:max_dim0). mode-(b) uses
        # torch.empty -> the freed slots (num_reqs < padding, batch-change /
        # slot recycle) hold UNINITIALIZED garbage. The captured op baked its
        # read extent at max_num_seqs, so for seq_lens/start_pos/
        # query_start_loc it reads garbage -> a garbage-huge seq_len drives a
        # block-table read off the end of the KV pool -> invalid GM address ->
        # 507035 EZ9999 vector-core "cross-device memory access timeout".
        # Non-deterministic because torch.empty's content depends on the
        # allocator free-list state. Mirrors vllm-ascend dsa_v1.py
        # `seq_lens[num_reqs_actual:].fill_(0)`. The op bounds reads by
        # seqused_kv so a zeroed freed slot (value 0) is never followed.
        if src.shape[0] < max_dim0:
            dst[src.shape[0]:max_dim0].zero_()
        return dst[:src.shape[0]], dst
    # (c) fixed-shape.
    if dst is not None and dst.shape == src.shape and dst.dtype == src.dtype:
        dst.copy_(src)
        return dst, dst
    out = src.clone()
    return out, out


def _pin_meta(name, src, max_dim0=None, max_dim1=None):
    """``_pin`` against a named attribute on ``_DECODE_META`` (shared buffer).

    Stores the FULL backing buffer (NOT the prefix view) so the next call's
    shape guard sees the max dims; returns the active view to the caller."""
    view, backing = _pin(getattr(_DECODE_META, name), src,
                         max_dim0=max_dim0, max_dim1=max_dim1)
    setattr(_DECODE_META, name, backing)
    return view


def _pin_list(lst, idx, src, max_dim0=None, max_dim1=None):
    """``_pin`` against element ``idx`` of list ``lst`` (per-layer buffer).

    Stores the FULL backing buffer (NOT the prefix view) so the next call's
    shape guard sees the max dims; returns the active view to the caller."""
    view, backing = _pin(lst[idx], src, max_dim0=max_dim0, max_dim1=max_dim1)
    lst[idx] = backing
    return view
