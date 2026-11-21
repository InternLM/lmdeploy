# Copyright (c) OpenMMLab. All rights reserved.
"""Paged KPool decode scoring on Ascend via a Triton kernel.

Ported from vllm-ascend's GLM-Next lightning indexer
(``vllm_ascend/ops/triton/glm5_next_lightning_indexer.py``).  The kernel
reads each row's pooled length **on device**, so key loads are masked to
``pools < visible`` and read traffic follows the *actual* request lengths
instead of the captured static page-table width.  Groups beyond a row's
length - including everything a padded replay row addresses - are written
as a LOWEST fp32 sentinel, so a single ``torch.topk`` selects without extra
masking and poisoned slots can never leak a NaN into the selection.

The gather loop the caller keeps as fallback can only read at *static*
width: ``index_select`` materializes every page the table width covers, so
with a wide pool it moved hundreds of GiB of keys per decode step no
matter how short the requests were - this kernel is what makes the static
width cheap to scan.

Every per-replay value (pooled lengths, page tables) stays a device
tensor, and all Python values derive from shapes or config constants, so
the whole scores->topk sequence captures into an ``NPUGraph`` unchanged.
"""

from __future__ import annotations

import logging

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - non-Ascend deployments
    _HAS_TRITON = False

logger = logging.getLogger(__name__)

# Groups per program tile; the shapes vllm-ascend runs on Atlas NPUs.
TRITON_POOL_CHUNK_SIZE = 128
TRITON_MAX_PROGRAMS = 256
# Invalid slots must sort below every representable real score.
NEG_INF_SENTINEL = torch.finfo(torch.float32).min

# Sticky fallback switch: one failed launch (untranslated op, unusable
# stride, capture-time fault, ...) permanently reverts the caller to its
# gather loop.  Better slow and steady than a per-step exception.
_launch_disabled = False

if _HAS_TRITON:

    @triton.jit(do_not_specialize=['rows', 'num_blocks', 'width',
                                   'scores_stride_r'])
    def _kpool_paged_scores_kernel(
        qbar_ptr,
        cache_ptr,
        pool_lens_ptr,
        table_ptr,
        scores_ptr,
        rows,
        num_blocks,
        width,
        scores_stride_r,
        cache_stride_b: tl.constexpr,
        cache_stride_t: tl.constexpr,
        cache_stride_d: tl.constexpr,
        table_stride_r: tl.constexpr,
        table_stride_p: tl.constexpr,
        pools_per_block: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_POOL: tl.constexpr,
        LOWEST: tl.constexpr,
    ):
        """Score ``qbar`` against every pooled group of the static width.

        One program iterates ``(row, chunk)`` tiles grid-stride style.  The
        ``pool_start < visible`` branch keeps fully out-of-range tiles to a
        sentinel-only store, so a wide pool costs little beyond its fp32
        score row.
        """
        chunks = tl.cdiv(width, BLOCK_POOL)
        for tile in range(tl.program_id(0), rows * chunks,
                          tl.num_programs(0)):
            row, chunk = tile // chunks, tile % chunks
            pools = chunk * BLOCK_POOL + tl.arange(0, BLOCK_POOL)
            in_width = pools < width
            visible = tl.load(pool_lens_ptr + row)
            scores = tl.full((BLOCK_POOL,), LOWEST, tl.float32)
            pool_start = chunk * BLOCK_POOL
            if pool_start < visible:
                valid = in_width & (pools < visible)
                # Int32 vector offsets: one page holds pools_per_block *
                # head_dim elements, so only within-page arithmetic is
                # needed and the caller guarantees the span fits int32.
                pages = pools // pools_per_block
                blocks = tl.load(table_ptr + row * table_stride_r +
                                 pages * table_stride_p, in_width, other=0)
                # Stale or foreign page ids must not fault the gather.
                blocks = tl.minimum(tl.maximum(blocks, 0), num_blocks - 1)
                offsets = (blocks[:, None] * cache_stride_b +
                           (pools % pools_per_block)[:, None] *
                           cache_stride_t)
                dims = tl.arange(0, head_dim)
                keys = tl.load(
                    cache_ptr + offsets + dims[None, :] * cache_stride_d,
                    valid[:, None], other=0)
                query = tl.load(qbar_ptr + row * head_dim + dims)
                logits = tl.sum(keys.to(tl.float32) * query[None, :], axis=1)
                scores = tl.where(
                    valid & (logits == logits),
                    tl.minimum(tl.maximum(logits, LOWEST), -LOWEST), LOWEST)
            tl.store(scores_ptr + row * scores_stride_r + pools, scores,
                     in_width)


def paged_kpool_topk_bf16(qbar: torch.Tensor, cache: torch.Tensor,
                          pool_lens: torch.Tensor,
                          block_table: torch.Tensor, width: int, k: int):
    """Select the top ``k`` KPool groups of a fixed-shape decode batch.

    Args:
        qbar: ``[rows, head_dim]`` fp32 pooled queries (weighted head sum).
        cache: ``[pages, pools_per_block, 1, head_dim]`` bf16 pooled cache.
        pool_lens: ``[rows]`` device tensor of closed-pool counts.
        block_table: ``[rows, table_width]`` device tensor of page ids.
        width: static number of groups to scan (>= every real length).
        k: groups per row to select (already clamped to ``width``).

    Returns:
        ``(values, ids)`` with ``ids`` absolute group ids in
        ``[rows, k]``, or ``None`` when the kernel path is unusable - the
        caller then falls back to its gather loop.
    """
    global _launch_disabled
    if not _HAS_TRITON or _launch_disabled:
        return None
    rows, head_dim = qbar.shape
    k = min(k, width)
    if (rows == 0 or k <= 0 or qbar.dtype != torch.float32
            or cache.dtype != torch.bfloat16 or cache.dim() != 4
            or head_dim != cache.size(3)
            or pool_lens.device != qbar.device
            or block_table.device != qbar.device):
        return None
    # The kernel computes cache offsets in int32; the span must fit.
    cache_span = ((cache.size(0) - 1) * cache.stride(0) +
                  (cache.size(1) - 1) * cache.stride(1) +
                  (cache.size(3) - 1) * cache.stride(3))
    if cache_span > torch.iinfo(torch.int32).max:
        return None
    scores = torch.empty((rows, width), dtype=torch.float32,
                         device=qbar.device)
    chunks = triton.cdiv(width, TRITON_POOL_CHUNK_SIZE)
    try:
        _kpool_paged_scores_kernel[(min(rows * chunks,
                                        TRITON_MAX_PROGRAMS),)](
            qbar,
            cache,
            pool_lens.to(torch.int32),
            block_table,
            scores,
            rows,
            cache.size(0),
            width,
            scores.stride(0),
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            pools_per_block=cache.size(1),
            head_dim=head_dim,
            BLOCK_POOL=TRITON_POOL_CHUNK_SIZE,
            LOWEST=NEG_INF_SENTINEL,
        )
        return torch.topk(scores, k, dim=-1, sorted=False)
    except Exception as exc:  # noqa: BLE001 - any failure means: fall back
        _launch_disabled = True
        logger.warning('Ascend Triton KPool scorer disabled (%r); '
                       'selection falls back to the paged gather loop.', exc)
        return None


_expand_disabled = False


if _HAS_TRITON:

    @triton.jit(do_not_specialize=['k', 'ids_stride_r', 'out_stride_r'])
    def _kpool_expand_kernel(
        ids_ptr,
        lens_ptr,
        out_ptr,
        ids_stride_r,
        out_stride_r,
        k,
        pool_size: tl.constexpr,
        group_budget: tl.constexpr,
        out_w: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Expand selected KPool groups to logical token ids, v2 geometry.

        v1 ran one program per row with BLOCK_COL=1024 lanes and paid an
        AIV-scalarized int64 gather plus per-lane div/mod for every column
        (1121us/launch). v2 runs one program per OUTPUT GROUP (grid
        (rows, budget+1), pool_size lanes): the slot index js comes from
        program_id, the picked id is one scalar load (address clamped to
        k-1, invalidated via where), and kv %% pool is kv - groups*pool.
        Semantics are identical to v1 / the python blend, selected-column
        order included (sparse-attention reduction order is observable).

        v3.1 (row-tiled [16,pool] programs, shifts) measured 57us
        standalone but 1434us in-engine (A/B twin census d6r3d_16ranks:
        this v2 kernel 120.1us on the same tensors) - wide-tile AIV
        geometry hits an engine-only slow path; v2 stays.

        v4 (2026-11-17): with index_kpool=4 the v2 grid is (rows,
        budget+1) = (64, 513) four-lane programs - 32832 launches of
        almost no work, 119.2us/step in the d9 census, 93% serial on the
        main stream.  v4 flattens each row into BLOCK=1024-lane programs
        (grid (rows, ceil(out_w/BLOCK))); js/local come from shifts, and
        the picked-id load becomes a vector gather (v1's killer was a
        2051-wide scalarized gather per single row, not gathering per
        se).  Standalone 51.2us vs v2 114.7us, bitwise-identical output
        over short- and long-history regimes.
        """
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = col < out_w
        js = col // pool_size
        local = col - js * pool_size
        kv = tl.load(lens_ptr + row).to(tl.int32)
        groups = kv // pool_size
        tail_cnt = kv - groups * pool_size
        hist_slots = tl.minimum(groups, group_budget)
        off = tl.minimum(js, k - 1)
        picked = tl.load(ids_ptr + row * ids_stride_r + off, mask=m,
                         other=-1).to(tl.int32)
        picked = tl.where(js < k, picked, -1)
        is_long = groups > group_budget
        selected = tl.where(is_long, picked, js)
        valid = (js < hist_slots) & (selected >= 0) & (selected < groups)
        hist = selected * pool_size + local
        tail_here = (js == hist_slots) & (local < tail_cnt)
        val = tl.where(valid, hist,
                       tl.where(tail_here, groups * pool_size + local, -1))
        tl.store(out_ptr + row * out_stride_r + col, val.to(tl.int32),
                 mask=m)


def paged_kpool_expand_npu(best_ids: torch.Tensor,
                           kv_seqlens: torch.Tensor,
                           pool_size: int,
                           topk: int):
    """Blend + expand + tail-append a decode batch's KPool selection.

    Args:
        best_ids: ``[rows, k]`` int64 absolute group ids in selection
            order (``k`` may be 0 or below ``topk // pool_size``).
        kv_seqlens: ``[rows]`` int64 device tensor of KV lengths.
        pool_size / topk: the indexer geometry constants.

    Returns:
        ``[rows, topk + pool_size - 1]`` int32 logical token ids, or
        ``None`` when the kernel path is unusable - the caller then falls
        back to its python blend.
    """
    global _expand_disabled
    if not _HAS_TRITON or _expand_disabled:
        return None
    if (best_ids.dtype != torch.int64 or best_ids.dim() != 2
            or kv_seqlens.dtype != torch.int64
            or kv_seqlens.device != best_ids.device
            or topk % pool_size != 0):
        return None
    rows, k = best_ids.shape
    if k == 0:
        # An empty ids tensor has a null data_ptr and Ascend masked loads
        # still touch the address, so hand the kernel a valid -1 column;
        # the (js < k) mask keeps it out of every result.
        best_ids = torch.full((rows, 1), -1, dtype=torch.int64,
                              device=best_ids.device)
        k = 1
    out_w = topk + pool_size - 1
    if rows == 0:
        return torch.empty((0, out_w), dtype=torch.int32,
                           device=best_ids.device)
    out = torch.empty((rows, out_w), dtype=torch.int32,
                      device=best_ids.device)
    try:
        # v4 geometry: BLOCK flat output lanes per program instead of one
        # program per (row, 4-lane output group); see the kernel docstring.
        _kpool_expand_kernel[(rows, triton.cdiv(out_w, 1024))](
            best_ids,
            kv_seqlens,
            out,
            best_ids.stride(0),
            out.stride(0),
            k,
            pool_size=pool_size,
            group_budget=topk // pool_size,
            out_w=out_w,
            BLOCK=1024,
        )
        return out
    except Exception as exc:  # noqa: BLE001 - any failure means: fall back
        _expand_disabled = True
        logger.warning('Ascend Triton KPool expander disabled (%r); '
                       'selection falls back to the python blend.', exc)
        return None


_UPDATE_DISABLED = False

if _HAS_TRITON:

    @triton.jit(do_not_specialize=['states', 'entries', 'bt_stride',
                                   'tail_stride_r'])
    def _kpool_update_kernel(
        keys_ptr, scores_ptr, tail_k_ptr, tail_s_ptr,
        state_ids_ptr, hist_ptr, cache_ptr, bt_ptr, ape_ptr,
        states, entries, bt_stride, tail_stride_r,
        HD: tl.constexpr,
    ):
        """One program per batch row; POOL=4 slots unrolled to mirror the
        eager slot loop's sequential fp32 accumulation order exactly."""
        POOL: tl.constexpr = 4
        row = tl.program_id(0)
        d = tl.arange(0, HD)
        sid = tl.load(state_ids_ptr + row)
        hist = tl.load(hist_ptr + row)
        valid = sid >= 0
        safe = tl.maximum(sid, 0)
        h = tl.maximum(hist, 0)
        prev_len = h % POOL
        group = h // POOL
        should_close = valid & (prev_len == POOL - 1)

        base = safe * tail_stride_r
        k0 = tl.load(tail_k_ptr + base + 0 * HD + d)
        k1 = tl.load(tail_k_ptr + base + 1 * HD + d)
        k2 = tl.load(tail_k_ptr + base + 2 * HD + d)
        k3 = tl.load(tail_k_ptr + base + 3 * HD + d)
        s0 = tl.load(tail_s_ptr + base + 0 * HD + d)
        s1 = tl.load(tail_s_ptr + base + 1 * HD + d)
        s2 = tl.load(tail_s_ptr + base + 2 * HD + d)
        s3 = tl.load(tail_s_ptr + base + 3 * HD + d)
        new_k = tl.load(keys_ptr + row * HD + d)
        new_s = tl.load(scores_ptr + row * HD + d)

        zero = tl.zeros_like(k0)
        k0 = tl.where(prev_len > 0, k0, zero)
        s0 = tl.where(prev_len > 0, s0, zero)
        k0 = tl.where(prev_len == 0, new_k, k0)
        s0 = tl.where(prev_len == 0, new_s, s0)
        k1 = tl.where(prev_len > 1, k1, zero)
        s1 = tl.where(prev_len > 1, s1, zero)
        k1 = tl.where(prev_len == 1, new_k, k1)
        s1 = tl.where(prev_len == 1, new_s, s1)
        k2 = tl.where(prev_len > 2, k2, zero)
        s2 = tl.where(prev_len > 2, s2, zero)
        k2 = tl.where(prev_len == 2, new_k, k2)
        s2 = tl.where(prev_len == 2, new_s, s2)
        k3 = tl.where(prev_len == 3, new_k, zero)
        s3 = tl.where(prev_len == 3, new_s, zero)

        a0 = tl.load(ape_ptr + 0 * HD + d).to(tl.float32)
        a1 = tl.load(ape_ptr + 1 * HD + d).to(tl.float32)
        a2 = tl.load(ape_ptr + 2 * HD + d).to(tl.float32)
        a3 = tl.load(ape_ptr + 3 * HD + d).to(tl.float32)
        sc0 = s0.to(tl.float32) + a0
        sc1 = s1.to(tl.float32) + a1
        sc2 = s2.to(tl.float32) + a2
        sc3 = s3.to(tl.float32) + a3
        m = tl.maximum(tl.maximum(sc0, sc1), tl.maximum(sc2, sc3))
        p0 = tl.exp(sc0 - m)
        p1 = tl.exp(sc1 - m)
        p2 = tl.exp(sc2 - m)
        p3 = tl.exp(sc3 - m)
        den = ((p0 + p1) + p2) + p3
        acc = ((k0.to(tl.float32) * p0 + k1.to(tl.float32) * p1)
               + k2.to(tl.float32) * p2) + k3.to(tl.float32) * p3
        pooled = (acc / den).to(tl.bfloat16)

        if valid:
            nk0 = tl.where(should_close, zero, k0)
            nk1 = tl.where(should_close, zero, k1)
            nk2 = tl.where(should_close, zero, k2)
            nk3 = tl.where(should_close, zero, k3)
            tl.store(tail_k_ptr + base + 0 * HD + d, nk0)
            tl.store(tail_k_ptr + base + 1 * HD + d, nk1)
            tl.store(tail_k_ptr + base + 2 * HD + d, nk2)
            tl.store(tail_k_ptr + base + 3 * HD + d, nk3)
            ns0 = tl.where(should_close, zero, s0)
            ns1 = tl.where(should_close, zero, s1)
            ns2 = tl.where(should_close, zero, s2)
            ns3 = tl.where(should_close, zero, s3)
            tl.store(tail_s_ptr + base + 0 * HD + d, ns0)
            tl.store(tail_s_ptr + base + 1 * HD + d, ns1)
            tl.store(tail_s_ptr + base + 2 * HD + d, ns2)
            tl.store(tail_s_ptr + base + 3 * HD + d, ns3)
            if should_close:
                page_col = group // entries
                page_col = tl.minimum(tl.maximum(page_col, 0), bt_stride - 1)
                page = tl.load(bt_ptr + row * bt_stride + page_col)
                cslot = group % entries
                tl.store(cache_ptr + (page * entries + cslot) * HD + d,
                         pooled)


def block_offsets_bad(bt, rows):
    return bt.dim() != 2 or bt.size(0) != rows or bt.stride(1) != 1


def paged_kpool_update_npu(keys, scores, tail_k, tail_s, state_ids,
                           hist, cache, block_table, ape):
    """Fused one-token KPool decode update; None -> caller keeps eager loop.

    Only the plain decode shape is handled: steps==1 (no MTP ring), 3D
    BF16 tail states, paged BF16 cache with one head per entry.
    """
    global _UPDATE_DISABLED
    if not _HAS_TRITON or _UPDATE_DISABLED:
        return False
    if (keys.dtype != torch.bfloat16 or keys.dim() != 2
            or keys.size(1) & (keys.size(1) - 1)
            or scores.shape != keys.shape or scores.dtype != torch.bfloat16
            or tail_k.dtype != torch.bfloat16 or tail_k.dim() != 3
            or tail_k.shape[1] != 4 or tail_s.shape != tail_k.shape
            or state_ids.shape != (keys.size(0),)
            or hist.shape != state_ids.shape
            or cache.dtype != torch.bfloat16 or cache.dim() != 4
            or cache.size(2) != 1 or cache.size(3) != keys.size(1)
            or block_offsets_bad(block_table, keys.size(0))
            or ape.shape != (4, keys.size(1))
            or keys.stride(0) != keys.size(1) or keys.stride(1) != 1
            or scores.stride(0) != keys.size(1) or scores.stride(1) != 1
            or tail_k.stride(1) != keys.size(1) or tail_k.stride(2) != 1
            or tail_s.stride(0) != tail_k.stride(0)
            or tail_s.stride(1) != keys.size(1) or tail_s.stride(2) != 1
            or cache.stride(3) != 1 or cache.stride(1) != keys.size(1)):
        return False
    try:
        _kpool_update_kernel[(keys.size(0),)](
            keys, scores, tail_k, tail_s, state_ids, hist, cache,
            block_table, ape,
            tail_k.size(0), cache.size(1), block_table.stride(0),
            tail_k.stride(0), HD=keys.size(1))
        return True
    except Exception as exc:  # noqa: BLE001
        _UPDATE_DISABLED = True
        logger.warning('Ascend Triton KPool update disabled (%r); '
                       'decode falls back to the eager loop.', exc)
        return False

