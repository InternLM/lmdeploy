"""Thin wrappers around the TurboMind DFlash2 kernel bindings."""

import torch

try:
    from lmdeploy.turbomind import _tm
except Exception:  # pragma: no cover - the extension may not be built
    _tm = None


def is_available():
    return (_tm is not None and hasattr(_tm, 'grouped_conv') and hasattr(_tm, 'dflash2_attention'))


def _stream():
    return torch.cuda.current_stream().cuda_stream


def grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps, out=None):
    hidden = hidden.contiguous()
    delta = delta.contiguous()
    base = base.contiguous()
    if out is None:
        out = torch.empty_like(hidden)
    _tm.grouped_conv(hidden, delta, base, out, block_size, num_groups, group_size, taps, _stream())
    return out


def candidate_selector_scores(hidden, unary_logits, predecessor_codebook, successor_codebook, candidate_ids,
                              anchor_token_ids, top_k, out=None):
    hidden = hidden.contiguous()
    unary_logits = unary_logits.contiguous().float()
    predecessor_codebook = predecessor_codebook.contiguous()
    successor_codebook = successor_codebook.contiguous()
    candidate_ids = candidate_ids.contiguous().int()
    anchor_token_ids = anchor_token_ids.contiguous().int()
    if out is None:
        batch, length = hidden.shape[0], hidden.shape[1]
        out = torch.empty((batch, length, top_k, top_k), dtype=torch.float32, device=hidden.device)
    _tm.candidate_selector_scores(hidden, unary_logits, predecessor_codebook, successor_codebook, candidate_ids,
                                  anchor_token_ids, out, top_k, _stream())
    return out


def dflash2_attention(q, k_ctx, v_ctx, k_blk, v_blk, scale, window, out=None):
    q, k_ctx, v_ctx, k_blk, v_blk = (t.contiguous() for t in (q, k_ctx, v_ctx, k_blk, v_blk))
    if out is None:
        out = torch.empty_like(q)
    _tm.dflash2_attention(q, k_ctx, v_ctx, k_blk, v_blk, out, scale, window, _stream())
    return out


def dflash2_attention_int4(q, k_ctx_data, k_ctx_param, v_ctx_data, v_ctx_param, k_blk, v_blk, scale, window,
                           out=None):
    q = q.contiguous()
    k_ctx_data = k_ctx_data.contiguous()
    k_ctx_param = k_ctx_param.contiguous()
    v_ctx_data = v_ctx_data.contiguous()
    v_ctx_param = v_ctx_param.contiguous()
    k_blk = k_blk.contiguous()
    v_blk = v_blk.contiguous()
    if out is None:
        out = torch.empty_like(q)
    _tm.dflash2_attention_int4(q, k_ctx_data, k_ctx_param, v_ctx_data, v_ctx_param, k_blk, v_blk, out, scale, window,
                               _stream())
    return out
