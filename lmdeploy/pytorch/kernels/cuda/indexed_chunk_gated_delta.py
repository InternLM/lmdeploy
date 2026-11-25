# Copyright (c) OpenMMLab. All rights reserved.
"""Inference-only indexed GDN/KDA prefill using FLA 0.5.2 chunk math.

State propagation is LMDeploy-owned. Single-token KDA fuses the input transforms; other gate scans, WY transforms and
output contractions remain the installed FLA implementations. Neither entry point gathers initial states nor
stages/scatters dense final states. The caller owns destination validation and supplies prebuilt packed chunk metadata.
"""
import torch
import triton
import triton.language as tl
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp

from .indexed_chunk_state import indexed_chunk_state_fwd


@triton.jit
def _single_token_kda_inputs_kernel(Q, K, G, Beta, A, Bias, QNorm, KNorm, Gate, BetaSigmoid,
                                    lower_bound, scale,
                                    ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                                    HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    column = tl.arange(0, BLOCK)
    head = row % HEADS
    mask = (row[:, None] < ROWS) & (column[None, :] < DIM)
    offsets = row[:, None] * DIM + column[None, :]
    q = tl.load(Q + offsets, mask, 0).to(tl.float32)
    k = tl.load(K + offsets, mask, 0).to(tl.float32)
    # Keep FLA's normalization and output-dtype rounding before WY transforms.
    q = q * (1 / tl.sqrt(tl.sum(q * q, 1) + 1e-6))[:, None]
    k = k * (1 / tl.sqrt(tl.sum(k * k, 1) + 1e-6))[:, None]
    tl.store(QNorm + offsets, q, mask)
    tl.store(KNorm + offsets, k, mask)
    gate = tl.load(G + offsets, mask, 0).to(tl.float32)
    if HAS_BIAS:
        gate += tl.load(Bias + head[:, None] * DIM + column[None, :], mask, 0).to(tl.float32)
    a = tl.load(A + head, row < ROWS, 0).to(tl.float32)
    # Each sequence has one token, so the chunk-local prefix sum is identity.
    gate = lower_bound * tl.sigmoid(exp(a)[:, None] * gate)
    tl.store(Gate + offsets, gate * scale, mask)
    beta = tl.load(Beta + row, row < ROWS, 0).to(tl.float32)
    tl.store(BetaSigmoid + row, tl.sigmoid(beta), row < ROWS)


def _single_token_kda_inputs(q, k, g, beta, a_log, dt_bias, lower_bound):
    """Fuse FLA's four input transforms without changing chunk/state math."""
    # Match the gate transform's input_guard for parameter views as well.
    a_log = a_log.contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.contiguous()
    q_norm = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_norm = torch.empty_like(q_norm)
    gate = torch.empty(g.shape, device=g.device, dtype=torch.float32)
    beta_sigmoid = torch.empty(beta.shape, device=beta.device, dtype=torch.float32)
    rows, heads, dim = q.shape[1] * q.shape[2], q.shape[2], q.shape[3]
    _single_token_kda_inputs_kernel[(triton.cdiv(rows, 8),)](
        q, k, g, beta, a_log, dt_bias, q_norm, k_norm, gate, beta_sigmoid,
        lower_bound, RCP_LN2,
        ROWS=rows, HEADS=heads, DIM=dim, HAS_BIAS=dt_bias is not None,
        BLOCK=triton.next_power_of_2(dim), num_warps=4)
    return q_norm, k_norm, gate, beta_sigmoid


@torch.inference_mode()
def indexed_chunk_gated_delta_rule(q, k, v, g, beta, pool, input_ids, final_ids, *, scale=None,
                                  state_v_first=False, cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                                  checkpoint_ids=None, checkpoint_offsets=None, checkpoint_pool=None):
    """GDN with preprocessed q/k, log decay g and sigmoid beta (Qwen path)."""
    if cu_seqlens is not None and (chunk_indices is None or chunk_offsets is None):
        raise ValueError('Indexed varlen prefill requires prebuilt chunk metadata.')
    # Match FLA's public input_guard for projection views. This is activation
    # layout normalization, never a gather/clone of the persistent state pool.
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    scale = q.shape[-1]**-0.5 if scale is None else scale
    common = dict(cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=64)
    g = chunk_local_cumsum(g, scale=RCP_LN2, **common)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, **common)
    h, v_new = indexed_chunk_state_fwd(
        k, w, u, pool, input_ids, final_ids, g=g, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
        checkpoint_ids=checkpoint_ids, checkpoint_offsets=checkpoint_offsets, checkpoint_pool=checkpoint_pool)
    return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale,
                       state_v_first=state_v_first, **common).to(q.dtype)


@torch.inference_mode()
def indexed_chunk_kda(q, k, v, g, beta, pool, input_ids, final_ids, *, A_log=None, dt_bias=None, scale=None,
                      use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                      safe_gate=False, lower_bound=None, state_v_first=True,
                      cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                      checkpoint_ids=None, checkpoint_offsets=None, checkpoint_pool=None):
    """KDA including GLM's normalized q/k, raw gate/beta and safe-gate path."""
    if cu_seqlens is not None and (chunk_indices is None or chunk_offsets is None):
        raise ValueError('Indexed varlen prefill requires prebuilt chunk metadata.')
    # Match FLA's public input_guard for projection views. This is activation
    # layout normalization, never a gather/clone of the persistent state pool.
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    scale = q.shape[-1]**-0.5 if scale is None else scale
    common = dict(cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=64)
    # Token, sequence and chunk counts agree only when every packed row has
    # one token. Check metadata shapes without synchronizing device lengths.
    single_token = (cu_seqlens is not None and q.shape[0] == 1
                    and q.shape[1] == cu_seqlens.numel() - 1 == chunk_indices.shape[0])
    # GVA has separate query/value head counts; keep its FLA transforms.
    if (single_token and q.shape[2] == v.shape[2] and use_qk_l2norm_in_kernel and use_gate_in_kernel
            and use_beta_sigmoid_in_kernel and lower_bound is not None):
        q, k, g, beta = _single_token_kda_inputs(q, k, g, beta, A_log, dt_bias, lower_bound)
    else:
        if use_qk_l2norm_in_kernel:
            q, _ = l2norm_fwd(q)
            k, _ = l2norm_fwd(k)
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta, scale=1.0)
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(g=g, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2,
                                      lower_bound=lower_bound, **common)
        else:
            g = chunk_local_cumsum(g, scale=RCP_LN2, **common)
    w, u, _, kg, aqk, _ = chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale,
                                            safe_gate=safe_gate, disable_recompute=False, **common)
    h, v_new = indexed_chunk_state_fwd(
        kg, w, u, pool, input_ids, final_ids, gk=g, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
        checkpoint_ids=checkpoint_ids, checkpoint_offsets=checkpoint_offsets, checkpoint_pool=checkpoint_pool)
    return chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=aqk, h=h, scale=scale,
                             state_v_first=state_v_first, **common).to(q.dtype)
