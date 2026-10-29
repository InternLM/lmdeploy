# Copyright (c) OpenMMLab. All rights reserved.
"""Inference-only indexed GDN/KDA prefill using FLA 0.5.2 chunk math.

Only the common state propagation is LMDeploy-owned. Gate scans, WY transforms and output contractions remain the
installed FLA implementations. Neither entry point gathers initial states nor stages/scatters dense final states. The
caller owns destination validation and supplies prebuilt packed chunk metadata.
"""
import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

from .indexed_chunk_state import indexed_chunk_state_fwd


@torch.inference_mode()
def indexed_chunk_gated_delta_rule(q, k, v, g, beta, pool, input_ids, final_ids, *, scale=None,
                                  state_v_first=False, cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                                  checkpoint_ids=None, checkpoint_offsets=None):
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
        checkpoint_ids=checkpoint_ids, checkpoint_offsets=checkpoint_offsets)
    return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale,
                       state_v_first=state_v_first, **common).to(q.dtype)


@torch.inference_mode()
def indexed_chunk_kda(q, k, v, g, beta, pool, input_ids, final_ids, *, A_log=None, dt_bias=None, scale=None,
                      use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                      safe_gate=False, lower_bound=None, state_v_first=True,
                      cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                      checkpoint_ids=None, checkpoint_offsets=None):
    """KDA including GLM's normalized q/k, raw gate/beta and safe-gate path."""
    if cu_seqlens is not None and (chunk_indices is None or chunk_offsets is None):
        raise ValueError('Indexed varlen prefill requires prebuilt chunk metadata.')
    # Match FLA's public input_guard for projection views. This is activation
    # layout normalization, never a gather/clone of the persistent state pool.
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    scale = q.shape[-1]**-0.5 if scale is None else scale
    common = dict(cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=64)
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
        checkpoint_ids=checkpoint_ids, checkpoint_offsets=checkpoint_offsets)
    return chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=aqk, h=h, scale=scale,
                             state_v_first=state_v_first, **common).to(q.dtype)
