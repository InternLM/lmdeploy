# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def fused_recurrent_gated_delta_rule_fwd(H,
                                         K,
                                         HV,
                                         V,
                                         q_stride: Sequence[int],
                                         k_stride: Sequence[int],
                                         v_stride: Sequence[int],
                                         state_stride: Sequence[int],
                                         scale,
                                         dtype,
                                         g_dtype=None,
                                         beta_dtype=None,
                                         use_g: bool = False,
                                         use_beta: bool = False,
                                         use_qk_l2norm_in_kernel: bool = False,
                                         output_final_state: bool = False,
                                         use_state_indices: bool = False,
                                         num_warps: int = 1):

    num_threads = num_warps * 32
    num_bits = T.DataType(dtype).bits
    num_elems = 128 // num_bits
    warp_size = 32
    k_per_thr = T.ceildiv(K, warp_size)
    v_per_warp = num_elems
    num_waves = 2
    v_per_cta = v_per_warp * num_warps * num_waves

    B = T.dynamic('B')
    N = B if not use_state_indices else T.dynamic('N')

    # dtype
    if g_dtype is None:
        g_dtype = dtype
    if beta_dtype is None:
        beta_dtype = dtype

    @T.prim_func
    def fused_recurrent_gated_delta_rule_main(
        Query: T.StridedTensor([B, H, K], dtype=dtype, strides=q_stride),
        Key: T.StridedTensor([B, H, K], dtype=dtype, strides=k_stride),
        Value: T.StridedTensor([B, HV, V], dtype=dtype, strides=v_stride),
        Out: T.Tensor([B, HV, V], dtype=dtype),
        G: T.Tensor([B, HV], dtype=g_dtype),
        Beta: T.Tensor([B, HV], dtype=beta_dtype),
        State: T.StridedTensor([N, HV, K, V], dtype=dtype, strides=state_stride),
        StateIndices: T.Tensor([B], dtype=torch.int64) = None,
    ):
        with T.Kernel(T.ceildiv(V, v_per_cta), B * HV, threads=num_threads) as (v_start, bhv_idx):
            tidx = T.get_thread_binding(0)
            b_id = bhv_idx // HV
            hv_id = bhv_idx % HV
            h_id = hv_id // (HV // H)
            warp_id = tidx // warp_size
            lane_id = tidx % warp_size
            k_off = lane_id * k_per_thr

            # state_idx
            if use_state_indices:
                state_id = StateIndices[b_id]
            else:
                state_id = b_id

            # load states
            h_smem = T.alloc_shared([K, v_per_cta], dtype)
            for i, j in T.Parallel(K, v_per_cta):
                v_idx = v_start * v_per_cta + j
                if v_idx < V:
                    h_smem[i, j] = State[state_id, hv_id, i, v_idx]
                else:
                    h_smem[i, j] = 0.0

            # load q, k, g, beta
            q_local = T.alloc_local([k_per_thr], T.float32)
            k_local = T.alloc_local([k_per_thr], T.float32)
            for i in T.Parallel(k_per_thr):
                k_idx = (k_off + i) % K
                q_local[i] = Query[b_id, h_id, k_idx]
            for i in T.Parallel(k_per_thr):
                k_idx = (k_off + i) % K
                k_local[i] = Key[b_id, h_id, k_idx]

            # normalize
            if use_qk_l2norm_in_kernel:
                k_sum = T.alloc_var(T.float32)
                q_sum = T.alloc_var(T.float32)
                k_sum = 0
                q_sum = 0
                for i in T.Unroll(k_per_thr):
                    k_sum += k_local[i] * k_local[i]
                    q_sum += q_local[i] * q_local[i]
                k_sum = T.warp_reduce_sum(k_sum)
                q_sum = T.warp_reduce_sum(q_sum)
                k_norm = T.rsqrt(k_sum + 1e-6)
                q_norm = T.rsqrt(q_sum + 1e-6)
                for i in T.Unroll(k_per_thr):
                    k_local[i] = k_local[i] * k_norm
                    q_local[i] = q_local[i] * q_norm

            for i in T.Parallel(k_per_thr):
                q_local[i] = q_local[i] * scale

            # load g, beta
            if use_g:
                g = T.cast(G[b_id, hv_id], T.float32)
            else:
                g = 0.0
            g_exp = T.exp(g)
            if use_beta:
                beta = T.cast(Beta[b_id, hv_id], T.float32)
            else:
                beta = 1.0

            for wave_id in range(num_waves):
                v_warp_off = wave_id * num_warps * v_per_warp + warp_id * v_per_warp
                v_off = v_start * v_per_cta + v_warp_off

                # load v
                v_local = T.alloc_local([v_per_warp], dtype)
                for i in T.Parallel(v_per_warp):
                    v_idx = (v_off + i) % V
                    v_local[i] = Value[b_id, hv_id, v_idx]

                # load states local
                h_local = T.alloc_local([k_per_thr, v_per_warp], T.float32)
                for j in T.Unroll(k_per_thr):
                    k_idx = k_off + j
                    for i in T.Vectorized(v_per_warp):
                        h_local[j, i] = h_smem[k_idx, v_warp_off + i]

                # update states
                for i in T.Unroll(v_per_warp):
                    hk = T.alloc_var(T.float32)
                    hk = 0
                    for j in T.Unroll(k_per_thr):
                        h_local[j, i] = h_local[j, i] * g_exp
                        hk += h_local[j, i] * k_local[j]
                    hk = T.warp_reduce_sum(hk)
                    v = (v_local[i] - hk) * beta
                    for j in T.Unroll(k_per_thr):
                        h_local[j, i] = h_local[j, i] + k_local[j] * v

                # store states
                if output_final_state and state_id >= 0:
                    for j in T.Unroll(k_per_thr):
                        if (k_off + j) < K:
                            for i in T.Vectorized(v_per_warp):
                                if v_off + i < V:
                                    State[state_id, hv_id, k_off + j, v_off + i] = h_local[j, i]

                # compute output
                o_local = T.alloc_local([v_per_warp], dtype)
                for i in T.Unroll(v_per_warp):
                    # o = q * h
                    o = T.alloc_var(T.float32)
                    o = 0.0
                    for j in T.Unroll(k_per_thr):
                        o += q_local[j] * h_local[j, i]
                    o = T.warp_reduce_sum(o)
                    o_local[i] = o

                if lane_id == 0 and state_id >= 0:
                    for i in T.Vectorized(v_per_warp):
                        v_idx = (v_off + i)
                        if v_idx < V:
                            Out[b_id, hv_id, v_idx] = o_local[i]

    return fused_recurrent_gated_delta_rule_main


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, HV, V]
        g: [B, T, HV], optional
        beta: [B, T, HV], optional
        scale: float, optional
        initial_state: [N, HV, K, V], optional, if state_indices is not proviced, N=B
        use_qk_l2norm_in_kernel: whether to apply l2 normalization on q and k in the kernel
        state_indices: [B], optional, the indices to update in the recurrent state, required

    Returns:
        o: [B, T, HV, V]
        final_state: [N, HV, K, V] if output_final_state else None
    """
    _, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    assert T == 1, 'Only support T=1 for now'
    if scale is None:
        scale = 1 / (q.shape[-1]**0.5)
    g_dtype = torch.float32
    beta_dtype = torch.float32
    if g is not None:
        assert g.is_contiguous()
        g_dtype = g.dtype
    if beta is not None:
        assert beta.is_contiguous()
        beta_dtype = beta.dtype
    if state_indices is not None:
        assert state_indices.is_contiguous()
        assert initial_state is not None, 'initial_state is required when state_indices is provided'
        assert state_indices.shape == (q.shape[0], )

    o = torch.empty_like(v)
    final_state = initial_state
    if final_state is not None:
        state_stride = final_state.stride()
    else:
        state_stride = (0, 0, 0, 0)

    q, k, v = q[:, 0], k[:, 0], v[:, 0]
    g = g[..., 0, :] if g is not None else None
    beta = beta[..., 0, :] if beta is not None else None

    num_warps = 4
    kernel = fused_recurrent_gated_delta_rule_fwd(H,
                                                  K,
                                                  HV,
                                                  V,
                                                  q_stride=q.stride(),
                                                  k_stride=k.stride(),
                                                  v_stride=v.stride(),
                                                  state_stride=state_stride,
                                                  scale=scale,
                                                  dtype=q.dtype,
                                                  g_dtype=g_dtype,
                                                  beta_dtype=beta_dtype,
                                                  use_g=g is not None,
                                                  use_beta=beta is not None,
                                                  use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                                                  output_final_state=output_final_state,
                                                  use_state_indices=state_indices is not None,
                                                  num_warps=num_warps)

    kernel(q, k, v, o[:, 0], g, beta, final_state, state_indices)

    if not output_final_state:
        final_state = None
    return o, final_state
