# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import tilelang
import tilelang.language as T
import torch


@T.macro
def normalize_qk(k_local: T.Buffer, q_local: T.Buffer, k_per_thr: int) -> None:
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


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def fused_recurrent_gated_delta_rule_fwd(SEQLEN,
                                         H,
                                         K,
                                         HV,
                                         V,
                                         NUM_STATE,
                                         q_stride: Sequence[int],
                                         k_stride: Sequence[int],
                                         v_stride: Sequence[int],
                                         state_stride: Sequence[int],
                                         scale,
                                         dtype,
                                         state_dtype,
                                         g_dtype=None,
                                         beta_dtype=None,
                                         use_g: bool = False,
                                         use_beta: bool = False,
                                         use_qk_l2norm_in_kernel: bool = False,
                                         output_final_state: bool = False,
                                         use_state_indices: bool = False,
                                         is_circular_buffer: bool = False,
                                         num_warps: int = 1):

    num_threads = num_warps * 32
    state_num_bits = T.DataType(state_dtype).bits
    data_num_bits = T.DataType(dtype).bits
    state_vec_width = 128 // state_num_bits
    data_vec_width = 128 // data_num_bits
    warp_size = 32
    k_per_thr = T.ceildiv(K, warp_size)
    v_per_warp = max(state_vec_width, data_vec_width, 8)
    # Target v_per_cta >= V to minimize grid_V blocks.
    # More waves means fewer blocks but more sequential wave iterations.
    target_v_per_cta = max(V, v_per_warp * num_warps * 2)
    num_waves = T.ceildiv(target_v_per_cta, v_per_warp * num_warps)
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
        Query: T.StridedTensor([B, SEQLEN, H, K], dtype=dtype, strides=q_stride),
        Key: T.StridedTensor([B, SEQLEN, H, K], dtype=dtype, strides=k_stride),
        Value: T.StridedTensor([B, SEQLEN, HV, V], dtype=dtype, strides=v_stride),
        Out: T.Tensor([B, SEQLEN, HV, V], dtype=dtype),
        G: T.Tensor([B, SEQLEN, HV], dtype=g_dtype),
        Beta: T.Tensor([B, SEQLEN, HV], dtype=beta_dtype),
        State: T.StridedTensor([N, NUM_STATE, HV, K, V], dtype=state_dtype, strides=state_stride),
        StateIndices: T.Tensor([B], dtype=torch.int64) = None,
        CacheSeqlens: T.Tensor([B], dtype=torch.int32) = None,
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

            if is_circular_buffer:
                state_seq_id = CacheSeqlens[b_id] % NUM_STATE
                state_update_id = T.alloc_var(T.int32)
                state_update_id = (state_seq_id + 1) % NUM_STATE
            else:
                state_seq_id = 0
                state_update_id = 0

            # load states
            h_smem = T.alloc_shared([K, v_per_cta], state_dtype)
            T.annotate_layout({h_smem: tilelang.layout.make_swizzled_layout(h_smem)})
            for i, j in T.Parallel(K, v_per_cta):
                v_idx = v_start * v_per_cta + j
                if v_idx < V:
                    h_smem[i, j] = State[state_id, state_seq_id, hv_id, i, v_idx]
                else:
                    h_smem[i, j] = 0.0

            # since H is more heavy than qkv, we would put wave loop outside
            for wave_id in range(num_waves):
                # load states local

                v_warp_off = wave_id * num_warps * v_per_warp + warp_id * v_per_warp
                v_off = v_start * v_per_cta + v_warp_off
                h_local = T.alloc_local([k_per_thr, v_per_warp], T.float32)
                if is_circular_buffer:
                    state_update_id = (state_seq_id + 1) % NUM_STATE
                for j in T.Unroll(k_per_thr):
                    k_idx = k_off + j
                    for vg in T.Unroll(v_per_warp // state_vec_width):
                        for i in T.Vectorized(state_vec_width):
                            idx = vg * state_vec_width + i
                            h_local[j, idx] = h_smem[k_idx, v_warp_off + idx]

                for seq_id in range(SEQLEN):
                    # load q, k, g, beta
                    q_local = T.alloc_local([k_per_thr], T.float32)
                    k_local = T.alloc_local([k_per_thr], T.float32)
                    for i in T.Vectorized(k_per_thr):
                        k_idx = (k_off + i) % K
                        q_local[i] = Query[b_id, seq_id, h_id, k_idx]
                    for i in T.Vectorized(k_per_thr):
                        k_idx = (k_off + i) % K
                        k_local[i] = Key[b_id, seq_id, h_id, k_idx]

                    # normalize
                    if use_qk_l2norm_in_kernel:
                        normalize_qk(k_local, q_local, k_per_thr)

                    for i in T.Vectorized(k_per_thr):
                        q_local[i] = q_local[i] * scale

                    # load g, beta
                    if use_g:
                        g = T.cast(G[b_id, seq_id, hv_id], T.float32)
                    else:
                        g = 0.0
                    g_exp = T.exp(g)
                    if use_beta:
                        beta = T.cast(Beta[b_id, seq_id, hv_id], T.float32)
                    else:
                        beta = 1.0

                    # load v
                    v_local = T.alloc_local([v_per_warp], dtype)
                    for vg in T.Unroll(v_per_warp // data_vec_width):
                        for i in T.Vectorized(data_vec_width):
                            idx = vg * data_vec_width + i
                            v_idx = (v_off + idx) % V
                            v_local[idx] = Value[b_id, seq_id, hv_id, v_idx]

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
                        if is_circular_buffer:
                            for j in T.Unroll(k_per_thr):
                                if (k_off + j) < K:
                                    for vg in T.Unroll(v_per_warp // state_vec_width):
                                        for i in T.Vectorized(state_vec_width):
                                            idx = vg * state_vec_width + i
                                            if v_off + idx < V:
                                                State[state_id, state_update_id, hv_id, k_off + j,
                                                      v_off + idx] = h_local[j, idx]
                            state_update_id = (state_update_id + 1) % NUM_STATE

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
                        for vg in T.Unroll(v_per_warp // data_vec_width):
                            for i in T.Vectorized(data_vec_width):
                                idx = vg * data_vec_width + i
                                v_idx = (v_off + idx)
                                if v_idx < V:
                                    Out[b_id, seq_id, hv_id, v_idx] = o_local[idx]

                # write h_local back to h_smem for coalesced global store
                if output_final_state and state_id >= 0 and not is_circular_buffer:
                    for j in T.Unroll(k_per_thr):
                        k_idx = k_off + j
                        for vg in T.Unroll(v_per_warp // state_vec_width):
                            for i in T.Vectorized(state_vec_width):
                                idx = vg * state_vec_width + i
                                h_smem[k_idx, v_warp_off + idx] = h_local[j, idx]

            # coalesced state writeback via shared memory
            if output_final_state and state_id >= 0 and not is_circular_buffer:
                for i, j in T.Parallel(K, v_per_cta):
                    v_idx = v_start * v_per_cta + j
                    if v_idx < V:
                        State[state_id, state_update_id, hv_id, i, v_idx] = h_smem[i, j]

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
    cache_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, HV, V]
        g: [B, T, HV], optional
        beta: [B, T, HV], optional
        scale: float, optional
        initial_state: Tensor, optional. Recurrent state with shape
            [N, HV, K, V] or [N, NUM_STATE, HV, K, V]. If ``state_indices``
            is not provided, N = B. When using circular buffers
            (i.e. ``cache_seqlens`` is not None), ``NUM_STATE`` specifies
            the number of state slots per sequence (e.g. buffer size).
        use_qk_l2norm_in_kernel: whether to apply l2 normalization on q and k in the kernel
        state_indices: [B], optional, the indices to update in the recurrent state, required
        cache_seqlens: [B], optional, the cached sequence lengths for each batch element
    Returns:
        o: [B, T, HV, V]
        final_state: Recurrent state if ``output_final_state`` is True,
            otherwise None. The returned state has shape [N, HV, K, V] if
            the input ``initial_state`` was 4D, or [N, NUM_STATE, HV, K, V]
            if a 5D state was provided (e.g. when using circular buffers).
    """
    # T is imported as tilelang.language, use seqlen instead
    _, seqlen, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
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
    if cache_seqlens is not None:
        assert cache_seqlens.is_contiguous(), 'cache_seqlens must be contiguous'
        assert cache_seqlens.shape == (q.shape[0], ), 'cache_seqlens must have shape (B,) where B is the batch size'
        assert cache_seqlens.dtype == torch.int32, 'cache_seqlens must have dtype torch.int32'
        assert cache_seqlens.device == q.device, 'cache_seqlens must be on the same device as q'

    o = torch.empty_like(v)
    final_state = initial_state
    state_dtype = q.dtype
    if final_state is not None:
        state_dim = final_state.dim()
        # expand dim
        if state_dim == 4:
            final_state = final_state.unsqueeze(1)
        state_stride = final_state.stride()
        state_dtype = final_state.dtype

        # set and check num states
        num_states = final_state.shape[1]
    else:
        state_dim = 4
        state_stride = (0, 0, 0, 0, 0)
        num_states = 1

    num_warps = 4
    kernel = fused_recurrent_gated_delta_rule_fwd(seqlen,
                                                  H,
                                                  K,
                                                  HV,
                                                  V,
                                                  NUM_STATE=num_states,
                                                  q_stride=q.stride(),
                                                  k_stride=k.stride(),
                                                  v_stride=v.stride(),
                                                  state_stride=state_stride,
                                                  scale=scale,
                                                  dtype=q.dtype,
                                                  state_dtype=state_dtype,
                                                  g_dtype=g_dtype,
                                                  beta_dtype=beta_dtype,
                                                  use_g=g is not None,
                                                  use_beta=beta is not None,
                                                  use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                                                  output_final_state=output_final_state,
                                                  use_state_indices=state_indices is not None,
                                                  is_circular_buffer=cache_seqlens is not None,
                                                  num_warps=num_warps)

    kernel(q, k, v, o, g, beta, final_state, state_indices, cache_seqlens)

    if not output_final_state:
        final_state = None
    elif final_state is not None and state_dim == 4:
        final_state = final_state.squeeze(1)
    return o, final_state
