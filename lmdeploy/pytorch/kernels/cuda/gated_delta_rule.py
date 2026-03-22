# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import tilelang
import tilelang.language as T
import torch


@T.macro
def vec_store_state(State: T.Buffer, h_local: T.Buffer, state_id, state_update_id, hv_id, k_off, v_off, K: int, V: int,
                    k_per_thr: int, v_per_warp: int, state_vw: int) -> None:
    """Per-warp vectorized store of h_local[k_per_thr, v_per_warp] to global
    State.

    Each warp writes its own V-slice independently. Access pattern is K-strided (uncoalesced) since each lane owns a
    K-slice. Used as fallback for circular buffer when the coalesced path is not available.
    """
    for j in T.Unroll(k_per_thr):
        if (k_off + j) < K:
            for vg in T.Unroll(v_per_warp // state_vw):
                for i in T.Vectorized(state_vw):
                    idx = vg * state_vw + i
                    if v_off + idx < V:
                        State[state_id, state_update_id, hv_id, k_off + j, v_off + idx] = h_local[j, idx]


@T.macro
def vec_store_output(Out: T.Buffer, o_local: T.Buffer, b_id, seq_id, hv_id, v_off, V: int, v_per_warp: int,
                     data_vw: int) -> None:
    """Lane-0 vectorized store of warp-reduced output to global Out."""
    for vg in T.Unroll(v_per_warp // data_vw):
        for i in T.Vectorized(data_vw):
            idx = vg * data_vw + i
            v_idx = v_off + idx
            if v_idx < V:
                Out[b_id, seq_id, hv_id, v_idx] = o_local[idx]


@T.macro
def normalize_qk(k_local: T.Buffer, q_local: T.Buffer, k_per_thr: int) -> None:
    """In-kernel L2 normalization of q and k vectors across the warp."""
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
    """JIT-compiled recurrent gated delta rule forward kernel.

    Thread/memory layout:
      - Grid: (ceil(V / v_per_cta), B * HV)
      - Each CTA handles one (batch, head) pair and a V-tile of width v_per_cta
      - K dimension is partitioned across warp lanes: k_per_thr = ceil(K / 32)
      - V dimension is partitioned across warps: v_per_warp elements per warp
      - State tile h_local[k_per_thr, v_per_warp] lives in f32 registers

    Two execution paths (compile-time branched, zero overhead for unused path):
      - Circular buffer (is_circular_buffer=True): state written every timestep.
        v_per_warp is increased to make num_waves=1, so h_local stays in
        registers for the entire sequence. State writeback uses shared memory
        staging + T.Parallel for coalesced global writes.
      - Non-circular (is_circular_buffer=False): state written once after all
        timesteps. Multiple waves allowed to reduce grid size. Final state
        writeback also uses smem + T.Parallel.

    Note: All ``if`` conditions on function parameters (is_circular_buffer,
    use_coalesced_circular_write, use_g, etc.) are evaluated at JIT compile
    time by tilelang and produce zero runtime overhead — dead branches are
    eliminated from the generated CUDA code.
    """

    # --- Tiling parameters ---
    # Each warp owns a [k_per_thr, v_per_warp] tile of the KxV state matrix
    # in registers (h_local). K is partitioned across lanes (k_per_thr per
    # lane), V is partitioned across warps (v_per_warp per warp).
    #
    # "Waves" are sequential iterations over V within a single CTA. When
    # num_waves > 1, each warp processes multiple V-tiles sequentially, which
    # means h_local is overwritten between waves — the state cannot be kept
    # in registers across all timesteps.
    num_threads = num_warps * 32
    state_num_bits = T.DataType(state_dtype).bits
    data_num_bits = T.DataType(dtype).bits
    state_vec_width = 128 // state_num_bits
    data_vec_width = 128 // data_num_bits
    warp_size = 32
    k_per_thr = T.ceildiv(K, warp_size)
    v_per_warp = max(state_vec_width, data_vec_width, 8)
    if is_circular_buffer:
        # Circular buffer writes state to global memory at EVERY timestep
        # (not just after the final step). To avoid re-loading state from
        # smem between waves, we increase v_per_warp so that num_waves == 1.
        # This keeps h_local in f32 registers across all T timesteps,
        # eliminating wave-loop overhead and enabling the coalesced write
        # optimization below.
        #
        # Constraint: each lane holds k_per_thr * v_per_warp f32 registers.
        # max_v_per_warp = 128 / k_per_thr keeps register pressure under
        # 128 * sizeof(f32) = 512 bytes per lane.
        min_v_per_warp = v_per_warp
        max_v_per_warp = 128 // k_per_thr
        desired = T.ceildiv(V, num_warps)
        v_per_warp = T.ceildiv(min(desired, max_v_per_warp), min_v_per_warp) * min_v_per_warp
        v_per_warp = max(v_per_warp, min_v_per_warp)
        target_v_per_cta = V
    else:
        # Non-circular: state is written once after all timesteps.
        # Allow multiple waves to reduce grid size (fewer CTAs).
        target_v_per_cta = max(V, v_per_warp * num_warps * 2)
    # Vectorized width for each access type
    state_vw = min(state_vec_width, v_per_warp)
    data_vw = min(data_vec_width, v_per_warp)
    num_waves = T.ceildiv(target_v_per_cta, v_per_warp * num_warps)
    v_per_cta = v_per_warp * num_warps * num_waves
    # When num_waves == 1 in circular buffer mode, we can use a more
    # efficient coalesced state writeback path (see below).
    use_coalesced_circular_write = is_circular_buffer and num_waves == 1

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

            # --- Load initial state from global to shared memory ---
            # State layout: [N, NUM_STATE, HV, K, V]. We load a [K, v_per_cta]
            # tile via T.Parallel which distributes K*v_per_cta iterations
            # across all CTA threads for coalesced V-dimension reads.
            h_smem = T.alloc_shared([K, v_per_cta], state_dtype)
            T.annotate_layout({h_smem: tilelang.layout.make_swizzled_layout(h_smem)})
            for i, j in T.Parallel(K, v_per_cta):
                v_idx = v_start * v_per_cta + j
                if v_idx < V:
                    h_smem[i, j] = State[state_id, state_seq_id, hv_id, i, v_idx]
                else:
                    h_smem[i, j] = 0.0

            # --- Wave loop: iterate over V-tiles within this CTA ---
            # When num_waves == 1 (always for circular buffer), each warp
            # processes its V-slice once and h_local stays in registers for
            # the entire sequence. When num_waves > 1 (non-circular only),
            # h_local is reloaded from smem for each wave.
            for wave_id in range(num_waves):
                # Each warp's V-offset within the CTA tile
                v_warp_off = wave_id * num_warps * v_per_warp + warp_id * v_per_warp
                v_off = v_start * v_per_cta + v_warp_off

                # h_local[k_per_thr, v_per_warp]: per-lane state tile in f32
                # registers. Each lane holds k_per_thr rows of the K dimension
                # and v_per_warp columns of the V dimension.
                h_local = T.alloc_local([k_per_thr, v_per_warp], T.float32)
                if is_circular_buffer:
                    state_update_id = (state_seq_id + 1) % NUM_STATE
                for j in T.Unroll(k_per_thr):
                    k_idx = k_off + j
                    if k_idx < K:
                        for vg in T.Unroll(v_per_warp // state_vw):
                            for i in T.Vectorized(state_vw):
                                idx = vg * state_vw + i
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
                    for vg in T.Unroll(v_per_warp // data_vw):
                        for i in T.Vectorized(data_vw):
                            idx = vg * data_vw + i
                            v_idx = (v_off + idx) % V
                            v_local[idx] = Value[b_id, seq_id, hv_id, v_idx]

                    # --- Delta rule state update ---
                    # h[k,v] = g_exp * h[k,v] + k[k] * (v[v] - beta * k^T @ h[:,v])
                    # The inner loop processes one V-element at a time. For each v:
                    #   1. Decay: h[k,v] *= g_exp  (gating)
                    #   2. Dot:   hk = sum_k(h[k,v] * k[k])  (warp-reduced)
                    #   3. Delta: v_delta = (v[v] - hk) * beta
                    #   4. Update: h[k,v] += k[k] * v_delta
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

                    # --- Per-warp uncoalesced circular state store (fallback) ---
                    # Each warp writes its own [k_per_thr, v_per_warp] tile
                    # directly to global State. Access is K-strided (each lane
                    # writes different K rows), which is uncoalesced.
                    # Only used when use_coalesced_circular_write is False.
                    if output_final_state and state_id >= 0:
                        if is_circular_buffer and not use_coalesced_circular_write:
                            vec_store_state(State, h_local, state_id, state_update_id, hv_id, k_off, v_off, K, V,
                                            k_per_thr, v_per_warp, state_vw)
                            state_update_id = (state_update_id + 1) % NUM_STATE

                    # compute output: o[v] = sum_k(q[k] * h[k,v])
                    o_local = T.alloc_local([v_per_warp], dtype)
                    for i in T.Unroll(v_per_warp):
                        o = T.alloc_var(T.float32)
                        o = 0.0
                        for j in T.Unroll(k_per_thr):
                            o += q_local[j] * h_local[j, i]
                        o = T.warp_reduce_sum(o)
                        o_local[i] = o

                    # Only lane 0 has the correct warp-reduced output
                    if lane_id == 0 and state_id >= 0:
                        vec_store_output(Out, o_local, b_id, seq_id, hv_id, v_off, V, v_per_warp, data_vw)

                    # --- Coalesced circular state writeback (optimized path) ---
                    # Instead of each warp writing its own K-strided slice
                    # (uncoalesced), all warps cooperate:
                    #   1. Each warp writes h_local to its slice of h_smem
                    #   2. T.Parallel distributes the full [K, v_per_cta] write
                    #      across all CTA threads, giving consecutive threads
                    #      consecutive V addresses → coalesced global writes
                    # This yields ~2.5x speedup over the per-warp path because
                    # state writes dominate memory traffic (~78% of total).
                    if use_coalesced_circular_write and output_final_state and state_id >= 0:
                        for j in T.Unroll(k_per_thr):
                            k_idx = k_off + j
                            if k_idx < K:
                                for vg in T.Unroll(v_per_warp // state_vw):
                                    for i in T.Vectorized(state_vw):
                                        idx = vg * state_vw + i
                                        h_smem[k_idx, v_warp_off + idx] = h_local[j, idx]
                        for i, j in T.Parallel(K, v_per_cta):
                            v_idx = v_start * v_per_cta + j
                            if v_idx < V:
                                State[state_id, state_update_id, hv_id, i, v_idx] = h_smem[i, j]
                        state_update_id = (state_update_id + 1) % NUM_STATE

                # --- Non-circular: write h_local to smem after all timesteps ---
                # Each warp copies its register tile back to the shared memory
                # staging buffer. The actual global write happens once after
                # the wave loop exits (outside all waves).
                if output_final_state and state_id >= 0 and not is_circular_buffer:
                    for j in T.Unroll(k_per_thr):
                        k_idx = k_off + j
                        if k_idx < K:
                            for vg in T.Unroll(v_per_warp // state_vw):
                                for i in T.Vectorized(state_vw):
                                    idx = vg * state_vw + i
                                    h_smem[k_idx, v_warp_off + idx] = h_local[j, idx]

            # --- Non-circular: coalesced state writeback from smem to global ---
            # After all waves have written their tiles to h_smem, do one
            # cooperative coalesced write of the full [K, v_per_cta] tile.
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
        use_qk_l2norm_in_kernel: whether to apply l2 normalization on q and k
            in the kernel
        state_indices: [B], optional, the indices to update in the recurrent
            state, required
        cache_seqlens: [B], optional, the cached sequence lengths for each
            batch element
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
    assert K % 32 == 0, f'K ({K}) must be a multiple of 32 (warp size)'
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
        assert initial_state is not None, \
            'initial_state is required when state_indices is provided'
        assert state_indices.shape == (q.shape[0], )
    if cache_seqlens is not None:
        assert cache_seqlens.is_contiguous(), \
            'cache_seqlens must be contiguous'
        assert cache_seqlens.shape == (q.shape[0], ), \
            'cache_seqlens must have shape (B,)'
        assert cache_seqlens.dtype == torch.int32, \
            'cache_seqlens must have dtype torch.int32'
        assert cache_seqlens.device == q.device, \
            'cache_seqlens must be on the same device as q'

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
    kernel = fused_recurrent_gated_delta_rule_fwd(
        seqlen,
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
        num_warps=num_warps,
    )

    kernel(q, k, v, o, g, beta, final_state, state_indices, cache_seqlens)

    if not output_final_state:
        final_state = None
    elif final_state is not None and state_dim == 4:
        final_state = final_state.squeeze(1)
    return o, final_state
