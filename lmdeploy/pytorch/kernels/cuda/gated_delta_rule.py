# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import tilelang
import tilelang.language as T
import torch


@T.macro
def load_state_tile_from_smem(h_smem: T.Buffer, h_local: T.Buffer, k_off, v_warp_off, K: int, k_per_thr: int,
                              v_per_warp: int, state_vw: int) -> None:
    """Load one warp's logical [K, V] state tile from shared memory."""
    for j in T.Unroll(k_per_thr):
        k_idx = k_off + j
        if k_idx < K:
            for vg in T.Unroll(v_per_warp // state_vw):
                for i in T.Vectorized(state_vw):
                    idx = vg * state_vw + i
                    h_local[j, idx] = h_smem[k_idx, v_warp_off + idx]


@T.macro
def load_transposed_state_tile(State: T.Buffer, h_local: T.Buffer, state_id, state_seq_id, hv_id, k_off, v_off,
                               K: int, V: int, k_per_thr: int, v_per_warp: int) -> None:
    """Load one warp's state tile directly from transposed State [V, K]."""
    for i in T.Unroll(v_per_warp):
        v_idx = v_off + i
        if v_idx < V:
            for j in T.Vectorized(k_per_thr):
                h_local[j, i] = State[state_id, state_seq_id, hv_id, v_idx, k_off + j]
        else:
            for j in T.Vectorized(k_per_thr):
                h_local[j, i] = 0.0


@T.macro
def stage_state_tile_to_smem(h_smem: T.Buffer, h_local: T.Buffer, k_off, v_warp_off, K: int, k_per_thr: int,
                             v_per_warp: int, state_vw: int) -> None:
    """Stage one warp's logical [K, V] state tile to shared memory."""
    for j in T.Unroll(k_per_thr):
        k_idx = k_off + j
        if k_idx < K:
            for vg in T.Unroll(v_per_warp // state_vw):
                for i in T.Vectorized(state_vw):
                    idx = vg * state_vw + i
                    h_smem[k_idx, v_warp_off + idx] = h_local[j, idx]


@T.macro
def store_default_state_tile_direct(State: T.Buffer, h_local: T.Buffer, state_id, state_update_id, hv_id, k_off,
                                    v_off, K: int, V: int, k_per_thr: int, v_per_warp: int,
                                    state_vw: int) -> None:
    """Per-warp store for the default State layout [K, V]."""
    for j in T.Unroll(k_per_thr):
        if (k_off + j) < K:
            for vg in T.Unroll(v_per_warp // state_vw):
                for i in T.Vectorized(state_vw):
                    idx = vg * state_vw + i
                    if v_off + idx < V:
                        State[state_id, state_update_id, hv_id, k_off + j, v_off + idx] = h_local[j, idx]


@T.macro
def store_transposed_state_tile(State: T.Buffer, h_local: T.Buffer, state_id, state_update_id, hv_id, k_off, v_off,
                                K: int, V: int, k_per_thr: int, v_per_warp: int) -> None:
    """Per-warp store for transposed State layout [V, K]."""
    for i in T.Unroll(v_per_warp):
        v_idx = v_off + i
        if v_idx < V:
            for j in T.Vectorized(k_per_thr):
                State[state_id, state_update_id, hv_id, v_idx, k_off + j] = h_local[j, i]


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
def zero_output_tile(Out: T.Buffer, b_id, hv_id, v_start, lane_id, warp_id, V: int, SEQLEN: int, num_waves: int,
                     num_warps: int, v_per_warp: int, v_per_cta: int, data_vw: int) -> None:
    """Zero output for requests that do not own a valid recurrent state."""
    if lane_id == 0:
        for wave_id in range(num_waves):
            v_warp_off = wave_id * num_warps * v_per_warp + warp_id * v_per_warp
            v_off = v_start * v_per_cta + v_warp_off
            for seq_id in range(SEQLEN):
                for vg in T.Unroll(v_per_warp // data_vw):
                    for i in T.Vectorized(data_vw):
                        idx = vg * data_vw + i
                        v_idx = v_off + idx
                        if v_idx < V:
                            Out[b_id, seq_id, hv_id, v_idx] = 0.0


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
    k_norm = T.sqrt(k_sum + 1e-6)
    q_norm = T.sqrt(q_sum + 1e-6)
    for i in T.Unroll(k_per_thr):
        k_local[i] = k_local[i] / k_norm
        q_local[i] = q_local[i] / q_norm


@T.macro
def precompute_shared_token_inputs(Query: T.Buffer, Key: T.Buffer, G: T.Buffer, Beta: T.Buffer, q_smem: T.Buffer,
                                   k_smem: T.Buffer, g_exp_smem: T.Buffer, beta_smem: T.Buffer, b_id, h_id, hv_id,
                                   warp_id, k_off, K: int, SEQLEN: int, k_per_thr: int, scale,
                                   use_qk_l2norm_in_kernel: bool, use_g: bool, use_beta: bool) -> None:
    """Stage q/k/g/beta values that are shared by all V waves in a CTA."""
    if warp_id == 0:
        for seq_pre in range(SEQLEN):
            q_pre = T.alloc_local([k_per_thr], T.float32)
            k_pre = T.alloc_local([k_per_thr], T.float32)
            for i in T.Vectorized(k_per_thr):
                k_idx = (k_off + i) % K
                q_pre[i] = Query[b_id, seq_pre, h_id, k_idx]
            for i in T.Vectorized(k_per_thr):
                k_idx = (k_off + i) % K
                k_pre[i] = Key[b_id, seq_pre, h_id, k_idx]
            if use_qk_l2norm_in_kernel:
                normalize_qk(k_pre, q_pre, k_per_thr)
            for i in T.Vectorized(k_per_thr):
                k_idx = (k_off + i) % K
                q_smem[seq_pre, k_idx] = q_pre[i] * scale
                k_smem[seq_pre, k_idx] = k_pre[i]
    T.sync_threads()

    for s in T.Parallel(SEQLEN):
        if use_g:
            g = T.cast(G[b_id, s, hv_id], T.float32)
            g_exp_smem[s] = T.exp(g)
        else:
            g_exp_smem[s] = 1.0
        if use_beta:
            beta_smem[s] = T.cast(Beta[b_id, s, hv_id], T.float32)
        else:
            beta_smem[s] = 1.0
    T.sync_threads()


@T.macro
def load_shared_qk(q_smem: T.Buffer, k_smem: T.Buffer, q_local: T.Buffer, k_local: T.Buffer, seq_id, k_off, K: int,
                   k_per_thr: int) -> None:
    """Load precomputed q/k from CTA shared memory."""
    for i in T.Vectorized(k_per_thr):
        k_idx = (k_off + i) % K
        q_local[i] = q_smem[seq_id, k_idx]
    for i in T.Vectorized(k_per_thr):
        k_idx = (k_off + i) % K
        k_local[i] = k_smem[seq_id, k_idx]


@T.macro
def load_global_qk(Query: T.Buffer, Key: T.Buffer, q_local: T.Buffer, k_local: T.Buffer, b_id, seq_id, h_id, k_off,
                   K: int, k_per_thr: int, scale, use_qk_l2norm_in_kernel: bool) -> None:
    """Load q/k from global memory, optionally normalize, and scale q."""
    for i in T.Vectorized(k_per_thr):
        k_idx = (k_off + i) % K
        q_local[i] = Query[b_id, seq_id, h_id, k_idx]
    for i in T.Vectorized(k_per_thr):
        k_idx = (k_off + i) % K
        k_local[i] = Key[b_id, seq_id, h_id, k_idx]

    if use_qk_l2norm_in_kernel:
        normalize_qk(k_local, q_local, k_per_thr)

    for i in T.Vectorized(k_per_thr):
        q_local[i] = q_local[i] * scale


@T.macro
def load_value_tile(Value: T.Buffer, v_local: T.Buffer, b_id, seq_id, hv_id, v_off, V: int, v_per_warp: int,
                    data_vw: int) -> None:
    """Load one warp's V tile for the current token."""
    for vg in T.Unroll(v_per_warp // data_vw):
        for i in T.Vectorized(data_vw):
            idx = vg * data_vw + i
            v_idx = (v_off + idx) % V
            v_local[idx] = Value[b_id, seq_id, hv_id, v_idx]


@T.macro
def update_recurrent_state(h_local: T.Buffer, k_local: T.Buffer, v_local: T.Buffer, g_exp, beta, k_per_thr: int,
                           v_per_warp: int) -> None:
    """Apply one gated delta-rule token update to a warp-local state tile."""
    for i in T.Unroll(v_per_warp):
        hk = T.alloc_var(T.float32)
        hk = 0
        for j in T.Unroll(k_per_thr):
            h_local[j, i] = h_local[j, i] * g_exp
            hk += h_local[j, i] * k_local[j]
        hk = T.warp_reduce_sum(hk)
        v_delta = (v_local[i] - hk) * beta
        for j in T.Unroll(k_per_thr):
            h_local[j, i] = h_local[j, i] + k_local[j] * v_delta


@T.macro
def compute_output_tile(q_local: T.Buffer, h_local: T.Buffer, o_local: T.Buffer, k_per_thr: int,
                        v_per_warp: int) -> None:
    """Compute o[v] = sum_k(q[k] * h[k,v]) for one warp-local state tile."""
    for i in T.Unroll(v_per_warp):
        o = T.alloc_var(T.float32)
        o = 0.0
        for j in T.Unroll(k_per_thr):
            o += q_local[j] * h_local[j, i]
        o = T.warp_reduce_sum(o)
        o_local[i] = o


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
                                         transpose_state_layout: bool = False,
                                         num_warps: int = 1):
    """Build the layout-specific recurrent GDR TileLang kernel.

    Common compile-time metadata is computed once here. The only structural branch is the returned T.prim_func body,
    because default and transposed state layouts use different state IO strategies.
    """

    num_threads = num_warps * 32
    state_num_bits = T.DataType(state_dtype).bits
    data_num_bits = T.DataType(dtype).bits
    state_vec_width = 128 // state_num_bits
    data_vec_width = 128 // data_num_bits
    warp_size = 32
    k_per_thr = T.ceildiv(K, warp_size)
    v_per_warp = max(state_vec_width, data_vec_width, 8)
    if is_circular_buffer:
        # Circular buffer writes state to global memory at every timestep.
        # The transposed path can use smaller V tiles because its state IO is
        # already coalesced and benefits more from lower register pressure.
        min_v_per_warp = v_per_warp
        max_v_per_warp = 128 // k_per_thr
        if transpose_state_layout:
            max_v_per_warp = min(max_v_per_warp, 8 if state_num_bits <= 16 else 16)
        desired = T.ceildiv(V, num_warps)
        v_per_warp = T.ceildiv(min(desired, max_v_per_warp), min_v_per_warp) * min_v_per_warp
        v_per_warp = max(v_per_warp, min_v_per_warp)
        target_v_per_cta = V
    else:
        target_v_per_cta = max(V, v_per_warp * num_warps * 2)

    state_vw = min(state_vec_width, v_per_warp)
    data_vw = min(data_vec_width, v_per_warp)
    num_waves = T.ceildiv(target_v_per_cta, v_per_warp * num_warps)
    v_per_cta = v_per_warp * num_warps * num_waves
    use_coalesced_circular_write = is_circular_buffer and num_waves == 1
    use_shared_token_inputs = SEQLEN <= 8
    write_circular_state = output_final_state and is_circular_buffer
    write_direct_circular_state = write_circular_state and not use_coalesced_circular_write
    write_coalesced_circular_state = write_circular_state and use_coalesced_circular_write
    write_final_state = output_final_state and not is_circular_buffer

    B = T.dynamic('B')
    N = B if not use_state_indices else T.dynamic('N')

    if g_dtype is None:
        g_dtype = dtype
    if beta_dtype is None:
        beta_dtype = dtype

    @T.prim_func
    def fused_recurrent_gated_delta_rule_default_main(
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

            if use_state_indices:
                state_id = T.cast(StateIndices[b_id], T.int64)
            else:
                state_id = b_id

            if is_circular_buffer:
                state_seq_id = CacheSeqlens[b_id] % NUM_STATE
                state_update_id = T.alloc_var(T.int32)
                state_update_id = (state_seq_id + 1) % NUM_STATE
            else:
                state_seq_id = 0
                state_update_id = 0

            h_smem = T.alloc_shared([K, v_per_cta], state_dtype)
            T.annotate_layout({h_smem: tilelang.layout.make_swizzled_layout(h_smem)})
            if state_id >= 0 and state_id < N:
                for i, j in T.Parallel(K, v_per_cta):
                    v_idx = v_start * v_per_cta + j
                    if v_idx < V:
                        h_smem[i, j] = State[state_id, state_seq_id, hv_id, i, v_idx]
                    else:
                        h_smem[i, j] = 0.0

                for wave_id in range(num_waves):
                    v_warp_off = wave_id * num_warps * v_per_warp + warp_id * v_per_warp
                    v_off = v_start * v_per_cta + v_warp_off

                    h_local = T.alloc_local([k_per_thr, v_per_warp], T.float32)
                    if is_circular_buffer:
                        state_update_id = (state_seq_id + 1) % NUM_STATE

                    load_state_tile_from_smem(h_smem, h_local, k_off, v_warp_off, K, k_per_thr, v_per_warp, state_vw)

                    for seq_id in range(SEQLEN):
                        q_local = T.alloc_local([k_per_thr], T.float32)
                        k_local = T.alloc_local([k_per_thr], T.float32)
                        load_global_qk(Query, Key, q_local, k_local, b_id, seq_id, h_id, k_off, K, k_per_thr, scale,
                                       use_qk_l2norm_in_kernel)

                        g_exp = T.alloc_var(T.float32)
                        beta = T.alloc_var(T.float32)
                        if use_g:
                            if lane_id == 0:
                                g = T.cast(G[b_id, seq_id, hv_id], T.float32)
                                g_exp = T.exp(g)
                            else:
                                g_exp = 0.0
                            g_exp = T.shfl_sync(0xFFFFFFFF, g_exp, 0)
                        else:
                            g_exp = 1.0
                        if use_beta:
                            if lane_id == 0:
                                beta = T.cast(Beta[b_id, seq_id, hv_id], T.float32)
                            else:
                                beta = 0.0
                            beta = T.shfl_sync(0xFFFFFFFF, beta, 0)
                        else:
                            beta = 1.0

                        v_local = T.alloc_local([v_per_warp], dtype)
                        load_value_tile(Value, v_local, b_id, seq_id, hv_id, v_off, V, v_per_warp, data_vw)
                        update_recurrent_state(h_local, k_local, v_local, g_exp, beta, k_per_thr, v_per_warp)

                        if write_direct_circular_state:
                            store_default_state_tile_direct(State, h_local, state_id, state_update_id, hv_id, k_off,
                                                            v_off, K, V, k_per_thr, v_per_warp, state_vw)
                            state_update_id = (state_update_id + 1) % NUM_STATE

                        o_local = T.alloc_local([v_per_warp], dtype)
                        compute_output_tile(q_local, h_local, o_local, k_per_thr, v_per_warp)

                        if lane_id == 0:
                            vec_store_output(Out, o_local, b_id, seq_id, hv_id, v_off, V, v_per_warp, data_vw)

                        if write_coalesced_circular_state:
                            stage_state_tile_to_smem(h_smem, h_local, k_off, v_warp_off, K, k_per_thr, v_per_warp,
                                                     state_vw)
                            for i, j in T.Parallel(K, v_per_cta):
                                v_idx = v_start * v_per_cta + j
                                if v_idx < V:
                                    State[state_id, state_update_id, hv_id, i, v_idx] = h_smem[i, j]
                            state_update_id = (state_update_id + 1) % NUM_STATE

                        if write_final_state:
                            stage_state_tile_to_smem(h_smem, h_local, k_off, v_warp_off, K, k_per_thr, v_per_warp,
                                                     state_vw)

                    if write_final_state:
                        for i, j in T.Parallel(K, v_per_cta):
                            v_idx = v_start * v_per_cta + j
                            if v_idx < V:
                                State[state_id, state_update_id, hv_id, i, v_idx] = h_smem[i, j]
            else:
                zero_output_tile(Out, b_id, hv_id, v_start, lane_id, warp_id, V, SEQLEN, num_waves, num_warps,
                                 v_per_warp, v_per_cta, data_vw)

    @T.prim_func
    def fused_recurrent_gated_delta_rule_transposed_main(
        Query: T.StridedTensor([B, SEQLEN, H, K], dtype=dtype, strides=q_stride),
        Key: T.StridedTensor([B, SEQLEN, H, K], dtype=dtype, strides=k_stride),
        Value: T.StridedTensor([B, SEQLEN, HV, V], dtype=dtype, strides=v_stride),
        Out: T.Tensor([B, SEQLEN, HV, V], dtype=dtype),
        G: T.Tensor([B, SEQLEN, HV], dtype=g_dtype),
        Beta: T.Tensor([B, SEQLEN, HV], dtype=beta_dtype),
        State: T.StridedTensor([N, NUM_STATE, HV, V, K], dtype=state_dtype, strides=state_stride),
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

            if use_state_indices:
                state_id = T.cast(StateIndices[b_id], T.int64)
            else:
                state_id = b_id

            if is_circular_buffer:
                state_seq_id = CacheSeqlens[b_id] % NUM_STATE
                state_update_id = T.alloc_var(T.int32)
                state_update_id = (state_seq_id + 1) % NUM_STATE
            else:
                state_seq_id = 0
                state_update_id = 0

            if use_shared_token_inputs:
                q_smem = T.alloc_shared([SEQLEN, K], T.float32)
                k_smem = T.alloc_shared([SEQLEN, K], T.float32)
                g_exp_smem = T.alloc_shared([SEQLEN], T.float32)
                beta_smem = T.alloc_shared([SEQLEN], T.float32)

            if state_id >= 0 and state_id < N:
                if use_shared_token_inputs:
                    precompute_shared_token_inputs(Query, Key, G, Beta, q_smem, k_smem, g_exp_smem, beta_smem, b_id,
                                                   h_id, hv_id, warp_id, k_off, K, SEQLEN, k_per_thr, scale,
                                                   use_qk_l2norm_in_kernel, use_g, use_beta)

                for wave_id in range(num_waves):
                    v_warp_off = wave_id * num_warps * v_per_warp + warp_id * v_per_warp
                    v_off = v_start * v_per_cta + v_warp_off

                    h_local = T.alloc_local([k_per_thr, v_per_warp], T.float32)
                    if is_circular_buffer:
                        state_update_id = (state_seq_id + 1) % NUM_STATE

                    load_transposed_state_tile(State, h_local, state_id, state_seq_id, hv_id, k_off, v_off, K, V,
                                               k_per_thr, v_per_warp)

                    for seq_id in range(SEQLEN):
                        q_local = T.alloc_local([k_per_thr], T.float32)
                        k_local = T.alloc_local([k_per_thr], T.float32)
                        if use_shared_token_inputs:
                            load_shared_qk(q_smem, k_smem, q_local, k_local, seq_id, k_off, K, k_per_thr)
                        else:
                            load_global_qk(Query, Key, q_local, k_local, b_id, seq_id, h_id, k_off, K, k_per_thr,
                                           scale, use_qk_l2norm_in_kernel)

                        g_exp = T.alloc_var(T.float32)
                        beta = T.alloc_var(T.float32)
                        if use_shared_token_inputs:
                            g_exp = g_exp_smem[seq_id]
                            beta = beta_smem[seq_id]
                        else:
                            if use_g:
                                if lane_id == 0:
                                    g = T.cast(G[b_id, seq_id, hv_id], T.float32)
                                    g_exp = T.exp(g)
                                else:
                                    g_exp = 0.0
                                g_exp = T.shfl_sync(0xFFFFFFFF, g_exp, 0)
                            else:
                                g_exp = 1.0
                            if use_beta:
                                if lane_id == 0:
                                    beta = T.cast(Beta[b_id, seq_id, hv_id], T.float32)
                                else:
                                    beta = 0.0
                                beta = T.shfl_sync(0xFFFFFFFF, beta, 0)
                            else:
                                beta = 1.0

                        v_local = T.alloc_local([v_per_warp], dtype)
                        load_value_tile(Value, v_local, b_id, seq_id, hv_id, v_off, V, v_per_warp, data_vw)
                        update_recurrent_state(h_local, k_local, v_local, g_exp, beta, k_per_thr, v_per_warp)

                        if write_circular_state:
                            store_transposed_state_tile(State, h_local, state_id, state_update_id, hv_id, k_off, v_off,
                                                        K, V, k_per_thr, v_per_warp)
                            state_update_id = (state_update_id + 1) % NUM_STATE

                        o_local = T.alloc_local([v_per_warp], dtype)
                        compute_output_tile(q_local, h_local, o_local, k_per_thr, v_per_warp)

                        if lane_id == 0:
                            vec_store_output(Out, o_local, b_id, seq_id, hv_id, v_off, V, v_per_warp, data_vw)

                    if write_final_state:
                        store_transposed_state_tile(State, h_local, state_id, state_update_id, hv_id, k_off, v_off, K,
                                                    V, k_per_thr, v_per_warp)
            else:
                zero_output_tile(Out, b_id, hv_id, v_start, lane_id, warp_id, V, SEQLEN, num_waves, num_warps,
                                 v_per_warp, v_per_cta, data_vw)

    if transpose_state_layout:
        return fused_recurrent_gated_delta_rule_transposed_main
    return fused_recurrent_gated_delta_rule_default_main


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
    transpose_state_layout: bool = False,
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
            [N, HV, K, V] or [N, NUM_STATE, HV, K, V]. If
            ``transpose_state_layout`` is True, the last two dimensions are
            [V, K]. If ``state_indices`` is not provided, N = B. When using
            circular buffers (i.e. ``cache_seqlens`` is not None),
            ``NUM_STATE`` specifies the number of state slots per sequence
            (e.g. buffer size).
        use_qk_l2norm_in_kernel: whether to apply l2 normalization on q and k
            in the kernel
        state_indices: [B], optional, the indices to update in the recurrent
            state, required
        cache_seqlens: [B], optional, the cached sequence lengths for each
            batch element
        transpose_state_layout: whether recurrent state is stored as [V, K]
            instead of [K, V]
    Returns:
        o: [B, T, HV, V]
        final_state: Recurrent state if ``output_final_state`` is True,
            otherwise None. The returned state has the same layout and rank as
            the input ``initial_state``.
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

    assert initial_state is not None, 'initial_state is required'
    o = torch.empty_like(v)
    final_state = initial_state
    state_dtype = q.dtype
    if final_state is not None:
        state_dim = final_state.dim()
        expected_state_shape = (V, K) if transpose_state_layout else (K, V)
        assert final_state.shape[-2:] == expected_state_shape, (
            f'initial_state last two dims must be {expected_state_shape} when '
            f'transpose_state_layout={transpose_state_layout}, got {tuple(final_state.shape[-2:])}')
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

    num_warps = 2 if transpose_state_layout and cache_seqlens is not None else 4
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
        transpose_state_layout=transpose_state_layout,
        num_warps=num_warps,
    )

    kernel(q, k, v, o, g, beta, final_state, state_indices, cache_seqlens)

    if not output_final_state:
        final_state = None
    elif final_state is not None and state_dim == 4:
        final_state = final_state.squeeze(1)
    return o, final_state
