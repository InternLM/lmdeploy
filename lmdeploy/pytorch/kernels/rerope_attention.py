# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl

assert triton.__version__ >= '2.1.0'


# bugfix from https://gist.github.com/chu-tianxiang/4307937fd94b49c75b61a6967716bae9#file-rerope-py  # noqa: E501
@triton.jit
def _rerope_fwd_kernel(
    Q1,
    Q2,
    K1,
    K2,
    V,
    sm_scale,
    # L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """rerope attention triton kernel."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q1_block_ptr = tl.make_block_ptr(base=Q1 + q_offset,
                                     shape=(N_CTX, BLOCK_DMODEL),
                                     strides=(stride_qm, stride_qk),
                                     offsets=(start_m * BLOCK_M, 0),
                                     block_shape=(BLOCK_M, BLOCK_DMODEL),
                                     order=(1, 0))
    Q2_block_ptr = tl.make_block_ptr(base=Q2 + q_offset,
                                     shape=(N_CTX, BLOCK_DMODEL),
                                     strides=(stride_qm, stride_qk),
                                     offsets=(start_m * BLOCK_M, 0),
                                     block_shape=(BLOCK_M, BLOCK_DMODEL),
                                     order=(1, 0))
    K1_block_ptr = tl.make_block_ptr(base=K1 + kv_offset,
                                     shape=(BLOCK_DMODEL, N_CTX),
                                     strides=(stride_kk, stride_kn),
                                     offsets=(0, 0),
                                     block_shape=(BLOCK_DMODEL, BLOCK_N),
                                     order=(0, 1))
    K2_block_ptr = tl.make_block_ptr(base=K2 + kv_offset,
                                     shape=(BLOCK_DMODEL, N_CTX),
                                     strides=(stride_kk, stride_kn),
                                     offsets=(0, 0),
                                     block_shape=(BLOCK_DMODEL, BLOCK_N),
                                     order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
                                    shape=(N_CTX, BLOCK_DMODEL),
                                    strides=(stride_vk, stride_vn),
                                    offsets=(0, 0),
                                    block_shape=(BLOCK_N, BLOCK_DMODEL),
                                    order=(1, 0))
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q1 = tl.load(Q1_block_ptr, boundary_check=(0, 1))
    dtype = q1.dtype

    q1 = (q1 * qk_scale).to(dtype)
    q2 = tl.load(Q2_block_ptr, boundary_check=(0, 1))
    q2 = (q2 * qk_scale).to(dtype)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float('-inf'))
        if start_n <= start_m * BLOCK_M - WINDOW - BLOCK_N or start_n >= (
                start_m + 1) * BLOCK_M + WINDOW:
            k2 = tl.load(K2_block_ptr)
            v = tl.load(V_block_ptr)
            qk += tl.dot(q2, k2, out_dtype=tl.float32)
        elif start_n > (
                start_m + 1
        ) * BLOCK_M - WINDOW and start_n < start_m * BLOCK_M + WINDOW - BLOCK_N:  # noqa: E501
            k1 = tl.load(K1_block_ptr)
            v = tl.load(V_block_ptr)
            qk += tl.dot(q1, k1, out_dtype=tl.float32)
        else:
            k1 = tl.load(K1_block_ptr)
            k2 = tl.load(K2_block_ptr)
            v = tl.load(V_block_ptr)
            qk1 = tl.dot(q1, k1, out_dtype=tl.float32)
            qk2 = tl.dot(q2, k2, out_dtype=tl.float32)
            qk += tl.where(
                tl.abs(offs_m[:, None] - (start_n + offs_n[None, :])) < WINDOW,
                qk1, qk2)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v.to(tl.float32))
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K1_block_ptr = tl.advance(K1_block_ptr, (0, BLOCK_N))
        K2_block_ptr = tl.advance(K2_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    # debug softmax output
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
                                    shape=(N_CTX, BLOCK_DMODEL),
                                    strides=(stride_om, stride_on),
                                    offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    tl.store(O_block_ptr, acc.to(dtype), boundary_check=(0, 1))


def rerope_attention_fwd(q1,
                         q2,
                         k1,
                         k2,
                         v,
                         causal,
                         sm_scale,
                         window,
                         BLOCK_M=64):
    """rerope attention forward."""
    # shape constraints
    Lq, Lk, Lv = q1.shape[-1], k1.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q1)
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    grid = (triton.cdiv(q1.shape[2], BLOCK_M), q1.shape[0] * q1.shape[1], 1)
    # L = torch.empty((q1.shape[0] * q1.shape[1], q1.shape[2]),
    #                 device=q1.device,
    #                 dtype=torch.float32)
    _rerope_fwd_kernel[grid](
        q1,
        q2,
        k1,
        k2,
        v,
        sm_scale,
        #  L,
        o,
        q1.stride(0),
        q1.stride(1),
        q1.stride(2),
        q1.stride(3),
        k1.stride(0),
        k1.stride(1),
        k1.stride(2),
        k1.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q1.shape[0],
        q1.shape[1],
        q1.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        IS_CAUSAL=causal,
        WINDOW=window,
        num_warps=num_warps,
        num_stages=num_stages)
    return o


if __name__ == '__main__':

    def test_rerope():
        import torch.utils.benchmark as benchmark
        Z = 1
        H = 40
        N_CTX = 2176
        D_HEAD = 128
        WINDOW = 512
        sm_scale = 0.0883883

        def torch_attention(q1, q2, k1, k2, v, causal, sm_scale, window):
            # reference implementation
            M = torch.tril(torch.ones((N_CTX, N_CTX), device='cuda'))
            p1 = torch.matmul(q1, k1.transpose(2, 3)) * sm_scale
            p2 = torch.matmul(q2, k2.transpose(2, 3)) * sm_scale
            if causal:
                p1[:, :, M == 0] = float('-inf')
                p2[:, :, M == 0] = float('-inf')
            x = torch.arange(N_CTX, dtype=torch.int, device='cuda')
            M2 = ((x[:, None] - x[None, :]).abs() < window)[None, None, :]
            p = torch.where(M2, p1, p2)
            p = torch.softmax(p.float(), dim=-1).half()
            ref_out = torch.matmul(p, v)
            return ref_out

        def torch_attention2(query_states1, query_states2, key_states1,
                             key_states2, value_states, causal, sm_scale,
                             window):
            query_states1 = query_states1.squeeze(0).contiguous()
            query_states2 = query_states2.squeeze(0).contiguous()
            key_states1 = key_states1.squeeze(0).contiguous()
            key_states2 = key_states2.squeeze(0).contiguous()
            value_states = value_states.squeeze(0).contiguous()

            attn_weights1 = torch.matmul(
                query_states1, key_states1.transpose(1, 2)) * sm_scale
            attn_weights2 = torch.matmul(
                query_states2, key_states2.transpose(1, 2)) * sm_scale

            position_ids = torch.arange(
                query_states1.shape[1],
                device=query_states1.device).unsqueeze(0)
            rectified_mask = (position_ids[:, -N_CTX:, None] -
                              position_ids[:, None]).abs() < window
            attn_weights = torch.where(rectified_mask, attn_weights1,
                                       attn_weights2)

            if causal:
                tgt_len = attn_weights.shape[-1]
                dtype = attn_weights.dtype
                device = attn_weights.device
                mask = torch.full((tgt_len, tgt_len),
                                  torch.finfo(dtype).min,
                                  device=device)
                mask_cond = torch.arange(mask.size(-1), device=device)
                mask.masked_fill_(
                    mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(dtype)
                attn_weights = attn_weights + mask

            # upcast attention to fp32
            attn_weights = torch.nn.functional.softmax(attn_weights,
                                                       dim=-1,
                                                       dtype=torch.float32).to(
                                                           query_states1.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            return attn_output

        q1 = torch.empty((Z, H, N_CTX, D_HEAD),
                         dtype=torch.float16,
                         device='cuda').normal_(mean=0., std=0.5).contiguous()
        q2 = torch.empty((Z, H, N_CTX, D_HEAD),
                         dtype=torch.float16,
                         device='cuda').normal_(mean=0., std=0.5).contiguous()
        k1 = torch.empty((Z, H, N_CTX, D_HEAD),
                         dtype=torch.float16,
                         device='cuda').normal_(mean=0., std=0.5).contiguous()
        k2 = torch.empty((Z, H, N_CTX, D_HEAD),
                         dtype=torch.float16,
                         device='cuda').normal_(mean=0., std=0.5).contiguous()
        v = torch.empty((Z, H, N_CTX, D_HEAD),
                        dtype=torch.float16,
                        device='cuda').normal_(mean=0., std=0.5).contiguous()

        # q1 = torch.load('/workspace/GitProjects/lmdeploy/q1.pt',
        #                 map_location='cuda').contiguous()
        # q2 = torch.load('/workspace/GitProjects/lmdeploy/q2.pt',
        #                 map_location='cuda').contiguous()

        # k1 = torch.load('/workspace/GitProjects/lmdeploy/k1.pt',
        #                 map_location='cuda').contiguous()
        # k2 = torch.load('/workspace/GitProjects/lmdeploy/k2.pt',
        #                 map_location='cuda').contiguous()

        # v = torch.load('/workspace/GitProjects/lmdeploy/v.pt',
        #                map_location='cuda').contiguous()

        torch_output = torch_attention(q1, q2, k1, k2, v, True, sm_scale,
                                       WINDOW)
        torch_output2 = torch_attention2(q1, q2, k1, k2, v, True, sm_scale,
                                         WINDOW)
        assert torch.allclose(torch_output, torch_output2, atol=1e-2, rtol=0)
        for _ in range(100):
            triton_output = rerope_attention_fwd(q1, q2, k1, k2, v, True,
                                                 sm_scale, WINDOW)
            assert torch.allclose(
                torch_output, triton_output, atol=2e-2, rtol=0) is True

        def f(fn, q1, q2, k1, k2, v, sm_scale, window):
            fn(q1, q2, k1, k2, v, True, sm_scale, window)

        t0 = benchmark.Timer(stmt='f(fn, q1, q2, k1, k2, v, sm_scale, window)',
                             globals={
                                 'f': f,
                                 'fn': torch_attention2,
                                 'q1': q1,
                                 'q2': q2,
                                 'k1': k1,
                                 'k2': k2,
                                 'v': v,
                                 'sm_scale': sm_scale,
                                 'window': WINDOW
                             },
                             num_threads=torch.get_num_threads())
        print(t0.timeit(20))

        import time
        begin = time.time()
        LOOP = 100
        for _ in range(LOOP):
            rerope_attention_fwd(q1, q2, k1, k2, v, True, sm_scale, WINDOW)
        print(time.time() - begin)

    test_rerope()
