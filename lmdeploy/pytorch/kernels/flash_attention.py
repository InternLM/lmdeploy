"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import pytest
import torch

import triton
from triton import Config, autotune, cdiv, heuristics, jit
import triton.language as tl
from torch.nn import functional as F

@triton.autotune(
    configs=[
        Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_stages=4, num_warps=1),
        Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=1),
        Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    ],
    key=['BATCH_SIZE','nheads', 'seqlen_q', 'seqlen_kv'],
)
@triton.heuristics(  # order should be the same as in function args, otherwise expect strange bugs
    {
        "EVEN_M": lambda args: True if args["seqlen_q"] % args["BLOCK_M"] == 0 else False,
        "EVEN_N": lambda args: True if args["seqlen_kv"] % args["BLOCK_N"] == 0 else False,
    }
)

@triton.jit
def _fwd_kernel(
    Q, K, V, O, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    nheads, seqlen_q , seqlen_kv , headdim,
    BATCH_SIZE, IS_CAUSUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, EVEN_M: tl.constexpr,  EVEN_N: tl.constexpr, 
):
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    # initialize offsets
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V, O
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[ None,:] * stride_kn + offs_d[:,None] * stride_kk)
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    
    # initialize pointer to m and l
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32) #
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    o_mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
    
    if IS_CAUSUAL:
        q = tl.load(q_ptrs)
        end = tl.minimum((start_m + 1) * BLOCK_M, seqlen_kv)
    else:
        if EVEN_M:
            q = tl.load(q_ptrs)
        else:
            
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0
            )
            
            
        end = seqlen_kv if EVEN_N else seqlen_kv // BLOCK_N * BLOCK_N
    
    for start_n in range(0, end , BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        if IS_CAUSUAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        
        acc += tl.dot(p, v) 
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    if not EVEN_N and not IS_CAUSUAL:
        k_mask = ((end + offs_n)[ None,:] < seqlen_kv) & (offs_d[:,None] < headdim)
        v_mask = ((end + offs_n)[:, None] < seqlen_kv) & (offs_d[None, :] < headdim)
        k = tl.load(k_ptrs, mask = k_mask, other=0.0)
        v = tl.load(v_ptrs, mask = v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where((end + offs_n)[ None,:] < seqlen_kv, qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        
        acc += tl.dot(p, v) 
    

    off_o = off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    out_ptrs = O + off_o
   
    tl.store(out_ptrs, acc.to(Q.dtype.element_ty), o_mask)
    



class _attention(torch.autograd.Function):

    def forward(ctx, q, k, v,  is_causual):
        # only support for Ampere now
        # capability = torch.cuda.get_device_capability()
        # if capability[0] < 8:
        #     raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
    
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        bs = q.shape[0]
        num_heads = q.shape[2]
        grid = lambda META: (cdiv(seqlen_q, META['BLOCK_M']), bs*num_heads)
        
        _fwd_kernel[grid](
            q, k, v, o, 1/(Lk ** 0.5),
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            num_heads, seqlen_q, seqlen_kv, q.shape[-1],
            bs,is_causual, 
            BLOCK_DMODEL=Lk
        )

        return o

    

attention = _attention.apply



BATCH, N_HEADS, D_HEAD = 1, 32, 128
GENERATION = True
IS_CAUSAL = False
# vary seq length for fixed head and batch=4
configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(4,12)],
    line_arg='provider',
    line_vals=['triton','pytorch','xformers','flash'] ,
    line_names=['Triton','Pytorch','Xformers','FlashAttention2'] ,
    styles=[  ('red', '-'),('brown', '-'), ('yellow', '-'),('green', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{GENERATION}-{IS_CAUSAL}',
    args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': torch.float16, }
) ]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, provider, dtype=torch.float16, device="cuda"):
    warmup = 50
    rep = 100
    
    if provider == "triton":
        
        Q_CTX = 1 if GENERATION else N_CTX
        
        q = torch.randn((BATCH, Q_CTX,H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn((BATCH, N_CTX,H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn((BATCH, N_CTX,H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, False)
        
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        print(_fwd_kernel.best_config)
        return ms
    
    if provider == "flash":
        Q_CTX = 1 if GENERATION else N_CTX
        q = torch.randn((BATCH,Q_CTX, H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn((BATCH,N_CTX, H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn((BATCH,N_CTX, H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
    
        sm_scale = 1.3

        from flash_attn.flash_attn_interface import flash_attn_func
        fn = lambda: flash_attn_func(q, k, v,0.0, sm_scale,False)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        
        return ms

    
    if provider == "pytorch":
        Q_CTX = 1 if GENERATION else N_CTX
        q = torch.randn((BATCH, H, Q_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        
        fn = lambda: F.scaled_dot_product_attention (q, k, v, is_causal=IS_CAUSAL)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    if provider == "xformers":
        Q_CTX = 1 if GENERATION else N_CTX
        q = torch.randn((BATCH,Q_CTX, H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn((BATCH,N_CTX, H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn((BATCH,N_CTX, H,  D_HEAD), dtype=dtype, device="cuda", requires_grad=False)
        from xformers.ops import memory_efficient_attention
        from xformers.ops import LowerTriangularMask
        if IS_CAUSAL:
            fn = lambda: memory_efficient_attention (q, k, v, LowerTriangularMask())
        else:
            fn = lambda: memory_efficient_attention (q, k, v)
        
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

from pathlib import Path
save_path = Path('./fused_attention')
save_path.mkdir(exist_ok=True)
bench_flash_attention.run(save_path=save_path, print_data=True)