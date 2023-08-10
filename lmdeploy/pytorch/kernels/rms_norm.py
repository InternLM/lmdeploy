import torch
from torch import nn
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import LlamaRMSNorm

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)

def rms_norm( x, weight, variance_epsilon):
    with torch.cuda.device(x.device):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        rms_norm_fwd_fused[(M,)](x_arg, y, weight, 
                                x_arg.stride(0), N, variance_epsilon,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y





BATCH = 32
HEAD = 32
DIM = 4096

configs = [triton.testing.Benchmark(
    x_names=['n_ctx'],
    x_vals=[2 ** i for i in range(4, 12)],
    line_arg='provider',
    line_vals=['triton'] ,
    line_names=[ 'Triton'] ,
    styles=[('red', '-')],
    ylabel='ms',
    plot_name=f'RMSNorm',
    args={'bs': BATCH,  'd': DIM, 'dtype': torch.float16})
]


@triton.testing.perf_report(configs)
def bench_rms_norm(bs,n_ctx,  d , provider, dtype=torch.float16, device="cuda"):
    
    warmup = 25
    rep = 100
    
    
    x = torch.randn((bs, n_ctx,  d), dtype=dtype, device="cuda", requires_grad=False)
    
    if provider == "triton":
        weight = torch.randn(d,dtype=dtype, device=device)
        fn = lambda: rms_norm(x, weight, 1e06)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms


from pathlib import Path
save_path = Path('./rms_norm')
save_path.mkdir(exist_ok=True)
bench_rms_norm.run(save_path=save_path, print_data=True)
