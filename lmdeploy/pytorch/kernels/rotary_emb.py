import triton
import triton.language as tl
import torch
from torch.nn import functional as F
import math

@triton.jit
def rotate_half_kernel(
        qk_seq_ptr,
        position_ids_ptr,
        qk_seq_stride,
        position_ids_batch_stride,
        seq_len,
        HEAD_DIM: tl.constexpr,
        BLOCK_HEIGHT: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        INV_BASE: tl.constexpr
):
    # qk_seq_ptr: (bsz, seq_len, 2, num_heads, head_dim) -- OK to be discontinuous in 2nd dimension.
    # position ids: (bsz, seq_len) -- must be contiguous in the last dimension.

    HALF_HEAD: tl.constexpr = HEAD_DIM // 2
    STEPS_PER_ROW: tl.constexpr = HALF_HEAD // BLOCK_WIDTH

    batch_seq = tl.program_id(axis=0)
    row_blk_x_col_blk = tl.program_id(axis=1)

    row_blk = row_blk_x_col_blk // STEPS_PER_ROW
    row = row_blk * BLOCK_HEIGHT
    if BLOCK_WIDTH < HALF_HEAD:
        col_blk = row_blk_x_col_blk % STEPS_PER_ROW
        col = col_blk * BLOCK_WIDTH
    else:
        col: tl.constexpr = 0

    # A block will never cross a sequence boundary, which simplifies things a lot.
    batch = batch_seq // seq_len
    seq = batch_seq % seq_len
    position_id = tl.load(position_ids_ptr + batch * position_ids_batch_stride + seq)
    # As sometimes happens, just calculating this on the fly is faster than loading it from memory.
    # Use `tl.libdevice.exp` rather than `tl.exp` -- the latter is less accurate.
    freq = tl.math.exp((col + tl.arange(0, BLOCK_WIDTH)).to(tl.float32) * INV_BASE) * position_id
    cos = tl.cos(freq).to(tl.float32)
    sin = tl.sin(freq).to(tl.float32)

    col_offsets: tl.constexpr = tl.arange(0, BLOCK_WIDTH)
    embed_offsets = (row * HEAD_DIM + col) + col_offsets
    x_ptrs = (qk_seq_ptr + batch_seq * qk_seq_stride) + embed_offsets

    for k in range(0, BLOCK_HEIGHT):
        x = tl.load(x_ptrs).to(tl.float32)
        y = tl.load(x_ptrs + HALF_HEAD).to(tl.float32)
        out_x = x * cos - y * sin
        tl.store(x_ptrs, out_x)
        out_y = x * sin + y * cos
        tl.store(x_ptrs + HALF_HEAD, out_y)
        x_ptrs += HEAD_DIM


def rotate_half(qk, position_ids, config=None):
    with torch.cuda.device(qk.device):
        batch_size, seq_len, qandk, num_heads, head_dim = qk.shape

        # This default is the fastest for most job sizes, at least on my RTX 4090, and when it's not it's within spitting distance of the best option. There are some odd cases where having a block height of 2 or 4 helps but the difference is within 5%. It makes sense that this configuration is fast from a memory bandwidth and caching perspective.
        config = config or {'BLOCK_HEIGHT': 1, 'BLOCK_WIDTH': min(128, head_dim // 2), 'num_warps': 1}
        config['BLOCK_HEIGHT'] = min(config['BLOCK_HEIGHT'], 2 * num_heads)

        assert qk.stride(3) == head_dim
        assert qk.stride(4) == 1
        assert position_ids.shape == (batch_size, seq_len)
        assert position_ids.stride(1) == 1, 'position_ids must be contiguous in the last dimension'
        assert (2 * num_heads) % config['BLOCK_HEIGHT'] == 0, f'number of rows not evenly divisible by {config["BLOCK_HEIGHT"]}'
        assert (head_dim // 2) % config['BLOCK_WIDTH'] == 0, f'number of columns ({head_dim // 2}) not evenly divisible by {config["BLOCK_WIDTH"]}'

        qk_by_seq = qk.view(batch_size * seq_len, 2 * num_heads * head_dim)
        grid = (qk_by_seq.shape[0], (2 * num_heads // config['BLOCK_HEIGHT']) * (head_dim // 2 // config['BLOCK_WIDTH']))

        # Must be the same as the theta of the frequencies used to train the model.
        BASE = 10000.0

        rotate_half_kernel[grid](
            qk_by_seq,
            position_ids,
            qk_by_seq.stride(0),
            position_ids.stride(0),
            seq_len,
            HEAD_DIM=head_dim,
            BLOCK_HEIGHT=config['BLOCK_HEIGHT'],
            BLOCK_WIDTH=config['BLOCK_WIDTH'],
            INV_BASE=-2.0 * math.log(BASE) / head_dim,
            num_warps=config['num_warps']
        )



BATCH = 32
HEAD = 32
DIM = 128

configs = [triton.testing.Benchmark(
    x_names=['n_ctx'],
    x_vals=[2 ** i for i in range(4, 12)],
    line_arg='provider',
    line_vals=['xformers','triton'] ,
    line_names=['Xformers', 'Triton'] ,
    styles=[  ('yellow', '-'),('red', '-')],
    ylabel='ms',
    plot_name=f'rotary',
    args={'bs': BATCH, 'h': HEAD, 'd': DIM, 'dtype': torch.float16})
]


@triton.testing.perf_report(configs)
def bench_rotate_half(bs,n_ctx, h, d , provider, dtype=torch.float16, device="cuda"):
    
    warmup = 25
    rep = 100
    
    
    qk = torch.randn((bs, n_ctx, 2, h,  d), dtype=dtype, device="cuda", requires_grad=False)
    position_inds = torch.arange(n_ctx, device=device).repeat(bs, 1).to(device)
    if provider == "triton":
        
        fn = lambda: rotate_half(qk, position_inds)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    
    if provider == "xformers":
        from xformers.components.positional_embedding import RotaryEmbedding
        mod = RotaryEmbedding(d).to(device)
        fn = lambda: mod(qk[:,:,0], qk[:,:,1])
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms


from pathlib import Path
save_path = Path('./rotate_half')
save_path.mkdir(exist_ok=True)
bench_rotate_half.run(save_path=save_path, print_data=True)