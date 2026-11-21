"""Tight-loop benchmark for SM75/SM90 verification attention.

Run as a module:
  python -m tests.turbomind.attention.bench_verification_attention \
      --context 131072 --quant-policy 4 --splits 0
"""

import argparse

import torch

from .verification_attention import run_verification_attention


def _parse_context(value: str) -> int:
    value = value.strip().lower()
    if value.endswith('k'):
        return int(float(value[:-1]) * 1024)
    return int(value)


def _make_inputs(*, context: int, p: int, hq: int, hk: int, d: int,
                 block_len: int, quant_policy: int, seed: int):
    device = torch.device('cuda')
    torch.manual_seed(seed)
    dtype = torch.float16
    history = context - p
    scale = 0.2

    prefix_k = torch.randn(history, hk, d, device=device, dtype=dtype) * scale
    prefix_v = torch.randn_like(prefix_k) * scale
    packed_qkv = torch.randn(p, hq + 2 * hk, d, device=device,
                             dtype=dtype) * scale
    q_bias = torch.empty(0, device=device, dtype=dtype)

    prefix_offsets = torch.tensor([0, history], device=device, dtype=torch.int32)
    q_offsets = torch.tensor([0, p], device=device, dtype=torch.int32)
    k_offsets = torch.tensor([0, context], device=device, dtype=torch.int32)
    finished = torch.zeros(1, device=device, dtype=torch.bool)

    page_count = (context + block_len - 1) // block_len
    block_ptr_offsets = torch.tensor([0, page_count], device=device, dtype=torch.int32)
    # fp16-sized pages are large enough for the int4 layout too; ProcessKV
    # writes within the same page and the block pointer stride stays valid.
    page_bytes = 2 * hk * block_len * d * 2
    cache = torch.full((page_count, page_bytes), 0xA5, device=device,
                       dtype=torch.uint8)
    permutation = torch.randperm(page_count, device='cpu').tolist()
    base = cache.data_ptr()
    block_ptrs = torch.tensor(
        [base + physical * page_bytes for physical in permutation],
        device=device,
        dtype=torch.int64)

    query_count = p * (hq // hk)
    capacity = 128 * query_count
    output = torch.empty((p, hq, d), device=device, dtype=dtype)
    partial_o = torch.empty((capacity, hq, d), device=device, dtype=torch.float32)
    partial_ml = torch.empty((capacity, hq, 2), device=device, dtype=torch.float32)

    return dict(
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        prefix_offsets=prefix_offsets,
        packed_qkv=packed_qkv,
        q_bias=q_bias,
        output=output,
        cache_storage=cache,
        block_ptrs=block_ptrs,
        block_ptr_offsets=block_ptr_offsets,
        q_offsets=q_offsets,
        k_offsets=k_offsets,
        finished=finished,
        partial_o=partial_o,
        partial_ml=partial_ml,
        query_head_count=hq,
        kv_head_count=hk,
        head_dim=d,
        block_len=block_len,
        max_query_length=p,
        max_key_length=context,
        window_size=0,
        requested_max_split_count=128,
        quant_policy=quant_policy,
        rope_type=0,
        rope_dim=0,
        rope_base=1_000_000.0,
        rope_factor=1.0,
        mrope_mode=0,
        mrope_section=(0, 0, 0),
    )


def run_case(args) -> None:
    context = _parse_context(args.context)
    p = args.p
    inputs = _make_inputs(context=context,
                          p=p,
                          hq=args.hq,
                          hk=args.hk,
                          d=args.d,
                          block_len=args.block_len,
                          quant_policy=args.quant_policy,
                          seed=args.seed)
    split = run_verification_attention(**inputs,
                                       split_count_override=args.splits,
                                       process_kv=True)
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        run_verification_attention(**inputs,
                                   split_count_override=args.splits,
                                   process_kv=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_verification_attention(**inputs,
                                   split_count_override=args.splits,
                                   process_kv=False)
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / args.iters
    print(f'context={context} p={p} hq={args.hq} hk={args.hk} d={args.d} '
          f'quant={args.quant_policy} split={split} iters={args.iters} '
          f'ms={ms:.6f}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', default='128k')
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--hq', type=int, default=24)
    parser.add_argument('--hk', type=int, default=4)
    parser.add_argument('--d', type=int, default=256)
    parser.add_argument('--block-len', type=int, default=64)
    parser.add_argument('--quant-policy', type=int, default=4)
    parser.add_argument('--splits', type=int, default=0,
                        help='0 = automatic; otherwise force split count')
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=91)
    run_case(parser.parse_args())


if __name__ == '__main__':
    main()
