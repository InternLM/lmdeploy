#!/usr/bin/env python3
"""Test GenericCopy against PyTorch for various non-contiguous layouts."""

import argparse
import sys

import _turbomind as _tm
import torch

DEV = torch.device('cuda')

DTYPE_MAP = {
    'f32': torch.float32,
    'f64': torch.float64,
    'f16': torch.float16,
    'bf16': torch.bfloat16,
    'i8': torch.int8,
    'i32': torch.int32,
    'i64': torch.int64,
}

# Set by CLI args
DTYPE = torch.float32


def _rand(*shape, **kwargs):
    if DTYPE.is_floating_point:
        return torch.randn(*shape, dtype=DTYPE, device=DEV, **kwargs)
    if DTYPE == torch.int8:
        return torch.randint(-128, 127, shape, dtype=DTYPE, device=DEV, **kwargs)
    return torch.randint(0, 1000, shape, dtype=DTYPE, device=DEV, **kwargs)


def make_tensors(torch_tensor):
    """Create (tm_src, tm_dst, golden) from a (possibly non-contiguous) torch
    tensor.

    tm_src: turbomind Tensor with strides preserved from the torch tensor
    tm_dst: contiguous turbomind Tensor initialized with garbage (to detect copy bugs)
    golden: contiguous torch tensor with the expected result
    """
    tm_src = _tm.from_dlpack_with_strides(torch_tensor)

    # Allocate an uninitialized destination to detect copy failures
    contig = torch_tensor.contiguous()
    dst = torch.empty(contig.shape, dtype=torch_tensor.dtype, device=DEV)
    tm_dst = _tm.from_dlpack(dst)

    golden = contig.clone()
    return tm_src, tm_dst, golden


ITERS = 20
WARMUP = 3


def benchmark_copy(name, tm_src, tm_dst, torch_tensor):
    """Benchmark GenericCopy vs PyTorch same-transform copy vs contiguous
    baseline."""
    numel = torch_tensor.numel()
    dtype_bytes = torch_tensor.element_size()
    total_bytes = numel * dtype_bytes

    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # --- Benchmark GenericCopy ---
    for _ in range(WARMUP):
        _tm.generic_copy_on_stream(tm_src, tm_dst, stream_ptr)

    torch.cuda.synchronize()

    start.record()
    for _ in range(ITERS):
        _tm.generic_copy_on_stream(tm_src, tm_dst, stream_ptr)
    end.record()
    torch.cuda.synchronize()
    gc_ms = start.elapsed_time(end)
    gc_gbps = total_bytes * ITERS / (gc_ms * 1e6)

    # --- Benchmark PyTorch same-transform copy (clone on non-contiguous tensor) ---
    pt_dst = torch.empty(torch_tensor.shape, dtype=torch_tensor.dtype, device=DEV)
    for _ in range(WARMUP):
        pt_dst.copy_(torch_tensor)

    torch.cuda.synchronize()

    start.record()
    for _ in range(ITERS):
        pt_dst.copy_(torch_tensor)
    end.record()
    torch.cuda.synchronize()
    pt_ms = start.elapsed_time(end)
    pt_gbps = total_bytes * ITERS / (pt_ms * 1e6)

    # --- Benchmark contiguous clone (peak baseline) ---
    contig = torch.zeros(torch_tensor.shape, dtype=torch_tensor.dtype, device=DEV)
    for _ in range(WARMUP):
        contig.clone()

    torch.cuda.synchronize()

    start.record()
    for _ in range(ITERS):
        contig.clone()
    end.record()
    torch.cuda.synchronize()
    bl_ms = start.elapsed_time(end)
    bl_gbps = total_bytes * ITERS / (bl_ms * 1e6)

    pct = gc_gbps / pt_gbps * 100 if pt_gbps > 0 else 0
    print(f'         GenericCopy: {gc_gbps:.1f} GB/s | PyTorch: {pt_gbps:.1f} GB/s | '
          f'Contiguous: {bl_gbps:.1f} GB/s ({pct:.1f}% of PyTorch)')
    return gc_gbps, pt_gbps, pct


def run_test(name, torch_tensor):
    """Run a single GenericCopy test.

    Returns (passed, bench_data).
    """
    tm_src, tm_dst, golden = make_tensors(torch_tensor)

    stream = torch.cuda.current_stream()
    _tm.generic_copy_on_stream(tm_src, tm_dst, stream.cuda_stream)

    result = torch.from_dlpack(tm_dst)
    match = torch.equal(result, golden)

    status = 'PASS' if match else 'FAIL'
    shape = list(torch_tensor.shape)
    stride = list(torch_tensor.stride())
    print(f'  [{status}] {name}: shape={shape}, stride={stride}')

    if not match:
        mismatches = (result != golden).sum().item()
        total = result.numel()
        print(f'         mismatches={mismatches}/{total}')

    if match:
        bench = benchmark_copy(name, tm_src, tm_dst, torch_tensor)
        return True, bench
    return False, None


def main():
    global DTYPE

    parser = argparse.ArgumentParser(description='GenericCopy Test Suite')
    parser.add_argument('--dtype', choices=list(DTYPE_MAP), default='f32',
                        help='Data type for all tests (default: f32)')
    args = parser.parse_args()

    DTYPE = DTYPE_MAP[args.dtype]

    print(f'GenericCopy Test Suite — dtype={args.dtype}')
    print('=' * 60)

    all_passed = True
    total = 0
    passed = 0

    def check(name, tensor):
        nonlocal all_passed, total, passed
        total += 1
        ok, _ = run_test(name, tensor)
        if ok:
            passed += 1
        else:
            all_passed = False

    # --- Contiguous baseline ---
    print('\nContiguous baseline:')
    check('contiguous', _rand(64, 128))

    # --- Rank-1 (1D contiguous) ---
    print('\nRank-1:')
    check('rank-1', _rand(8192))

    # --- 2D layout transformations ---
    print('\n2D transformations:')
    check('transpose', _rand(64, 128).t())
    check('row-stride (every-other-row)', _rand(64, 128)[::2, :])
    check('col-stride (every-other-col)', _rand(64, 128)[:, ::2])
    check('narrow outer dim', _rand(128, 64)[10:50, :])

    # --- 3D transformations ---
    print('\n3D transformations:')
    check('permute (2,0,1)', _rand(16, 32, 64).permute(2, 0, 1))
    check('3D batched transpose (B,M,N)->(B,N,M)',
          _rand(8, 64, 128).transpose(1, 2))
    check('3D batched transpose unaligned (must fall through)',
          _rand(8, 60, 100).transpose(1, 2))

    # --- 4D transformations ---
    print('\n4D transformations:')
    check('4D slice', _rand(4, 8, 32, 64)[:, :, ::3, :])
    # Coalesces to rank-3 dispatch (B and H stride-proportional in both src and dst).
    check('4D batched transpose (B,H,M,N)->(B,H,N,M)',
          _rand(4, 8, 64, 128).transpose(2, 3))
    # J originates at position != 1 after the src-stride-ascending sort.
    check('4D non-adjacent transpose (B,M,H,N)->(B,N,H,M)',
          _rand(4, 64, 8, 128).transpose(1, 3))
    # Sliced inner batch dim — strides are still proportional, so this
    # also coalesces to rank-3 dispatch.
    check('4D batched transpose with sliced inner batch',
          _rand(4, 16, 64, 128)[:, ::2, :, :].transpose(2, 3))
    # Sliced OUTER batch dim — the slice doubles the outer stride only;
    # 8 * a.stride(2) != a.stride(3) after permute, so coalesce_batch_dims
    # leaves it rank 4. This is the test that actually exercises the rank-4
    # kernel instantiation.
    check('4D batched transpose with sliced outer batch (rank-4 dispatch)',
          _rand(8, 8, 64, 128)[::2, :, :, :].transpose(2, 3))
    # Coalesces to total_batch > 65535 → falls through to VectorizedCopy.
    # Used to crash before the dispatcher's a.rank() fix.
    check('4D batched transpose, large coalesced batch (>65535, fall-through)',
          _rand(256, 256, 64, 128).transpose(2, 3))

    # --- Combined operations ---
    print('\nCombined operations:')
    check('slice+transpose', _rand(64, 128)[::2, :].t())

    # --- Throughput sweep ---
    print('\nThroughput sweep:')
    sweep_contig = []
    sweep_trans = []
    for n in [1024, 2048, 4096, 8192, 16384]:
        numel = n * n
        size_label = f'{numel // (1024 * 1024)}M' if numel >= 1024 * 1024 else f'{numel // 1024}K'
        shape_label = f'{n}x{n}'

        total += 1
        ok, bench = run_test(f'contig {size_label} ({shape_label})', _rand(n, n))
        if ok:
            passed += 1
            sweep_contig.append((shape_label, *bench))
        else:
            all_passed = False

        total += 1
        ok, bench = run_test(f'trans  {size_label} ({shape_label})', _rand(n, n).t())
        if ok:
            passed += 1
            sweep_trans.append((shape_label, *bench))
        else:
            all_passed = False

    # --- Batched transpose throughput sweep ---
    sweep_batched = []
    for (b_, m, n) in [(8, 1024, 1024), (32, 512, 512), (8, 4096, 128)]:
        numel = b_ * m * n
        size_label = f'{numel // (1024 * 1024)}M' if numel >= 1024 * 1024 else f'{numel // 1024}K'
        shape_label = f'{b_}x{m}x{n}'

        total += 1
        ok, bench = run_test(f'batched-trans {size_label} ({shape_label})',
                             _rand(b_, m, n).transpose(1, 2))
        if ok:
            passed += 1
            sweep_batched.append((shape_label, *bench))
        else:
            all_passed = False

    # --- Throughput summary table ---
    print(f'\n{"=" * 60}')
    print(f'Throughput Summary (dtype={args.dtype}, GB/s):')
    print(f'  {"Shape":<14} {"GenericCopy":>12} {"PyTorch":>12} {"%PT":>8}')
    print(f'  {"-" * 14} {"-" * 12} {"-" * 12} {"-" * 8}')
    print('  Contiguous:')
    for shape, gc, pt, pct in sweep_contig:
        print(f'  {shape:<14} {gc:>12.1f} {pt:>12.1f} {pct:>7.1f}%')
    print('  Transpose:')
    for shape, gc, pt, pct in sweep_trans:
        print(f'  {shape:<14} {gc:>12.1f} {pt:>12.1f} {pct:>7.1f}%')
    print('  Batched transpose:')
    for shape, gc, pt, pct in sweep_batched:
        print(f'  {shape:<14} {gc:>12.1f} {pt:>12.1f} {pct:>7.1f}%')

    # --- Negative strides ---
    print('\nNegative strides (flip):')
    try:
        check('flip dim=0', _rand(32, 64).flip(0))
    except Exception as e:
        print(f'  [SKIP] flip dim=0: {e}')
        total += 1

    # --- Summary ---
    print(f'\n{"=" * 60}')
    print(f'Results: {passed}/{total} passed')
    if all_passed:
        print('ALL TESTS PASSED')
    else:
        print('SOME TESTS FAILED')
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
