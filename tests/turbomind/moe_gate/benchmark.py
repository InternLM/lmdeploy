from __future__ import annotations

import argparse
import statistics

import torch

from . import turbomind_moe_gate
from .cases import bench_cases

DEFAULT_L2_FLUSH_BYTES = 64 * 1024 * 1024


class L2CacheFlusher:
    def __init__(self, device: torch.device, nbytes: int = DEFAULT_L2_FLUSH_BYTES):
        self.bytes = nbytes
        self._buffer = torch.empty(nbytes, device=device, dtype=torch.uint8)
        self._fill_value = 0

    def flush(self) -> None:
        self._buffer.fill_(self._fill_value)
        self._fill_value = 255 if self._fill_value == 0 else 0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float('nan')
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def time_moe_gate(
    logits: torch.Tensor,
    top_k: int,
    buffers: turbomind_moe_gate.MoeGateV2Buffers,
    token_mask: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    l2_flush: bool,
) -> dict[str, float]:
    device = logits.device
    flusher = L2CacheFlusher(device) if l2_flush else None
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        turbomind_moe_gate.moe_gate_v2(logits, top_k, token_mask=token_mask, buffers=buffers)
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(iters):
        if flusher is not None:
            flusher.flush()
        start.record(stream)
        turbomind_moe_gate.moe_gate_v2(logits, top_k, token_mask=token_mask, buffers=buffers)
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end))

    median_ms = statistics.median(samples)
    mean_ms = statistics.fmean(samples)
    p90_ms = _percentile(samples, 90)
    tokens, experts = logits.shape
    bytes_per_call = tokens * experts * 4
    gbps = (bytes_per_call / (median_ms / 1e3)) / 1e9
    return {
        'median_ms': median_ms,
        'mean_ms': mean_ms,
        'p90_ms': p90_ms,
        'gbps': gbps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Benchmark invokeMoeGate_V2 via _turbomind.moe_gate_v2')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--l2-flush', action='store_true', default=True)
    parser.add_argument('--no-l2-flush', action='store_false', dest='l2_flush')
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print('status=unavailable reason=no_cuda')
        return 1
    if not turbomind_moe_gate.is_available():
        print('status=unavailable reason=no_moe_gate_v2')
        return 1

    print(f'{"name":<24} {"median_ms":>10} {"mean_ms":>10} {"p90_ms":>10} {"GB/s":>10}')
    for case in bench_cases():
        logits = torch.randn(case.tokens, case.experts, device='cuda', dtype=torch.float32)
        buffers = turbomind_moe_gate.allocate_moe_gate_v2_buffers(case.tokens, case.experts, case.top_k)
        token_mask = torch.ones(case.tokens, device='cuda', dtype=torch.bool)
        stats = time_moe_gate(logits,
                              case.top_k,
                              buffers,
                              token_mask,
                              warmup=args.warmup,
                              iters=args.iters,
                              l2_flush=args.l2_flush)
        print(
            f'{case.name:<24} {stats["median_ms"]:10.3f} {stats["mean_ms"]:10.3f} '
            f'{stats["p90_ms"]:10.3f} {stats["gbps"]:10.2f}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
