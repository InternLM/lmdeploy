# Copyright (c) OpenMMLab. All rights reserved.
"""Compare the production NCCL LM-head gather with symmetric memory.

Example:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=8 \
     benchmark/profile_lmhead_allgather.py \
     --tokens 1 8 32 128 --warmup 20 --repeat 200

The benchmark reports the maximum CUDA-event latency across TP ranks. V1
includes the owning clone performed by ``MultimemAllGatherer``.
"""

import argparse
import math
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lmdeploy.pytorch.backends.cuda.comm.symm_mem_allgather import MultimemAllGatherer


def _nccl_gather(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Match ``ParallelLMHead`` allocation, gather and vocabulary layout."""
    world_size = dist.get_world_size(group)
    output = local.new_empty((world_size, ) + tuple(local.shape))
    dist.all_gather_into_tensor(output, local, group=group)
    return output.movedim(0, 1).reshape(local.shape[0], -1)


def _percentile(samples: list[float], q: float) -> float:
    index = max(0, min(len(samples) - 1, math.ceil(q * len(samples)) - 1))
    return sorted(samples)[index]


def _measure_paired(
    baseline: Callable[[], torch.Tensor],
    target: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    group: dist.ProcessGroup,
    local_rank: int,
) -> dict[str, float]:
    """Interleave both paths and reduce every sample to the slowest rank."""
    for _ in range(warmup):
        baseline()
        target()
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[local_rank])

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    rank_times = torch.empty(2, dtype=torch.float64, device='cuda')
    baseline_samples = []
    target_samples = []
    for iteration in range(repeat):
        dist.barrier(group=group, device_ids=[local_rank])
        functions = (baseline, target) if iteration % 2 == 0 else (target, baseline)
        outputs = []
        for index, function in enumerate(functions):
            starts[index].record()
            outputs.append(function())
            ends[index].record()
        ends[-1].synchronize()
        elapsed = [starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(2)]
        if iteration % 2:
            elapsed.reverse()
        rank_times[0] = elapsed[0]
        rank_times[1] = elapsed[1]
        dist.all_reduce(rank_times, op=dist.ReduceOp.MAX, group=group)
        baseline_us, target_us = rank_times.tolist()
        baseline_samples.append(baseline_us)
        target_samples.append(target_us)
        del outputs

    baseline_median = statistics.median(baseline_samples)
    target_median = statistics.median(target_samples)
    return {
        'baseline_median_us': baseline_median,
        'target_median_us': target_median,
        'baseline_p95_us': _percentile(baseline_samples, 0.95),
        'target_p95_us': _percentile(target_samples, 0.95),
        'speedup': baseline_median / target_median,
    }


def _capture(function: Callable[[], torch.Tensor], group: dist.ProcessGroup, local_rank: int):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[local_rank])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[local_rank])
    return graph.replay, output, graph


def _report(rank: int, tokens: int, phase: str, metric: dict[str, float]) -> None:
    if rank != 0:
        return
    print(
        f'RESULT tokens={tokens} phase={phase} '
        f'base_median_us={metric["baseline_median_us"]:.3f} '
        f'v1_median_us={metric["target_median_us"]:.3f} '
        f'base_p95_us={metric["baseline_p95_us"]:.3f} '
        f'v1_p95_us={metric["target_p95_us"]:.3f} '
        f'speedup={metric["speedup"]:.4f}x',
        flush=True,
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocab-size', type=int, default=154880)
    parser.add_argument('--hidden-size', type=int, default=4096)
    parser.add_argument('--tokens', type=int, nargs='+', default=[1, 8, 32, 128])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=200)
    parser.add_argument('--skip-graph', action='store_true')
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if args.vocab_size % world_size:
        raise ValueError(f'vocab-size {args.vocab_size} must be divisible by TP={world_size}')

    local_vocab = args.vocab_size // world_size
    gatherer = MultimemAllGatherer(group, rank, args.vocab_size, max(args.tokens))
    if not gatherer.prepare(torch.device('cuda', local_rank)):
        raise RuntimeError('symmetric-memory multicast is unavailable for this TP topology')

    torch.manual_seed(1701 + rank)
    weight = torch.randn((local_vocab, args.hidden_size), dtype=torch.bfloat16, device='cuda')
    if rank == 0:
        print(
            f'CONFIG gpu={torch.cuda.get_device_name(local_rank)!r} tp={world_size} dtype=bf16 '
            f'hidden={args.hidden_size} global_vocab={args.vocab_size} local_vocab={local_vocab} '
            f'warmup={args.warmup} repeat={args.repeat}',
            flush=True,
        )

    for tokens in args.tokens:
        torch.manual_seed(2026 + rank * 17 + tokens)
        local_logits = torch.randn((tokens, local_vocab), dtype=torch.bfloat16, device='cuda')
        baseline_output = _nccl_gather(local_logits, group)
        target_output = gatherer(local_logits)
        if target_output is None:
            raise RuntimeError(f'symmetric-memory path rejected tokens={tokens}')
        torch.testing.assert_close(target_output, baseline_output, rtol=0, atol=0)

        def baseline():
            return _nccl_gather(local_logits, group)

        def target():
            return gatherer(local_logits)

        metric = _measure_paired(
            baseline,
            target,
            warmup=args.warmup,
            repeat=args.repeat,
            group=group,
            local_rank=local_rank,
        )
        _report(rank, tokens, 'gather', metric)

        torch.manual_seed(4096 + tokens)
        hidden = torch.randn((tokens, args.hidden_size), dtype=torch.bfloat16, device='cuda')

        def baseline_e2e():
            return _nccl_gather(F.linear(hidden, weight), group)

        def target_e2e():
            return gatherer(F.linear(hidden, weight))

        torch.testing.assert_close(target_e2e(), baseline_e2e(), rtol=0, atol=0)
        metric = _measure_paired(
            baseline_e2e,
            target_e2e,
            warmup=args.warmup,
            repeat=args.repeat,
            group=group,
            local_rank=local_rank,
        )
        _report(rank, tokens, 'gemm_gather', metric)

        if not args.skip_graph:
            baseline_replay, baseline_graph_output, baseline_graph = _capture(baseline_e2e, group, local_rank)
            target_replay, target_graph_output, target_graph = _capture(target_e2e, group, local_rank)
            baseline_replay()
            target_replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(target_graph_output, baseline_graph_output, rtol=0, atol=0)
            metric = _measure_paired(
                baseline_replay,
                target_replay,
                warmup=args.warmup,
                repeat=args.repeat,
                group=group,
                local_rank=local_rank,
            )
            _report(rank, tokens, 'graph_gemm_gather', metric)
            del baseline_replay, baseline_graph_output, baseline_graph
            del target_replay, target_graph_output, target_graph
            torch.cuda.synchronize()

    if rank == 0:
        print('PASS correctness=bitwise', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
