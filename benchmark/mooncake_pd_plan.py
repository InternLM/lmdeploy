# Copyright (c) OpenMMLab. All rights reserved.
"""Measure PD plan construction without loading a model or allocating GPUs."""
import json
from statistics import median, quantiles
from time import perf_counter

from lmdeploy.pytorch.disagg.conn.protocol import DistServeCachePoolInfo
from lmdeploy.pytorch.engine.cache_engine.migration import build_cache_pool_assignments


def main():
    pool = DistServeCachePoolInfo(shape=(60, 4096, 4096), dtype='torch.uint8', element_size=1, entry_axis=1)
    results = []
    for tokens in (8192, 32768):
        for fragmented in (False, True):
            pairs = [(i, i * (2 if fragmented else 1)) for i in range(tokens // 64)]
            samples = []
            for _ in range(30):
                begin = perf_counter()
                plan = build_cache_pool_assignments((pool,), (pool,), pairs)
                samples.append((perf_counter() - begin) * 1000)
            results.append(dict(tokens=tokens, fragmented=fragmented, descriptors=len(plan),
                                bytes=sum(item.length for item in plan), p50_ms=median(samples),
                                p99_ms=quantiles(samples, n=100, method='inclusive')[98], max_ms=max(samples)))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
