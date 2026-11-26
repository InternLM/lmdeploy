# Copyright (c) OpenMMLab. All rights reserved.
import time

import torch.multiprocessing as mp


def spawn_with_timeout(worker, args, nprocs=2, timeout=180):
    """Bound collective/graph tests even when a CUDA kernel cannot time out."""
    context = mp.spawn(worker, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=min(1, max(0, deadline - time.monotonic()))):
                return
        raise TimeoutError(f'{worker.__name__} exceeded {timeout}s')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
