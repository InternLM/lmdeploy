# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch
import triton

WARPS_PER_SM = {
    (8, 0): 64,
    (8, 6): 48,
    (8, 7): 48,
    (8, 9): 48,
    (9, 0): 64,
    (10, 0): 64,
    (10, 1): 48,
    (11, 0): 48,
    (12, 0): 48,
}

BLOCKS_PER_SM = {
    (8, 0): 32,
    (8, 6): 16,
    (8, 7): 16,
    (8, 9): 24,
    (9, 0): 32,
    (10, 0): 32,
    (10, 1): 24,
    (11, 0): 24,
    (12, 0): 24,
}

@functools.lru_cache
def get_device_props(device=None):
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)

    warps_per_sm = WARPS_PER_SM.get((props.major, props.minor), 32)
    blocks_per_sm = BLOCKS_PER_SM.get((props.major, props.minor), warps_per_sm // 2)
    out = dict(
        multi_processor_count=props.multi_processor_count,
        compute_capability=(props.major, props.minor),
        warps_per_sm=warps_per_sm,
        blocks_per_sm=blocks_per_sm,
    )
    return out


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == 'cuda'


@functools.lru_cache
def supports_tma(device=None):
    if not is_cuda():
        return False

    return get_device_props(device)['compute_capability'][0] >= 9

@functools.lru_cache
def supports_pdl():
    """Whether Triton programmatic dependent launch is available."""
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9
