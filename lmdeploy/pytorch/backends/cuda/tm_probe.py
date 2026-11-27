# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Registry probe for the TurboMind W4A16 GEMM route."""

import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

# Registry verdicts are deterministic per (group_size, dtype, device), and
# get_linear() creates a short-lived executor per call. Cache the translated
# constraints so per-module probes during model construction do not churn
# executors and device synchronizations for identical formats.
_PLAN_CONSTRAINTS_CACHE: dict = {}


def _registry_constraints(group_size: int, probe_dtype: torch.dtype):
    """Return the registry shape constraints, or a rejection string."""
    key = (group_size, probe_dtype, torch.cuda.current_device())
    if key in _PLAN_CONSTRAINTS_CACHE:
        return _PLAN_CONSTRAINTS_CACHE[key]

    from lmdeploy.turbomind.linear import get_linear
    from lmdeploy.turbomind.weight_format import AWQFormat

    try:
        plan = get_linear().get_weight_plan(
            weight_format=AWQFormat(block_in=group_size),
            dtype=probe_dtype,
        )
    except NotImplementedError as e:
        logger.debug('TurboMind GEMM probe rejected format '
                     f'awq group_size={group_size} dtype={probe_dtype}: {e}')
        constraints = (f'no GEMM family accepts awq group_size={group_size} '
                       f'dtype={probe_dtype}: {e}')
    else:
        constraints = plan.shape_constraints
    _PLAN_CONSTRAINTS_CACHE[key] = constraints
    return constraints


def turbomind_support_reason(in_features: int,
                             out_features: int,
                             w_bit: int,
                             group_size: int,
                             dtype: torch.dtype | None = None) -> str | None:
    """Return why the TurboMind GEMM route cannot serve this layer.

    Return None when the layer is supported. The kernel registry is the ground truth. It applies arch and shape filters
    on its own. This probe only translates the registry verdict into a readable reason string.
    """
    if w_bit != 4:
        return f'w_bit must be 4, but got {w_bit}'
    if group_size <= 0:
        return f'group_size must be positive, but got {group_size}'
    if not torch.cuda.is_available():
        return 'CUDA is unavailable'

    from lmdeploy import turbomind
    if not turbomind.is_available():
        return 'TurboMind native modules are not built'

    probe_dtype = torch.float16 if dtype is None else dtype
    constraints = _registry_constraints(group_size, probe_dtype)
    if isinstance(constraints, str):
        return constraints

    (min_k, min_n), (align_k, align_n) = constraints
    if in_features % align_k != 0:
        return f'K must be a multiple of {align_k}, but got {in_features}'
    if out_features % align_n != 0:
        return f'N must be a multiple of {align_n}, but got {out_features}'
    if in_features < min_k:
        return f'K must be at least {min_k}, but got {in_features}'
    if out_features < min_n:
        return f'N must be at least {min_n}, but got {out_features}'
    return None
