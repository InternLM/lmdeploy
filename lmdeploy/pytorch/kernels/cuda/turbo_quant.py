# Copyright (c) OpenMMLab. All rights reserved.
"""TurboQuant quantization utilities.

This module provides:
- Hadamard transform (orthogonal rotation) for quant_policy==QuantPolicy.TURBO_QUANT
- Lloyd-Max codebook for 2-bit (V cache) and 3-bit (K cache) quantization
"""
import logging
import math

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_TURBOQUANT_CACHE = {}

# Try to import fast_hadamard_transform for better performance
try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform

    _USE_FAST_HADAMARD = True
except ImportError:
    _USE_FAST_HADAMARD = False
    logger.info(
        'fast_hadamard_transform not installed, falling back to matmul-based '
        'Hadamard transform. Install it for better performance.'
    )


def hadamard_rotate(x: Tensor) -> Tensor:
    """Apply normalized Hadamard transform: y = x @ Q.T

    Q is an orthogonal matrix (Q @ Q.T = I), so the transform is invertible
    via the transpose: x = y @ Q.

    This function internally casts to float32 for computation to maintain
    precision, then casts back to the original dtype.

    Args:
        x: Input tensor of shape (..., d) where d is head dimension.

    Returns:
        Transformed tensor of same shape, in original dtype.
    """
    if _USE_FAST_HADAMARD:
        d = x.shape[-1]
        scale = 1.0 / math.sqrt(d)
        return _fast_hadamard_transform(x, scale=scale)

    # Fallback: use matmul with Walsh-Hadamard matrix
    orig_dtype = x.dtype
    x = x.float()
    Q = get_hadamard_matrix(
        x.shape[-1], device=x.device, dtype=torch.float32
    )
    result = torch.matmul(x, Q.T)
    return result.to(orig_dtype)


def hadamard_rotate_inv(x: Tensor) -> Tensor:
    """Inverse of hadamard_rotate: x = y @ Q

    Since Q is orthogonal: Q^{-1} = Q.T

    This function internally casts to float32 for computation to maintain
    precision, then casts back to the original dtype.

    Args:
        x: Input tensor of shape (..., d) where d is head dimension.

    Returns:
        Inverse-transformed tensor of same shape, in original dtype.
    """
    if _USE_FAST_HADAMARD:
        # Hadamard is self-inverse (up to scaling)
        d = x.shape[-1]
        scale = 1.0 / math.sqrt(d)
        return _fast_hadamard_transform(x, scale=scale)

    # Fallback: use matmul with Walsh-Hadamard matrix
    orig_dtype = x.dtype
    x = x.float()
    Q = get_hadamard_matrix(
        x.shape[-1], device=x.device, dtype=torch.float32
    )
    result = torch.matmul(x, Q)
    return result.to(orig_dtype)


def get_hadamard_matrix(d: int, device: str = 'cuda', dtype=torch.float32) -> Tensor:
    """Get cached Walsh-Hadamard matrix Q = H / sqrt(d).

    This is the standard Walsh-Hadamard matrix (same as fast_hadamard_transform).
    Q is orthogonal: Q @ Q.T = I, so Q^{-1} = Q.T.

    Args:
        d: head dimension (must be power of 2).
        device: target device.
        dtype: storage dtype for the matrix.

    Returns:
        Q: (d, d) tensor.
    """
    if d & (d - 1) != 0:
        raise ValueError(
            f'Hadamard matrix requires power-of-2 dimension, got d={d}'
        )

    cache_key = (d, str(device), str(dtype), 'hadamard_matrix')
    if cache_key in _TURBOQUANT_CACHE:
        return _TURBOQUANT_CACHE[cache_key]

    # Build normalized Walsh-Hadamard matrix
    with torch.no_grad():
        H = torch.tensor([[1.0]], dtype=torch.float32)
        n = 1
        while n < d:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
            n *= 2
        H = H / math.sqrt(d)
        Q = H.to(device=device, dtype=dtype)

    _TURBOQUANT_CACHE[cache_key] = Q
    return Q


def get_lloyd_max_codebook(d: int, bits: int, device: str = 'cuda') -> tuple[Tensor, Tensor]:
    """Get precomputed Lloyd-Max codebook for 2-bit and 3-bit quantization.

    The table is baked from the same construction logic as the original
    implementation under sigma=1, then scaled at runtime by sigma=1/sqrt(d).

    Supported:
        bits = 2 (V cache), 3 (K cache with QJL4)

    Args:
        d: head dimension.
        bits: quantization bits (2 or 3).
        device: target device.

    Returns:
        Tuple of (centroids, boundaries) tensors.
    """
    if bits not in (2, 3):
        raise NotImplementedError(
            f'Only 2-bit and 3-bit precomputed codebooks are supported, got bits={bits}'
        )

    cache_key = (d, bits, str(device), 'codebook')
    if cache_key in _TURBOQUANT_CACHE:
        return _TURBOQUANT_CACHE[cache_key]

    sigma = 1.0 / math.sqrt(d)

    # Precomputed with the original implementation logic at sigma=1:
    #   - range [-3, 3]
    #   - uniform midpoint initialization
    #   - 10 Lloyd-Max iterations
    if bits == 2:
        # 2-bit Lloyd-Max centroids (V cache)
        centroids_std = torch.tensor(
            [-1.5104176, -0.4527808, 0.4527808, 1.5104176],
            device=device, dtype=torch.float32
        )
        boundaries_std = torch.tensor(
            [-0.9815992, 0.0, 0.9815992],
            device=device, dtype=torch.float32
        )
    else:  # bits == 3
        # 3-bit Lloyd-Max centroids (K cache with QJL4)
        centroids_std = torch.tensor(
            [-2.1519456, -1.3439093, -0.7560052, -0.2450942,
              0.2450942,  0.7560052,  1.3439093,  2.1519456],
            device=device,
            dtype=torch.float32,
        )
        boundaries_std = torch.tensor(
            [-1.7479274, -1.0499573, -0.5005497, 0.0,
              0.5005497,  1.0499573,  1.7479274],
            device=device,
            dtype=torch.float32,
        )

    centroids = centroids_std * sigma
    boundaries = boundaries_std * sigma

    _TURBOQUANT_CACHE[cache_key] = (centroids, boundaries)
    return centroids, boundaries
