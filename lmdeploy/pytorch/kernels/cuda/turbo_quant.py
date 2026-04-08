# Copyright (c) OpenMMLab. All rights reserved.
"""TurboQuant quantization utilities.

This module provides:
- Hadamard transform (orthogonal rotation) for quant_policy==QuantPolicy.TURBO_QUANT
- Lloyd-Max codebook for 2-bit, 3-bit, and 4-bit quantization
"""
import math

import torch
from torch import Tensor

_TURBOQUANT_CACHE = {}


def hadamard_rotate(x: Tensor) -> Tensor:
    """Apply normalized Hadamard transform: y = x @ Q.T

    Q is an orthogonal matrix (Q @ Q.T = I), so the transform is invertible
    via the transpose: x = y @ Q.

    Args:
        x: Input tensor of shape (..., d) where d is head dimension.

    Returns:
        Transformed tensor of same shape.
    """
    Q = _get_hadamard_matrix(x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.matmul(x, Q.T)


def hadamard_rotate_inv(x: Tensor) -> Tensor:
    """Inverse of hadamard_rotate: x = y @ Q

    Since Q is orthogonal: Q^{-1} = Q.T

    Args:
        x: Input tensor of shape (..., d) where d is head dimension.

    Returns:
        Inverse-transformed tensor of same shape.
    """
    Q = _get_hadamard_matrix(x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.matmul(x, Q)


def _get_hadamard_matrix(d: int, device: str = 'cuda', dtype=torch.float32) -> Tensor:
    """Get cached orthogonal Hadamard matrix Q = H @ diag(signs) / sqrt(d).

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

    cache_key = (d, device, str(dtype), 'hadamard_matrix')
    if cache_key in _TURBOQUANT_CACHE:
        return _TURBOQUANT_CACHE[cache_key]

    # Build normalized Hadamard matrix
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

        # Deterministic diagonal signs
        idx = torch.arange(d)
        signs = torch.where((idx & 1) == 0, 1.0, -1.0)

        # Q = H @ diag(signs)
        Q = (H * signs.unsqueeze(0)).to(device=device, dtype=dtype)

    _TURBOQUANT_CACHE[cache_key] = Q
    return Q


def get_lloyd_max_codebook(d: int, bits: int, device: str = 'cuda') -> tuple[Tensor, Tensor]:
    """Get precomputed Lloyd-Max codebook for 2-bit, 3-bit and 4-bit.

    The table is baked from the same construction logic as the original
    implementation under sigma=1, then scaled at runtime by sigma=1/sqrt(d).

    Supported:
        bits = 2, 3, 4

    Args:
        d: head dimension.
        bits: quantization bits (2, 3, or 4).
        device: target device.

    Returns:
        Tuple of (centroids, boundaries) tensors.
    """
    if bits not in (2, 3, 4):
        raise NotImplementedError(
            f'Only 2-bit, 3-bit and 4-bit precomputed codebooks are supported, got bits={bits}'
        )

    cache_key = (d, bits, device, 'codebook')
    if cache_key in _TURBOQUANT_CACHE:
        return _TURBOQUANT_CACHE[cache_key]

    sigma = 1.0 / math.sqrt(d)

    # Precomputed with the original implementation logic at sigma=1:
    #   - range [-3, 3]
    #   - uniform midpoint initialization
    #   - 10 Lloyd-Max iterations
    if bits == 2:
        centroids_std = torch.tensor(
            [-1.5104176, -0.4527808, 0.4527808, 1.5104176],
            device=device, dtype=torch.float32
        )
        boundaries_std = torch.tensor(
            [-0.9815992, 0.0, 0.9815992],
            device=device, dtype=torch.float32
        )
    elif bits == 3:
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
    else:  # bits == 4
        centroids_std = torch.tensor(
            [-2.4175594, -1.7094618, -1.2629677, -0.9265621,
             -0.6470380, -0.4015197, -0.1756835,  0.0391761,
              0.2508093,  0.4675656,  0.6996375,  0.9615010,
              1.2788204,  1.7009784,  2.3481500,  3.0000000],
            device=device, dtype=torch.float32
        )
        boundaries_std = torch.tensor(
            [-2.0635107, -1.4862148, -1.0947649, -0.7868000,
             -0.5242788, -0.2886016, -0.0682537,  0.1449927,
              0.3591875,  0.5836016,  0.8305693,  1.1201607,
              1.4898994,  2.0245643,  2.6740751],
            device=device, dtype=torch.float32
        )

    centroids = centroids_std * sigma
    boundaries = boundaries_std * sigma

    _TURBOQUANT_CACHE[cache_key] = (centroids, boundaries)
    return centroids, boundaries
