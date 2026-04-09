"""Common test utilities for TurboQuant (quant_policy=QuantPolicy.TURBO_QUANT)
kernel tests.

This module contains shared helper functions for testing TurboQuant quantization,
which is used by quant_policy=QuantPolicy.TURBO_QUANT (K=3bit QJL4, V=2bit mixed precision).

TurboQuant is a quantization method that:
- Uses Lloyd-Max algorithm for optimal quantization
- Applies Hadamard rotation for better distribution
- Stores only L2 norms (not scales/zeros) for dequantization
"""

import math

import torch

from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
    get_lloyd_max_codebook,
    hadamard_rotate,
    hadamard_rotate_inv,
)


def _div_up(a, b):
    """Integer division with rounding up."""
    return (a + b - 1) // b


def _unpack_indices(packed: torch.Tensor, nbits: int, original_dim: int) -> torch.Tensor:
    """Unpack bit-packed indices back to integer tensor."""
    if nbits == 2:
        i0 = (packed & 0x03)
        i1 = ((packed >> 2) & 0x03)
        i2 = ((packed >> 4) & 0x03)
        i3 = ((packed >> 6) & 0x03)
        indices = torch.cat([i0, i1, i2, i3], dim=-1)
    elif nbits == 4:
        # Unpack 2 nibbles per byte: low nibble and high nibble
        i0 = (packed & 0x0F)
        i1 = ((packed >> 4) & 0x0F)
        indices = torch.cat([i0, i1], dim=-1)
    else:
        indices = packed

    # Trim to original dimension
    return indices[..., :original_dim].long()


def _unpack_qjl4_nibbles(packed: torch.Tensor, original_dim: int):
    """Unpack 4bit qjl nibbles into:
    - idx3: [0, 7]
    - bit1: [0, 1]
    """
    nib = _unpack_indices(packed, 4, original_dim)
    idx3 = nib & 0x7
    bit1 = (nib >> 3) & 0x1
    return idx3.long(), bit1.long()


def quant_turboquant_mse(kv: torch.Tensor, nbits: int):
    """TurboQuant MSE quantization (without QJL).

    Args:
        kv: input tensor of shape (..., head_dim)
        nbits: number of bits (only 2 supported)

    Returns:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for dequantization, shape (...,)
    """
    head_dim = kv.shape[-1]
    device = str(kv.device)

    # Get Lloyd-Max codebook
    _, boundaries = get_lloyd_max_codebook(head_dim, nbits, device=device)

    # Compute L2 norms
    norms = kv.float().norm(dim=-1, keepdim=True)

    # Normalize to unit sphere
    kv_unit = kv.float() / (norms + 1e-10)
    y = hadamard_rotate(kv_unit)

    # Quantize: find nearest centroid via searchsorted
    indices = torch.searchsorted(boundaries, y.contiguous())
    indices = indices.clamp(0, 2 ** nbits - 1)

    # Bit-pack indices (2-bit: 4 values per byte)
    if nbits == 2:
        q_kv1, q_kv2, q_kv3, q_kv4 = indices.split(indices.shape[-1] // 4, -1)
        q_kv = q_kv1 + q_kv2 * 4 + q_kv3 * 16 + q_kv4 * 64
    else:
        q_kv = indices

    return q_kv.to(torch.uint8), norms.squeeze(-1)


def quant_turboquant_qjl4(kv: torch.Tensor):
    """TurboQuant QJL4 quantization for K: 3bit MSE + 1bit QJL.

    Returns:
        q_kv: packed uint8 tensor, shape (..., head_dim // 2)
        meta: (..., 2)
              meta[..., 0] = mse_norm
              meta[..., 1] = qjl_norm
    """
    head_dim = kv.shape[-1]
    device = str(kv.device)

    # Get Lloyd-Max codebook (3-bit)
    centroids, boundaries = get_lloyd_max_codebook(head_dim, 3, device=device)

    # Compute MSE norm
    mse_norm = kv.float().norm(dim=-1, keepdim=True)
    kv_unit = kv.float() / (mse_norm + 1e-10)

    # Apply hadamard rotation
    y = hadamard_rotate(kv_unit)

    # Quantize: find nearest centroid
    idx3 = torch.searchsorted(boundaries, y.contiguous()).clamp(0, 7).long()
    c = centroids[idx3]

    # Compute QJL residual
    residual = y - c
    qjl_bit = (residual >= 0).long()
    qjl_norm = residual.norm(dim=-1, keepdim=True) / math.sqrt(head_dim)

    # Pack nibble: low 3 bits = MSE index, high 1 bit = QJL sign
    nibble = idx3 | (qjl_bit << 3)
    q1, q2 = nibble.split(nibble.shape[-1] // 2, dim=-1)
    q_kv = (q1 + (q2 << 4)).to(torch.uint8)

    meta = torch.cat([mse_norm, qjl_norm], dim=-1)
    return q_kv, meta


def dequantize_turboquant_mse(q_kv: torch.Tensor, norms: torch.Tensor, nbits: int):
    """TurboQuant MSE dequantization (without QJL).

    Args:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for rescaling, shape (...,)
        nbits: number of bits (only 2 supported)

    Returns:
        reconstructed kv tensor in original domain
    """
    # First dequantize to rotate domain
    y_hat = dequantize_turboquant_mse_rot(q_kv, norms, nbits)
    # Then inverse rotate to original domain
    x_hat = hadamard_rotate_inv(y_hat)
    return x_hat


def dequantize_turboquant_mse_rot(q_kv: torch.Tensor, norms: torch.Tensor, nbits: int):
    """TurboQuant MSE dequantization to ROTATE domain (no inverse rotation).

    Args:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for rescaling, shape (...,)
        nbits: number of bits (only 2 supported)

    Returns:
        reconstructed kv tensor in rotate domain
    """
    # Infer head_dim from packed shape
    if nbits == 2:
        head_dim = q_kv.shape[-1] * 4
    else:
        head_dim = q_kv.shape[-1]

    device = str(q_kv.device)

    # Get Lloyd-Max codebook
    centroids, _ = get_lloyd_max_codebook(head_dim, nbits, device=device)

    # Unpack indices
    indices = _unpack_indices(q_kv, nbits, head_dim)

    # Look up centroids
    y_hat = centroids[indices]

    # Rescale by norms (in rotate domain, no inverse rotation)
    y_hat = y_hat * norms.unsqueeze(-1)

    return y_hat


def dequantize_turboquant_qjl4(q_kv: torch.Tensor, meta: torch.Tensor):
    """Dequantize TurboQuant QJL4 to original domain."""
    # First dequantize to rotate domain
    y_hat = dequantize_turboquant_qjl4_rot(q_kv, meta)
    # Then inverse rotate to original domain
    x_hat = hadamard_rotate_inv(y_hat)
    return x_hat


def dequantize_turboquant_qjl4_rot(q_kv: torch.Tensor, meta: torch.Tensor):
    """Dequantize TurboQuant QJL4 to ROTATE domain (no inverse rotation)."""
    head_dim = q_kv.shape[-1] * 2
    device = str(q_kv.device)

    # Get Lloyd-Max codebook (3-bit)
    centroids, _ = get_lloyd_max_codebook(head_dim, 3, device=device)

    # Unpack nibbles
    idx3, bit1 = _unpack_qjl4_nibbles(q_kv, head_dim)
    sign = bit1.float() * 2.0 - 1.0

    # Get meta values
    mse_norm = meta[..., 0]
    qjl_norm = meta[..., 1]

    # Reconstruct in rotate domain (no inverse rotation)
    y_hat = centroids[idx3] + qjl_norm.unsqueeze(-1) * sign
    y_hat = y_hat * mse_norm.unsqueeze(-1)

    return y_hat


def compute_metrics(a: torch.Tensor, b: torch.Tensor):
    """Compute similarity metrics between two tensors.

    Args:
        a, b: tensors to compare

    Returns:
        dict with 'cosine', 'nmse', 'snr_db' keys
    """
    import math

    a_flat = a.flatten()
    b_flat = b.flatten()
    cosine = torch.cosine_similarity(a_flat, b_flat, dim=0).item()
    mse = ((a - b) ** 2).mean().item()
    nmse = mse / (b ** 2).mean().item()
    signal = (b ** 2).mean().item()
    noise = ((a - b) ** 2).mean().item()
    snr_db = 10 * math.log10(signal / (noise + 1e-10))
    return {'cosine': cosine, 'nmse': nmse, 'snr_db': snr_db}
