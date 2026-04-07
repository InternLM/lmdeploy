"""Tests for TurboQuant (quant_policy=42).

This module contains kernel-level tests for TurboQuant MSE quantization,
which is used by quant_policy=42 (K=4bit, V=2bit mixed precision).

TurboQuant is a quantization method that:
- Uses Lloyd-Max algorithm for optimal quantization
- Applies random rotation for better distribution
- Stores only L2 norms (not scales/zeros) for dequantization
"""

import math

import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import (
    _get_lloyd_max_codebook,
    _get_rotation_matrix,
)


def _div_up(a, b):
    return (a + b - 1) // b


# =============================================================================
# TurboQuant MSE Quantization/Dequantization Functions
# =============================================================================


def quant_turboquant_mse(kv: torch.Tensor, nbits: int):
    """TurboQuant MSE quantization (without QJL).

    Args:
        kv: input tensor of shape (..., head_dim)
        nbits: number of bits (2 or 4)

    Returns:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for dequantization, shape (...,)
    """
    head_dim = kv.shape[-1]
    device = str(kv.device)

    # Get rotation matrix
    Pi = _get_rotation_matrix(head_dim, device=device)

    # Get Lloyd-Max codebook
    centroids, boundaries = _get_lloyd_max_codebook(head_dim, nbits, device=device)
    # boundaries now contains n_levels - 1 boundaries directly
    decision_boundaries = boundaries  # (n_levels - 1,)

    # Compute L2 norms
    norms = kv.norm(dim=-1, keepdim=True)

    # Normalize to unit sphere
    kv_unit = kv / (norms + 1e-10)

    # Apply random rotation: y = kv_unit @ Pi^T
    y = torch.matmul(kv_unit, Pi.T)

    # Quantize: find nearest centroid via searchsorted
    indices = torch.searchsorted(decision_boundaries, y.contiguous())
    indices = indices.clamp(0, 2 ** nbits - 1)

    # Bit-pack indices
    if nbits == 4:
        q_kv1, q_kv2 = indices.split(indices.shape[-1] // 2, -1)
        q_kv = q_kv1 + q_kv2 * 16
    elif nbits == 2:
        q_kv1, q_kv2, q_kv3, q_kv4 = indices.split(indices.shape[-1] // 4, -1)
        q_kv = q_kv1 + q_kv2 * 4 + q_kv3 * 16 + q_kv4 * 64
    else:
        q_kv = indices

    return q_kv.to(torch.uint8), norms.squeeze(-1)


def _unpack_indices(packed: torch.Tensor, nbits: int, original_dim: int) -> torch.Tensor:
    """Unpack bit-packed indices back to integer tensor."""
    # Save original shape
    orig_shape = list(packed.shape)
    batch_dims = orig_shape[:-1]
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    # Flatten all batch dims
    packed_flat = packed.flatten()  # [batch_size * packed_last_dim]

    if nbits == 4:
        packed_d = ((original_dim + 1) // 2) * 2
        required_packed = packed_d // 2
        total_required = batch_size * required_packed
        if packed_flat.shape[-1] < total_required:
            packed_flat = torch.nn.functional.pad(packed_flat, (0, total_required - packed_flat.shape[-1]), value=0)
    elif nbits == 2:
        packed_d = ((original_dim + 3) // 4) * 4
        required_packed = packed_d // 4
        total_required = batch_size * required_packed
        if packed_flat.shape[-1] < total_required:
            packed_flat = torch.nn.functional.pad(packed_flat, (0, total_required - packed_flat.shape[-1]), value=0)

    # Unpack
    if nbits == 4:
        low = (packed & 0x0F)          # (..., d/2) ->  indices[0 : d/2]
        high = (packed >> 4) & 0x0F    # (..., d/2) ->  indices[d/2 : d]
        indices = torch.cat([low, high], dim=-1)  # (..., d)

    elif nbits == 2:
        i0 = (packed & 0x03)            # (..., d/4) -> indices[0 : d/4]
        i1 = ((packed >> 2) & 0x03)     # (..., d/4) -> indices[d/4 : d/2]
        i2 = ((packed >> 4) & 0x03)     # (..., d/4) -> indices[d/2 : 3d/4]
        i3 = ((packed >> 6) & 0x03)     # (..., d/4) -> indices[3d/4 : d]
        indices = torch.cat([i0, i1, i2, i3], dim=-1)  # (..., d)

    else:
        indices = packed

    # Trim to exact size and reshape
    new_shape = batch_dims + [original_dim]
    return indices[:, :original_dim].reshape(new_shape).long()


def dequantize_turboquant_mse(q_kv: torch.Tensor, norms: torch.Tensor, nbits: int):
    """TurboQuant MSE dequantization (without QJL).

    Args:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for rescaling, shape (...,)
        nbits: number of bits (2 or 4)

    Returns:
        reconstructed kv tensor
    """
    # Infer head_dim from packed shape
    if nbits == 4:
        head_dim = q_kv.shape[-1] * 2
    elif nbits == 2:
        head_dim = q_kv.shape[-1] * 4
    else:
        head_dim = q_kv.shape[-1]

    device = str(q_kv.device)

    # Get rotation matrix
    Pi = _get_rotation_matrix(head_dim, device=device)

    # Get Lloyd-Max codebook
    centroids, _ = _get_lloyd_max_codebook(head_dim, nbits, device=device)

    # Unpack indices
    indices = _unpack_indices(q_kv, nbits, head_dim)

    # Look up centroids
    y_hat = centroids[indices]  # (..., head_dim)

    # Rotate back: x_hat = y_hat @ Pi
    x_hat = torch.matmul(y_hat, Pi)

    # Rescale by norms
    x_hat = x_hat * norms.unsqueeze(-1)

    return x_hat

class TestTurboQuantMSE:
    """Verify TurboQuant MSE quantization-dequantization correctness.

    These tests verify the core TurboQuant MSE algorithm used by quant_policy=42.
    """

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture
    def n_vectors(self):
        yield 100

    @pytest.mark.parametrize('nbits', [2, 4])
    def test_quant_dequant_roundtrip(self, head_dim, n_vectors, nbits):
        """Test quantization-dequantization roundtrip."""
        torch.manual_seed(42)
        x = torch.randn(n_vectors, head_dim).cuda()

        # Quantize
        q_x, norms = quant_turboquant_mse(x, nbits)

        # Verify norms shape is correct
        assert norms.shape == (n_vectors,), f'norms shape incorrect: {norms.shape}'

        # Verify quantized values are in valid range
        max_val = 2 ** nbits - 1
        # Unpack and verify
        unpacked = _unpack_indices(q_x, nbits, head_dim)
        assert unpacked.max().item() <= max_val, 'quantized value exceeds range'
        assert unpacked.min().item() >= 0, 'quantized value less than 0'

        print(f'  bits={nbits}: quant OK, norms range=[{norms.min():.3f}, {norms.max():.3f}]')

    @pytest.mark.parametrize('nbits', [2, 4])
    def test_mse_within_theoretical_bound(self, head_dim, n_vectors, nbits):
        """Verify quantization-dequantization MSE is within theoretical bound
        (for unit vectors)."""
        torch.manual_seed(42)
        x = torch.randn(n_vectors, head_dim).cuda()
        # Normalize to unit sphere (theoretical bound is for unit vectors)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # Quantize
        q_x, norms = quant_turboquant_mse(x, nbits)

        # Dequantize
        x_reconstructed = dequantize_turboquant_mse(q_x, norms, nbits)

        # Compute MSE
        mse = ((x - x_reconstructed) ** 2).mean().item()

        # Theoretical bound: D_mse <= sqrt(3)*pi/2 * (1/4^bits)
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4 ** nbits))

        ratio = mse / theoretical_bound

        print(f'  bits={nbits}: MSE={mse:.6f}, theory_bound={theoretical_bound:.6f}, ratio={ratio:.3f}')

        # Theoretical bound is an upper bound, actual MSE must be less
        assert ratio < 1, f'MSE {mse} exceeds theoretical bound {theoretical_bound} (ratio={ratio:.3f})'

    @pytest.mark.parametrize('nbits', [2, 4])
    def test_reconstruction_quality(self, head_dim, n_vectors, nbits):
        """Verify reconstruction quality (using cosine similarity for unit
        vectors).

        For unit vectors, cosine similarity better reflects the effect of quantization on direction.
        """
        torch.manual_seed(42)
        x = torch.randn(n_vectors, head_dim).cuda()
        # Normalize to unit sphere
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # Quantize
        q_x, norms = quant_turboquant_mse(x, nbits)

        # Dequantize
        x_reconstructed = dequantize_turboquant_mse(q_x, norms, nbits)

        # Compute cosine similarity (after normalization)
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-10)
        recon_norm = x_reconstructed / (x_reconstructed.norm(dim=-1, keepdim=True) + 1e-10)
        cos_sim = (x_norm * recon_norm).sum(dim=-1).mean().item()

        print(f'  bits={nbits}: cos_sim={cos_sim:.4f}')

        # Cosine similarity should be close to 1.0
        # 4bit: ~0.90, 2bit: ~0.80
        if nbits == 4:
            assert cos_sim > 0.89, f'4bit cosine similarity {cos_sim} too low'
        else:
            assert cos_sim > 0.79, f'2bit cosine similarity {cos_sim} too low'

    def test_determinism(self, head_dim):
        """Verify same input produces same output."""
        torch.manual_seed(42)
        x = torch.randn(10, head_dim).cuda()

        # Two quantizations should produce the same result
        q1, n1 = quant_turboquant_mse(x, 4)
        q2, n2 = quant_turboquant_mse(x, 4)

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(n1, n2)

        # Two dequantizations should produce the same result
        r1 = dequantize_turboquant_mse(q1, n1, 4)
        r2 = dequantize_turboquant_mse(q2, n2, 4)

        torch.testing.assert_close(r1, r2)
        print('  determinism: OK')
