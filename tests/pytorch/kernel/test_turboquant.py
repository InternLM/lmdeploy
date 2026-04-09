"""Tests for TurboQuant (quant_policy=QuantPolicy.TURBO_QUANT).

This module contains kernel-level tests for TurboQuant MSE quantization,
which is used by quant_policy=QuantPolicy.TURBO_QUANT (K=4bit, V=2bit mixed precision).

TurboQuant is a quantization method that:
- Uses Lloyd-Max algorithm for optimal quantization
- Applies random rotation for better distribution
- Stores only L2 norms (not scales/zeros) for dequantization
"""

import math

import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
    _get_hadamard_matrix,
    get_lloyd_max_codebook,
)


def _div_up(a, b):
    return (a + b - 1) // b


# =============================================================================
# TurboQuant MSE Quantization/Dequantization Functions
# =============================================================================

_TQ_TEST_CACHE = {}


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
    Pi = _get_hadamard_matrix(head_dim, device=device)

    # Get Lloyd-Max codebook
    centroids, boundaries = get_lloyd_max_codebook(head_dim, nbits, device=device)
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


def quant_turboquant_qjl4(kv: torch.Tensor):
    """TurboQuant 4bit reference: 3bit MSE + 1bit QJL.

    Packed nibble layout for each coordinate:
        low  3 bits: MSE code index in [0, 7]
        high 1 bit : QJL residual sign

    Returns:
        q_kv: packed uint8 tensor, shape (..., D/2)
        meta: tensor of shape (..., 2)
              meta[..., 0] = mse_norm = ||x||
              meta[..., 1] = qjl_norm = ||residual|| / sqrt(D)
    """
    head_dim = kv.shape[-1]
    device = str(kv.device)

    Pi = _get_hadamard_matrix(head_dim, device=device)
    centroids, boundaries = get_lloyd_max_codebook(head_dim, bits=3,device=device)

    mse_norm = kv.norm(dim=-1, keepdim=True)  # (..., 1)
    kv_unit = kv / (mse_norm + 1e-10)
    y = torch.matmul(kv_unit, Pi.T)  # (..., D)

    idx3 = torch.searchsorted(boundaries, y.contiguous())
    idx3 = idx3.clamp(0, 7).to(torch.long)

    c = centroids[idx3]
    residual = y - c
    qjl_bit = (residual >= 0).to(torch.long)

    # Test-side reference qjl norm
    qjl_norm = residual.norm(dim=-1, keepdim=True) / math.sqrt(head_dim)

    # Pack 4bit nibble = low 3 bits mse idx + high 1 bit qjl sign
    nibble = idx3 | (qjl_bit << 3)

    q1, q2 = nibble.split(nibble.shape[-1] // 2, dim=-1)
    q_kv = q1 + (q2 << 4)

    meta = torch.cat([mse_norm, qjl_norm], dim=-1)  # (..., 2)
    return q_kv.to(torch.uint8), meta


def _unpack_indices(packed: torch.Tensor, nbits: int, original_dim: int) -> torch.Tensor:
    """Unpack bit-packed indices back to integer tensor."""
    # Save original shape
    orig_shape = list(packed.shape)
    batch_dims = orig_shape[:-1]
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

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


def _unpack_qjl4_nibbles(packed: torch.Tensor, original_dim: int):
    """Unpack 4bit qjl nibbles into:
    - idx3: [0, 7]
    - bit1: [0, 1]
    """
    nib = _unpack_indices(packed, 4, original_dim)
    idx3 = nib & 0x7
    bit1 = (nib >> 3) & 0x1
    return idx3.long(), bit1.long()


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
    Pi = _get_hadamard_matrix(head_dim, device=device)

    # Get Lloyd-Max codebook
    centroids, _ = get_lloyd_max_codebook(head_dim, nbits, device=device)

    # Unpack indices
    indices = _unpack_indices(q_kv, nbits, head_dim)

    # Look up centroids
    y_hat = centroids[indices]  # (..., head_dim)

    # Rotate back: x_hat = y_hat @ Pi
    x_hat = torch.matmul(y_hat, Pi)

    # Rescale by norms
    x_hat = x_hat * norms.unsqueeze(-1)

    return x_hat


def dequantize_turboquant_qjl4(q_kv: torch.Tensor, meta: torch.Tensor):
    """Dequantize test-side TurboQuant QJL4 (3bit MSE + 1bit QJL)."""
    head_dim = q_kv.shape[-1] * 2
    device = str(q_kv.device)

    Pi = _get_hadamard_matrix(head_dim, device=device)
    centroids, _ = get_lloyd_max_codebook(head_dim, bits=3, device=device)

    idx3, bit1 = _unpack_qjl4_nibbles(q_kv, head_dim)
    sign = bit1.to(torch.float32) * 2.0 - 1.0

    mse_norm = meta[..., 0]
    qjl_norm = meta[..., 1]

    c = centroids[idx3]
    y_hat = c + qjl_norm.unsqueeze(-1) * sign
    x_hat = torch.matmul(y_hat, Pi)
    x_hat = x_hat * mse_norm.unsqueeze(-1)
    return x_hat


class TestTurboQuantMSE:
    """Verify TurboQuant MSE quantization-dequantization correctness.

    These tests verify the core TurboQuant MSE algorithm used by quant_policy=QuantPolicy.TURBO_QUANT.
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


class TestTurboQuantQJL4:
    """Verify 4bit TurboQuant reference with 3bit MSE + 1bit QJL."""

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture
    def n_vectors(self):
        yield 100

    def test_quant_dequant_roundtrip(self, head_dim, n_vectors):
        torch.manual_seed(42)
        x = torch.randn(n_vectors, head_dim).cuda()

        q_x, meta = quant_turboquant_qjl4(x)

        assert q_x.shape == (n_vectors, head_dim // 2)
        assert meta.shape == (n_vectors, 2)

        idx3, bit1 = _unpack_qjl4_nibbles(q_x, head_dim)
        assert idx3.min().item() >= 0
        assert idx3.max().item() <= 7
        assert bit1.min().item() >= 0
        assert bit1.max().item() <= 1

        print(f'  qjl4: mse_norm range=[{meta[:,0].min():.3f}, {meta[:,0].max():.3f}]')
        print(f'  qjl4: qjl_norm range=[{meta[:,1].min():.3f}, {meta[:,1].max():.3f}]')

    def test_reconstruction_quality(self, head_dim, n_vectors):
        torch.manual_seed(42)
        x = torch.randn(n_vectors, head_dim).cuda()
        x = x / torch.norm(x, dim=-1, keepdim=True)

        q_x, meta = quant_turboquant_qjl4(x)
        x_reconstructed = dequantize_turboquant_qjl4(q_x, meta)

        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-10)
        recon_norm = x_reconstructed / (x_reconstructed.norm(dim=-1, keepdim=True) + 1e-10)
        cos_sim = (x_norm * recon_norm).sum(dim=-1).mean().item()
        mse = ((x - x_reconstructed)**2).mean().item()

        print(f'  qjl4: mse={mse:.6f}, cos_sim={cos_sim:.4f}')

        # This is a test-side reference construction, so use a moderate threshold first.
        assert cos_sim > 0.86, f'QJL4 cosine similarity {cos_sim} too low'

    def test_qjl4_not_worse_than_3bit_mse(self, head_dim, n_vectors):
        torch.manual_seed(42)
        x = torch.randn(n_vectors, head_dim).cuda()
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # Pure 3bit MSE baseline
        Pi = _get_hadamard_matrix(head_dim, device=str(x.device))
        centroids3, boundaries3 = get_lloyd_max_codebook(head_dim, bits=3, device=str(x.device))
        y = torch.matmul(x, Pi.T)
        idx3 = torch.searchsorted(boundaries3, y.contiguous()).clamp(0, 7)
        y3 = centroids3[idx3]
        x3 = torch.matmul(y3, Pi)

        mse_3bit = ((x - x3)**2).mean().item()

        q_x, meta = quant_turboquant_qjl4(x)
        x4 = dequantize_turboquant_qjl4(q_x, meta)
        mse_qjl4 = ((x - x4)**2).mean().item()

        print(f'  3bit_mse={mse_3bit:.6f}, qjl4={mse_qjl4:.6f}')
        assert mse_qjl4 <= mse_3bit * 1.05, 'QJL4 should not be significantly worse than pure 3bit MSE'

    def test_determinism(self, head_dim):
        torch.manual_seed(42)
        x = torch.randn(10, head_dim).cuda()

        q1, m1 = quant_turboquant_qjl4(x)
        q2, m2 = quant_turboquant_qjl4(x)

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(m1, m2)

        r1 = dequantize_turboquant_qjl4(q1, m1)
        r2 = dequantize_turboquant_qjl4(q2, m2)

        torch.testing.assert_close(r1, r2)
        print('  qjl4 determinism: OK')
