"""Tests for TurboQuant (quant_policy=QuantPolicy.TURBO_QUANT).

This module contains kernel-level tests for TurboQuant MSE quantization,
which is used by quant_policy=QuantPolicy.TURBO_QUANT (K=3bit QJL4, V=2bit mixed precision).

TurboQuant is a quantization method that:
- Uses Lloyd-Max algorithm for optimal quantization
- Applies random rotation for better distribution
- Stores only L2 norms (not scales/zeros) for dequantization
"""

import math

import pytest
import torch

# Also import turbo_quant kernels for direct access when needed
from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
    get_hadamard_matrix,
    get_lloyd_max_codebook,
)

# Import shared TurboQuant utilities to avoid duplication
from .turboquant_utils import (
    _unpack_indices,
    _unpack_qjl4_nibbles,
    dequantize_turboquant_mse,
    dequantize_turboquant_qjl4,
    quant_turboquant_mse,
    quant_turboquant_qjl4,
)


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

    @pytest.mark.parametrize('nbits', [2])
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

    @pytest.mark.parametrize('nbits', [2])
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

    @pytest.mark.parametrize('nbits', [2])
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
        # 2bit: ~0.80
        assert cos_sim > 0.79, f'2bit cosine similarity {cos_sim} too low'

    def test_determinism(self, head_dim):
        """Verify same input produces same output."""
        torch.manual_seed(42)
        x = torch.randn(10, head_dim).cuda()

        # Two quantizations should produce the same result
        q1, n1 = quant_turboquant_mse(x, 2)
        q2, n2 = quant_turboquant_mse(x, 2)

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(n1, n2)

        # Two dequantizations should produce the same result
        r1 = dequantize_turboquant_mse(q1, n1, 2)
        r2 = dequantize_turboquant_mse(q2, n2, 2)

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
        Pi = get_hadamard_matrix(head_dim, device=str(x.device))
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
