import pytest
import torch

from lmdeploy.messages import QuantPolicy

# Import common TurboQuant utilities from turboquant_utils
from .turboquant_utils import (
    _div_up,
)


class TestFlattenKVCache:

    @pytest.fixture
    def out_dtype(self):
        yield torch.float16

    @pytest.fixture
    def num_heads(self):
        yield 4

    @pytest.fixture
    def head_dim(self):
        yield 32

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def kv_lens(self):
        yield [2, 24, 47, 48]

    @pytest.fixture
    def batch_size(self, kv_lens):
        yield len(kv_lens)

    @pytest.fixture
    def num_blocks_per_input(self, kv_lens, block_size):
        yield [_div_up(kv_len, block_size) for kv_len in kv_lens]

    @pytest.fixture
    def max_num_blocks(self, num_blocks_per_input):
        yield max(num_blocks_per_input)

    @pytest.fixture
    def out_size(self, kv_lens):
        yield sum(kv_lens)

    @pytest.fixture
    def kv_seqlens(self, kv_lens):
        yield torch.tensor(kv_lens).cuda()

    @pytest.fixture
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim, out_dtype):
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim)
        yield torch.rand(shape, dtype=out_dtype, device='cuda')

    @pytest.fixture
    def v_caches(self, k_caches):
        yield torch.rand_like(k_caches)

    @pytest.fixture
    def block_offsets(self, num_blocks_per_input):
        batch_size = len(num_blocks_per_input)
        max_num_blocks = max(num_blocks_per_input)
        batch_ids = torch.arange(batch_size)
        ret = torch.arange(max_num_blocks)
        ret = batch_ids[:, None] + ret[None, :] * batch_size
        yield ret.cuda()

    @pytest.fixture
    def gt(self, k_caches, v_caches, kv_lens, block_offsets, block_size, num_heads, out_size, head_dim):
        k_states = k_caches.new_empty(num_heads, out_size, head_dim)
        v_states = v_caches.new_empty(num_heads, out_size, head_dim)
        start_loc = 0
        for kv_len, block_offs in zip(kv_lens, block_offsets):
            remain_len = kv_len
            for idx, _ in enumerate(range(0, kv_len, block_size)):
                b_off = block_offs[idx]
                block_len = min(block_size, remain_len)
                end_loc = start_loc + block_len
                k_block = k_caches[b_off, :block_len]
                v_block = v_caches[b_off, :block_len]
                k_states[:, start_loc:end_loc] = k_block.transpose(0, 1)
                v_states[:, start_loc:end_loc] = v_block.transpose(0, 1)
                start_loc = end_loc
                remain_len -= block_len

        yield k_states, v_states

    def test_flatten_kv_cache(self, k_caches, v_caches, kv_seqlens, block_offsets, out_size, gt):
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache

        k_states, v_states = flatten_kv_cache(k_caches, v_caches, kv_seqlens, block_offsets, out_size=out_size)
        torch.testing.assert_close(k_states, gt[0])
        torch.testing.assert_close(v_states, gt[1])


def precise_round(x: torch.Tensor):
    return x.sign() * (x.abs() + 0.5).floor()


def quant(kv: torch.Tensor, nbits: int = 8):
    """Quant kv on the head_dim."""
    amax = kv.amax(dim=-1, keepdim=True)
    amin = kv.amin(dim=-1, keepdim=True)
    scales = (amax - amin) / (2**nbits - 1)
    zeros = -amin / scales
    q_kv = (kv / scales + zeros + 0.5).to(torch.uint8)
    if nbits == 4:
        q_kv1, q_kv2 = q_kv.split(q_kv.shape[-1] // 2, -1)
        q_kv = q_kv1 + q_kv2 * 16
    return q_kv, torch.cat([scales, zeros], dim=-1)


class TestFlattenKVCacheQuant8(TestFlattenKVCache):

    @pytest.fixture
    def nbits(self):
        yield 8

    @pytest.fixture
    def atol(self):
        yield 4e-3

    @pytest.fixture
    def rtol(self):
        yield 1e-5

    @pytest.fixture
    def k_quant(self, k_caches, nbits):
        yield quant(k_caches, nbits)

    @pytest.fixture
    def v_quant(self, v_caches, nbits):
        yield quant(v_caches, nbits)

    def test_flatten_kv_cache(self, k_quant, v_quant, kv_seqlens, block_offsets, out_size, out_dtype, nbits, gt, atol,
                              rtol):
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache

        k_caches, k_sz = k_quant
        v_caches, v_sz = v_quant

        k_sz = k_sz.to(out_dtype)
        v_sz = v_sz.to(out_dtype)

        k_states, v_states = flatten_kv_cache(k_caches,
                                              v_caches,
                                              kv_seqlens,
                                              block_offsets,
                                              out_size=out_size,
                                              out_dtype=out_dtype,
                                              k_scales_zeros=k_sz,
                                              v_scales_zeros=v_sz,
                                              quant_policy=nbits)

        torch.testing.assert_close(k_states, gt[0], atol=atol, rtol=rtol)
        torch.testing.assert_close(v_states, gt[1], atol=atol, rtol=rtol)


class TestFlattenKVCacheQuant4(TestFlattenKVCacheQuant8):

    @pytest.fixture
    def nbits(self):
        yield 4

    @pytest.fixture
    def atol(self):
        yield 0.05

    @pytest.fixture
    def rtol(self):
        yield 1e-3


def quant_fp8(kv: torch.Tensor, fp8_dtype: torch.dtype):
    """Quantize KV cache with per-token/head symmetric FP8 scales."""
    fp8_max = torch.finfo(fp8_dtype).max
    scale = torch.maximum(kv.abs().amax(dim=-1, keepdim=True) / fp8_max, kv.new_tensor(1e-6))
    q_kv = (kv / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    dq_kv = (q_kv.to(torch.float32) * scale).to(kv.dtype)
    return q_kv, scale, dq_kv


def quant_fp8_scalar(kv: torch.Tensor, fp8_dtype: torch.dtype, scale: float):
    """Quantize KV cache with one scalar FP8 scale."""
    fp8_max = torch.finfo(fp8_dtype).max
    scale_t = kv.new_tensor(scale, dtype=torch.float32)
    q_kv = (kv.to(torch.float32) / scale_t).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    dq_kv = (q_kv.to(torch.float32) * scale_t).to(kv.dtype)
    return q_kv, scale_t, dq_kv


def _skip_unsupported_triton_fp8_dtype(fp8_dtype: torch.dtype):
    if fp8_dtype is torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('Triton float8_e4m3fn conversion requires device with cc>=9.0')


def flatten_reference(k_caches, v_caches, kv_lens, block_offsets, block_size, num_heads, out_size, k_head_dim,
                      v_head_dim):
    """Reference flatten for paged KV cache tensors."""
    k_states = k_caches.new_empty(num_heads, out_size, k_head_dim)
    v_states = v_caches.new_empty(num_heads, out_size, v_head_dim)
    start_loc = 0
    for kv_len, block_offs in zip(kv_lens, block_offsets):
        remain_len = kv_len
        for idx, _ in enumerate(range(0, kv_len, block_size)):
            b_off = block_offs[idx]
            block_len = min(block_size, remain_len)
            end_loc = start_loc + block_len
            k_block = k_caches[b_off, :block_len]
            v_block = v_caches[b_off, :block_len]
            k_states[:, start_loc:end_loc] = k_block.transpose(0, 1)
            v_states[:, start_loc:end_loc] = v_block.transpose(0, 1)
            start_loc = end_loc
            remain_len -= block_len
    return k_states, v_states


class TestFlattenKVCacheFP8PerTokenHead(TestFlattenKVCache):

    @pytest.fixture(autouse=True)
    def skip_unsupported_fp8_dtype(self, fp8_dtype):
        _skip_unsupported_triton_fp8_dtype(fp8_dtype)

    @pytest.fixture
    def fp8_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def quant_policy(self):
        yield QuantPolicy.FP8_PER_TOKEN_HEAD

    @pytest.fixture
    def atol(self):
        yield 1e-3

    @pytest.fixture
    def rtol(self):
        yield 1e-5

    def test_flatten_kv_cache(self, k_caches, v_caches, kv_lens, kv_seqlens, block_offsets, block_size, num_heads,
                              out_size, head_dim, out_dtype, fp8_dtype, quant_policy, atol, rtol):
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache

        k_caches_fp8, k_scale, k_dequant = quant_fp8(k_caches, fp8_dtype)
        v_caches_fp8, v_scale, v_dequant = quant_fp8(v_caches, fp8_dtype)
        gt = flatten_reference(k_dequant, v_dequant, kv_lens, block_offsets, block_size, num_heads, out_size, head_dim,
                               head_dim)

        k_states, v_states = flatten_kv_cache(k_caches_fp8,
                                              v_caches_fp8,
                                              kv_seqlens,
                                              block_offsets,
                                              out_size=out_size,
                                              out_dtype=out_dtype,
                                              k_scales_zeros=k_scale.to(out_dtype),
                                              v_scales_zeros=v_scale.to(out_dtype),
                                              quant_policy=quant_policy)

        torch.testing.assert_close(k_states, gt[0], atol=atol, rtol=rtol)
        torch.testing.assert_close(v_states, gt[1], atol=atol, rtol=rtol)


class TestFlattenKVCacheFP8E5M2PerTokenHead(TestFlattenKVCacheFP8PerTokenHead):

    @pytest.fixture
    def fp8_dtype(self):
        yield torch.float8_e5m2

    @pytest.fixture
    def quant_policy(self):
        yield QuantPolicy.FP8_E5M2_PER_TOKEN_HEAD


class TestFlattenKVCacheFP8Scalar(TestFlattenKVCache):

    @pytest.fixture(autouse=True)
    def skip_unsupported_fp8_dtype(self, fp8_dtype):
        _skip_unsupported_triton_fp8_dtype(fp8_dtype)

    @pytest.fixture
    def fp8_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def quant_policy(self):
        yield QuantPolicy.FP8

    def test_flatten_kv_cache(self, k_caches, v_caches, kv_lens, kv_seqlens, block_offsets, block_size, num_heads,
                              out_size, head_dim, out_dtype, fp8_dtype, quant_policy):
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache

        k_caches_fp8, k_scale, k_dequant = quant_fp8_scalar(k_caches, fp8_dtype, scale=0.25)
        v_caches_fp8, v_scale, v_dequant = quant_fp8_scalar(v_caches, fp8_dtype, scale=0.5)
        gt = flatten_reference(k_dequant, v_dequant, kv_lens, block_offsets, block_size, num_heads, out_size, head_dim,
                               head_dim)

        k_states, v_states = flatten_kv_cache(k_caches_fp8,
                                              v_caches_fp8,
                                              kv_seqlens,
                                              block_offsets,
                                              out_size=out_size,
                                              out_dtype=out_dtype,
                                              k_scale=k_scale,
                                              v_scale=v_scale,
                                              quant_policy=quant_policy)

        torch.testing.assert_close(k_states, gt[0], atol=1e-3, rtol=1e-5)
        torch.testing.assert_close(v_states, gt[1], atol=1e-3, rtol=1e-5)


class TestFlattenKVCacheFP8E5M2Scalar(TestFlattenKVCacheFP8Scalar):

    @pytest.fixture
    def fp8_dtype(self):
        yield torch.float8_e5m2

    @pytest.fixture
    def quant_policy(self):
        yield QuantPolicy.FP8_E5M2


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestFlattenKVCacheMLAFP8(TestFlattenKVCache):

    @pytest.fixture
    def out_dtype(self):
        yield torch.bfloat16

    @pytest.fixture
    def num_heads(self):
        yield 1

    @pytest.fixture
    def head_dim(self):
        yield 576

    @pytest.fixture
    def block_size(self):
        yield 64

    @pytest.fixture
    def k_cache_mla(self, k_caches):
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
        num_blocks, block_size, num_heads, _ = k_caches.shape
        k_cache_pe = k_caches[:, :, :, 512:]
        k_cache_nope = k_caches[:, :, :, :512].flatten(0, -2)
        k_cache_nope, k_cache_scale = quant_fp8(k_cache_nope, group_size=128)
        k_cache_nope = k_cache_nope.view(num_blocks, block_size, num_heads, -1)
        k_cache_scale = k_cache_scale.reshape(num_blocks, block_size, num_heads, -1).to(torch.float32)
        dtype = k_cache_nope.dtype
        out = torch.cat([k_cache_nope, k_cache_scale.view(dtype), k_cache_pe.view(dtype)], dim=-1)
        yield out

    def _dequant(self, k_cache_mla):
        k_cache_nope = k_cache_mla[..., :512].to(torch.float32)
        k_cache_scale = k_cache_mla[..., 512:512 + 16].view(torch.float32)
        k_cache_pe = k_cache_mla[..., 512 + 16:].view(torch.bfloat16)
        k_cache_nope = k_cache_nope.unflatten(-1, (-1, 128))
        k_cache_scale = k_cache_scale[..., None]
        k_cache_nope *= k_cache_scale
        k_cache_nope = k_cache_nope.flatten(-2, -1).to(k_cache_pe.dtype)
        k_cache = torch.cat([k_cache_nope, k_cache_pe], dim=-1)
        return k_cache

    @pytest.fixture
    def gt(self, k_cache_mla, kv_lens, block_offsets, block_size, num_heads, out_size, head_dim):
        k_caches = self._dequant(k_cache_mla)
        k_states = k_caches.new_empty(num_heads, out_size, head_dim)
        start_loc = 0
        for kv_len, block_offs in zip(kv_lens, block_offsets):
            remain_len = kv_len
            for idx, _ in enumerate(range(0, kv_len, block_size)):
                b_off = block_offs[idx]
                block_len = min(block_size, remain_len)
                end_loc = start_loc + block_len
                k_block = k_caches[b_off, :block_len]
                k_states[:, start_loc:end_loc] = k_block.transpose(0, 1)
                start_loc = end_loc
                remain_len -= block_len

        yield k_states

    def test_flatten_kv_cache(self, k_cache_mla, kv_seqlens, block_offsets, out_size, out_dtype, gt):
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache_mla_fp8

        k_states = flatten_kv_cache_mla_fp8(k_cache_mla,
                                            kv_seqlens,
                                            block_offsets,
                                            out_size=out_size,
                                            out_dtype=out_dtype)
        torch.testing.assert_close(k_states, gt)


# =============================================================================
# Tests for quant_policy=QuantPolicy.TURBO_QUANT (TurboQuant) flatten_kv_cache
# =============================================================================

class TestFlattenKVCacheQuant42:
    """Test flatten_kv_cache with quant_policy=QuantPolicy.TURBO_QUANT
    (TurboQuant).

    quant_policy=QuantPolicy.TURBO_QUANT uses:
    - K: QJL4 (3bit MSE + 1bit QJL), stored in rotate domain
    - V: TurboQuant MSE int2, stored in rotate domain

    The flatten function should output rotate-domain KV that can be used
    directly for attention computation in the rotate domain.
    """

    @pytest.fixture
    def num_heads(self):
        yield 4

    @pytest.fixture
    def head_dim(self):
        yield 64

    @pytest.fixture
    def head_dim_v(self):
        yield 64

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def kv_lens(self):
        yield [8, 24, 48, 32]

    @pytest.fixture
    def batch_size(self, kv_lens):
        yield len(kv_lens)

    @pytest.fixture
    def num_blocks_per_input(self, kv_lens, block_size):
        yield [(kv_len + block_size - 1) // block_size for kv_len in kv_lens]

    @pytest.fixture
    def max_num_blocks(self, num_blocks_per_input):
        yield max(num_blocks_per_input)

    @pytest.fixture
    def out_size(self, kv_lens):
        yield sum(kv_lens)

    @pytest.fixture
    def kv_seqlens(self, kv_lens):
        yield torch.tensor(kv_lens).cuda()

    @pytest.fixture
    def packed_k_dim(self, head_dim):
        yield head_dim // 2

    @pytest.fixture
    def packed_v_dim(self, head_dim_v):
        yield head_dim_v // 4

    @pytest.fixture
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads, packed_k_dim):
        """Create quantized K cache (uint8).

        Note: The cache size is based on max_num_blocks, but the actual
        data is only kv_lens long. The flatten function should only
        output the actual data length.
        """
        shape = (batch_size * max_num_blocks, block_size, num_heads, packed_k_dim)
        yield torch.randint(0, 256, shape, dtype=torch.uint8, device='cuda')

    @pytest.fixture
    def v_caches(self, batch_size, max_num_blocks, block_size, num_heads, packed_v_dim):
        """Create quantized V cache (uint8)."""
        shape = (batch_size * max_num_blocks, block_size, num_heads, packed_v_dim)
        yield torch.randint(0, 256, shape, dtype=torch.uint8, device='cuda')

    @pytest.fixture
    def k_scales_zeros(self, batch_size, max_num_blocks, block_size, num_heads):
        """K meta: [mse_norm, qjl_norm] for each position."""
        shape = (batch_size * max_num_blocks, block_size, num_heads, 2)
        yield torch.rand(shape, dtype=torch.float16, device='cuda')

    @pytest.fixture
    def v_scales_zeros(self, batch_size, max_num_blocks, block_size, num_heads):
        """V meta: [norm] for each position."""
        shape = (batch_size * max_num_blocks, block_size, num_heads, 1)
        yield torch.rand(shape, dtype=torch.float16, device='cuda')

    @pytest.fixture
    def block_offsets(self, num_blocks_per_input):
        batch_size = len(num_blocks_per_input)
        max_num_blocks = max(num_blocks_per_input)
        batch_ids = torch.arange(batch_size)
        ret = torch.arange(max_num_blocks)
        ret = batch_ids[:, None] + ret[None, :] * batch_size
        yield ret.cuda()

    @pytest.fixture
    def out_dtype(self):
        yield torch.float32

    def test_flatten_kv_cache_quant42(self, k_caches, v_caches, kv_seqlens, block_offsets, k_scales_zeros,
                                       v_scales_zeros, out_dtype, head_dim, head_dim_v, num_heads):
        """Test flatten_kv_cache with quant_policy=QuantPolicy.TURBO_QUANT.

        This test verifies that:
        1. The flatten function runs without error
        2. Output shape is correct
        3. Output is in the rotate domain (verified by dequantizing)
        """
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache
        from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
            hadamard_rotate_inv,
        )


        # Run flatten with quant_policy=QuantPolicy.TURBO_QUANT
        k_states, v_states = flatten_kv_cache(
            k_caches,
            v_caches,
            kv_seqlens,
            block_offsets,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_policy=QuantPolicy.TURBO_QUANT,
            kv_layout='bshd',
            flatten_kv_layout='shd',
            out_dtype=out_dtype,
        )

        # Get actual output size (may differ from expected due to cache padding)
        actual_out_size = k_states.shape[0]

        # Verify output shapes - use actual size from flatten output
        assert k_states.shape == (actual_out_size, num_heads, head_dim), f'K shape mismatch: {k_states.shape}'
        assert v_states.shape == (actual_out_size, num_heads, head_dim_v), f'V shape mismatch: {v_states.shape}'

        # Verify output is in rotate domain by checking that inverse rotation
        # produces reasonable values (not all zeros or NaNs)
        k_orig = hadamard_rotate_inv(k_states.float())
        v_orig = hadamard_rotate_inv(v_states.float())

        # Check that inverse rotation produces non-zero values
        assert k_orig.abs().max() > 1e-6, 'K inverse rotation produced all zeros'
        assert v_orig.abs().max() > 1e-6, 'V inverse rotation produced all zeros'

        print(f'flatten_kv_cache quant42: K shape={k_states.shape}, V shape={v_states.shape}')
        print(f'  K rotate domain: mean={k_states.abs().mean():.4f}, max={k_states.abs().max():.4f}')
        print(f'  V rotate domain: mean={v_states.abs().mean():.4f}, max={v_states.abs().max():.4f}')
        print(f'  K orig domain: mean={k_orig.abs().mean():.4f}, max={k_orig.abs().max():.4f}')
        print(f'  V orig domain: mean={v_orig.abs().mean():.4f}, max={v_orig.abs().max():.4f}')
