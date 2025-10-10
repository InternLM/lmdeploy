import pytest
import torch


def _div_up(a, b):
    return (a + b - 1) // b


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
