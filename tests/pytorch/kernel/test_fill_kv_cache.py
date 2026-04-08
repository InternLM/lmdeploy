import math

import pytest
import torch

# Import common TurboQuant utilities from turboquant_utils
from .turboquant_utils import (
    _div_up,
    dequantize_turboquant_qjl4,
    quant_turboquant_mse,
    quant_turboquant_qjl4,
)


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
    elif nbits == 2:
        q_kv1, q_kv2, q_kv3, q_kv4 = q_kv.split(q_kv.shape[-1] // 4, -1)
        q_kv = q_kv1 + q_kv2 * 4 + q_kv3 * 16 + q_kv4 * 64
    return q_kv, torch.cat([scales, zeros], dim=-1)


class TestFillKVCache:

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
    def seq_lens(self, request):
        yield request.param

    @pytest.fixture
    def history_lens(self, request):
        yield request.param

    @pytest.fixture
    def batch_size(self, seq_lens):
        yield len(seq_lens)

    @pytest.fixture
    def kv_lens(self, seq_lens, history_lens):
        yield [s + h for s, h in zip(seq_lens, history_lens)]

    @pytest.fixture
    def max_q_seq_length(self, seq_lens):
        yield max(seq_lens)

    @pytest.fixture
    def num_tokens(self, seq_lens):
        yield sum(seq_lens)

    @pytest.fixture
    def num_blocks_per_input(self, kv_lens, block_size):
        yield [_div_up(kv_len, block_size) for kv_len in kv_lens]

    @pytest.fixture
    def max_num_blocks(self, num_blocks_per_input):
        yield max(num_blocks_per_input)

    @pytest.fixture
    def q_seq_length(self, seq_lens):
        yield torch.tensor(seq_lens).cuda()

    @pytest.fixture
    def q_start_loc(self, q_seq_length):
        cum_seq_length = q_seq_length.cumsum(0)
        yield cum_seq_length - q_seq_length

    @pytest.fixture
    def kv_seq_length(self, kv_lens):
        yield torch.tensor(kv_lens).cuda()

    @pytest.fixture
    def k_states(self, num_tokens, num_heads, head_dim):
        yield torch.randn(num_tokens, num_heads, head_dim).cuda()

    @pytest.fixture
    def v_states(self, k_states):
        yield torch.randn_like(k_states)

    @pytest.fixture
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim):
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim)
        yield torch.full(shape, 0.0).cuda()

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
    def gt(self, k_states, v_states, k_caches, v_caches, seq_lens, history_lens, block_offsets, block_size):
        batch_size = len(seq_lens)
        k_caches = k_caches.clone()
        v_caches = v_caches.clone()
        splited_k_states = k_states.split(seq_lens)
        splited_v_states = v_states.split(seq_lens)

        for bidx in range(batch_size):
            k_state = splited_k_states[bidx]
            v_state = splited_v_states[bidx]
            h_len = history_lens[bidx]
            b_offs = block_offsets[bidx]
            block_id = _div_up(h_len + 1, block_size) - 1
            fill_start = h_len % block_size
            fill_size = min(block_size - fill_start, k_state.size(0))
            while True:
                boff = b_offs[block_id]
                tmp_ks = k_state[:fill_size]
                tmp_vs = v_state[:fill_size]
                fill_end = fill_start + fill_size
                k_caches[boff, fill_start:fill_end] = tmp_ks
                v_caches[boff, fill_start:fill_end] = tmp_vs

                k_state = k_state[fill_size:]
                v_state = v_state[fill_size:]
                block_id += 1
                fill_start = 0
                fill_size = min(block_size, k_state.size(0))
                if fill_size == 0:
                    break

        yield k_caches, v_caches

    @pytest.mark.parametrize(['seq_lens', 'history_lens'], [
        ((1, 1, 1, 1), (1, 16, 31, 24)),
        ((1, 8, 16, 24), (1, 16, 31, 24)),
    ],
                             indirect=True)
    def test_fill_kv_cache(self, k_states, v_states, k_caches, v_caches, block_offsets, q_start_loc, q_seq_length,
                           kv_seq_length, max_q_seq_length, gt):
        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache
        fill_kv_cache(k_states, v_states, k_caches, v_caches, q_start_loc, q_seq_length, kv_seq_length,
                      max_q_seq_length, block_offsets)

        torch.testing.assert_close(k_caches, gt[0])
        torch.testing.assert_close(v_caches, gt[1])


class TestFillKVCacheInt8(TestFillKVCache):

    @pytest.fixture
    def head_dim(self, request):
        yield request.param

    @pytest.fixture
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim):
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim)
        yield torch.full(shape, 0, dtype=torch.uint8).cuda()

    @pytest.fixture
    def v_caches(self, k_caches):
        yield torch.full_like(k_caches.to(torch.float32), 0).to(torch.uint8)

    @pytest.fixture
    def k_scales_zeros(self, batch_size, max_num_blocks, block_size, num_heads):
        shape = (batch_size * max_num_blocks, block_size, num_heads, 2)
        yield torch.full(shape, 0.0).cuda()

    @pytest.fixture
    def v_scales_zeros(self, k_scales_zeros):
        yield torch.zeros_like(k_scales_zeros)

    @pytest.fixture
    def nbits(self):
        yield 8

    @pytest.fixture
    def gt(self, k_states, v_states, k_caches, v_caches, seq_lens, history_lens, block_offsets, block_size,
           k_scales_zeros, v_scales_zeros, nbits):
        k_states, k_states_sz = quant(k_states, nbits)
        v_states, v_states_sz = quant(v_states, nbits)

        batch_size = len(seq_lens)
        k_caches = k_caches.clone()
        v_caches = v_caches.clone()
        k_scales_zeros = k_scales_zeros.clone()
        v_scales_zeros = v_scales_zeros.clone()

        splited_k_states = k_states.split(seq_lens)
        splited_v_states = v_states.split(seq_lens)
        splited_k_states_sz = k_states_sz.split(seq_lens)
        splited_v_states_sz = v_states_sz.split(seq_lens)

        for bidx in range(batch_size):
            k_state = splited_k_states[bidx]
            v_state = splited_v_states[bidx]
            k_state_sz = splited_k_states_sz[bidx]
            v_state_sz = splited_v_states_sz[bidx]

            h_len = history_lens[bidx]
            b_offs = block_offsets[bidx]
            block_id = _div_up(h_len + 1, block_size) - 1
            fill_start = h_len % block_size
            fill_size = min(block_size - fill_start, k_state.size(0))

            while True:
                boff = b_offs[block_id]
                tmp_ks = k_state[:fill_size]
                tmp_vs = v_state[:fill_size]
                tmp_ks_sz = k_state_sz[:fill_size]
                tmp_vs_sz = v_state_sz[:fill_size]
                fill_end = fill_start + fill_size

                k_caches[boff, fill_start:fill_end] = tmp_ks
                v_caches[boff, fill_start:fill_end] = tmp_vs
                k_scales_zeros[boff, fill_start:fill_end] = tmp_ks_sz
                v_scales_zeros[boff, fill_start:fill_end] = tmp_vs_sz

                k_state = k_state[fill_size:]
                v_state = v_state[fill_size:]
                k_state_sz = k_state_sz[fill_size:]
                v_state_sz = v_state_sz[fill_size:]

                block_id += 1
                fill_start = 0
                fill_size = min(block_size, k_state.size(0))
                if fill_size == 0:
                    break

        yield k_caches, v_caches, k_scales_zeros, v_scales_zeros

    @pytest.mark.parametrize('head_dim', [128, 96], indirect=True)
    @pytest.mark.parametrize(['seq_lens', 'history_lens'], [
        ((1, 1, 1, 1), (1, 16, 31, 24)),
        ((1, 8, 16, 24), (1, 16, 31, 24)),
    ],
                             indirect=True)
    def test_fill_kv_cache(self, k_states, v_states, k_caches, v_caches, k_scales_zeros, v_scales_zeros, block_offsets,
                           q_start_loc, q_seq_length, kv_seq_length, max_q_seq_length, gt):
        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache
        fill_kv_cache(k_states, v_states, k_caches, v_caches, q_start_loc, q_seq_length, kv_seq_length,
                      max_q_seq_length, block_offsets, k_scales_zeros, v_scales_zeros, 8)

        torch.testing.assert_close(k_caches / 256, gt[0] / 256, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(v_caches / 256, gt[1] / 256, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_scales_zeros, gt[2])
        torch.testing.assert_close(v_scales_zeros, gt[3])


class TestFillKVCacheInt4(TestFillKVCacheInt8):

    @pytest.fixture
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim):
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim // 2)
        yield torch.full(shape, 0, dtype=torch.uint8).cuda()

    @pytest.fixture
    def nbits(self):
        yield 4

    @pytest.mark.parametrize('head_dim', [128], indirect=True)
    @pytest.mark.parametrize(['seq_lens', 'history_lens'], [
        ((1, 1, 1, 1), (1, 16, 31, 24)),
        ((1, 8, 16, 24), (1, 16, 31, 24)),
    ],
                             indirect=True)
    def test_fill_kv_cache(self, k_states, v_states, k_caches, v_caches, k_scales_zeros, v_scales_zeros, block_offsets,
                           q_start_loc, q_seq_length, kv_seq_length, max_q_seq_length, gt, nbits):
        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache
        k_scales_zeros = torch.zeros_like(k_scales_zeros)
        v_scales_zeros = torch.zeros_like(v_scales_zeros)
        fill_kv_cache(k_states, v_states, k_caches, v_caches, q_start_loc, q_seq_length, kv_seq_length,
                      max_q_seq_length, block_offsets, k_scales_zeros, v_scales_zeros, nbits)

        torch.testing.assert_close(k_scales_zeros, gt[2])
        torch.testing.assert_close(v_scales_zeros, gt[3])
        torch.testing.assert_close(k_caches, gt[0])
        torch.testing.assert_close(v_caches, gt[1])


class TestFillKVCacheInt42(TestFillKVCacheInt4):
    """quant_policy == 42:

    - K: QJL4 = 3bit MSE + 1bit QJL
    - V: TurboQuant MSE int2
    """

    @pytest.fixture
    def head_dim(self, request):
        yield request.param

    @pytest.fixture
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim):
        # K raw dim = head_dim, packed dim = head_dim // 2
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim // 2)
        yield torch.full(shape, 0, dtype=torch.uint8).cuda()

    @pytest.fixture
    def v_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim):
        # V TurboQuant MSE int2 packed: raw dim = head_dim, packed dim = head_dim // 4
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim // 4)
        yield torch.full(shape, 0, dtype=torch.uint8).cuda()

    @pytest.fixture
    def k_scales_zeros(self, batch_size, max_num_blocks, block_size, num_heads):
        # K meta: [mse_norm, qjl_norm]
        shape = (batch_size * max_num_blocks, block_size, num_heads, 2)
        yield torch.full(shape, 0.0).cuda()

    @pytest.fixture
    def v_scales_zeros(self, batch_size, max_num_blocks, block_size, num_heads):
        # V TurboQuant MSE int2: [norm]
        shape = (batch_size * max_num_blocks, block_size, num_heads, 1)
        yield torch.full(shape, 0.0).cuda()

    @pytest.fixture
    def gt(self, k_states, v_states, k_caches, v_caches, seq_lens, history_lens, block_offsets, block_size,
           k_scales_zeros, v_scales_zeros):
        k_states_q, k_meta = quant_turboquant_qjl4(k_states)
        v_states_q, v_norm = quant_turboquant_mse(v_states, 2)
        v_meta = v_norm.unsqueeze(-1)

        batch_size = len(seq_lens)
        k_caches = k_caches.clone()
        v_caches = v_caches.clone()
        k_scales_zeros = k_scales_zeros.clone()
        v_scales_zeros = v_scales_zeros.clone()

        splited_k_states = k_states_q.split(seq_lens)
        splited_v_states = v_states_q.split(seq_lens)
        splited_k_meta = k_meta.split(seq_lens)
        splited_v_meta = v_meta.split(seq_lens)

        for bidx in range(batch_size):
            k_state = splited_k_states[bidx]
            v_state = splited_v_states[bidx]
            k_state_meta = splited_k_meta[bidx]
            v_state_meta = splited_v_meta[bidx]

            h_len = history_lens[bidx]
            b_offs = block_offsets[bidx]
            block_id = _div_up(h_len + 1, block_size) - 1
            fill_start = h_len % block_size
            fill_size = min(block_size - fill_start, k_state.size(0))

            while True:
                boff = b_offs[block_id]
                fill_end = fill_start + fill_size

                k_caches[boff, fill_start:fill_end] = k_state[:fill_size]
                v_caches[boff, fill_start:fill_end] = v_state[:fill_size]
                k_scales_zeros[boff, fill_start:fill_end] = k_state_meta[:fill_size]
                v_scales_zeros[boff, fill_start:fill_end] = v_state_meta[:fill_size]

                k_state = k_state[fill_size:]
                v_state = v_state[fill_size:]
                k_state_meta = k_state_meta[fill_size:]
                v_state_meta = v_state_meta[fill_size:]

                block_id += 1
                fill_start = 0
                fill_size = min(block_size, k_state.size(0))
                if fill_size == 0:
                    break

        yield k_caches, v_caches, k_scales_zeros, v_scales_zeros

    @pytest.mark.parametrize('head_dim', [128], indirect=True)
    @pytest.mark.parametrize(['seq_lens', 'history_lens'], [
        ((1, 1, 1, 1), (1, 16, 31, 24)),
        ((1, 8, 16, 24), (1, 16, 31, 24)),
    ],
                             indirect=True)
    def test_fill_kv_cache(self, k_states, v_states, k_caches, v_caches, k_scales_zeros, v_scales_zeros, block_offsets,
                           q_start_loc, q_seq_length, kv_seq_length, max_q_seq_length, gt):
        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache

        fill_kv_cache(
            k_states,
            v_states,
            k_caches,
            v_caches,
            q_start_loc,
            q_seq_length,
            kv_seq_length,
            max_q_seq_length,
            block_offsets,
            k_scales_zeros,
            v_scales_zeros,
            42,
        )

        torch.testing.assert_close(k_caches, gt[0])
        torch.testing.assert_close(v_caches, gt[1])
        torch.testing.assert_close(k_scales_zeros, gt[2], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v_scales_zeros, gt[3], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize('head_dim', [128], indirect=True)
    def test_qjl4_reference_sanity(self, head_dim):
        torch.manual_seed(42)
        x = torch.randn(64, 4, head_dim).cuda()
        q, meta = quant_turboquant_qjl4(x)
        rec = dequantize_turboquant_qjl4(q, meta)

        x_flat = x.flatten(0, -2)
        rec_flat = rec.flatten(0, -2)
        x_norm = x_flat / (x_flat.norm(dim=-1, keepdim=True) + 1e-10)
        rec_norm = rec_flat / (rec_flat.norm(dim=-1, keepdim=True) + 1e-10)
        cos = (x_norm * rec_norm).sum(dim=-1).mean().item()
        assert cos > 0.80, f'QJL4 reference cosine too low: {cos}'

    def test_fill_kv_cache_quant42_vs_python_reference(self):
        """Test fill_kv_cache with quant_policy=42 against Python reference.

        This test verifies that the fill_kv_cache kernel produces the same quantized output as the Python reference
        implementation.

        From debug.py: compares runtime fill_kv_cache output with Python reference quantization for the written tokens.
        """
        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import (
            _get_lloyd_max_codebook,
            butterfly_rotate,
            fill_kv_cache,
        )

        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)

        device = 'cuda'
        dtype = torch.float16

        batch = 1
        q_len = 1
        hist_len = 8
        kv_len = hist_len + q_len
        num_heads = 2
        k_dim = 64
        v_dim = 64
        block_size = 16

        # Generate test data
        k = torch.rand(batch, kv_len, num_heads, k_dim, dtype=dtype, device=device)
        v = torch.rand(batch, kv_len, num_heads, v_dim, dtype=dtype, device=device)

        seq_lens = torch.tensor([q_len], device=device)
        kv_seqlens = torch.tensor([kv_len], device=device)
        q_start_loc = torch.tensor([0], device=device)

        # Create block offsets
        num_blocks = (kv_seqlens + block_size - 1) // block_size
        block_offsets = torch.arange(num_blocks[0], device=device).unsqueeze(0)

        packed_k_dim = k_dim // 2
        packed_v_dim = v_dim // 4
        max_blocks = num_blocks[0].item() + 1

        # Initialize blocked caches
        blocked_k = torch.zeros(max_blocks, block_size, num_heads, packed_k_dim, dtype=torch.uint8, device=device)
        blocked_v = torch.zeros(max_blocks, block_size, num_heads, packed_v_dim, dtype=torch.uint8, device=device)
        blocked_ksz = torch.zeros(max_blocks, block_size, num_heads, 2, dtype=dtype, device=device)
        blocked_vsz = torch.zeros(max_blocks, block_size, num_heads, 1, dtype=dtype, device=device)

        # Get the token to write (last position)
        conti_k = k[:, hist_len:hist_len + q_len].reshape(-1, num_heads, k_dim)
        conti_v = v[:, hist_len:hist_len + q_len].reshape(-1, num_heads, v_dim)

        # Run fill_kv_cache
        fill_kv_cache(
            conti_k,
            conti_v,
            blocked_k,
            blocked_v,
            q_start_loc,
            seq_lens,
            kv_seqlens,
            q_len,
            block_offsets,
            k_scales_zeros=blocked_ksz,
            v_scales_zeros=blocked_vsz,
            quant_policy=42,
        )

        # Python reference quantization - only for the last token (the one being written)
        last_k = k[0, hist_len:hist_len + q_len]  # (heads, dim)
        last_v = v[0, hist_len:hist_len + q_len]

        # Quantize K using QJL4 - only for last token
        head_dim = k_dim
        centroids, boundaries = _get_lloyd_max_codebook(head_dim, 3, device=device)
        mse_norm = last_k.float().norm(dim=-1, keepdim=True)
        kv_unit = last_k.float() / (mse_norm + 1e-10)
        y = butterfly_rotate(kv_unit)
        idx3 = torch.searchsorted(boundaries, y.contiguous()).clamp(0, 7).long()
        c = centroids[idx3]
        residual = y - c
        qjl_bit = (residual >= 0).long()
        qjl_norm = residual.norm(dim=-1, keepdim=True) / math.sqrt(head_dim)
        nibble = idx3 | (qjl_bit << 3)
        q1, q2 = nibble.split(nibble.shape[-1] // 2, dim=-1)
        ref_k_q = (q1 + (q2 << 4)).to(torch.uint8)
        ref_k_meta = torch.cat([mse_norm, qjl_norm], dim=-1)

        # Quantize V using MSE int2 - only for last token
        _, boundaries_v = _get_lloyd_max_codebook(v_dim, 2, device=device)
        v_norms = last_v.float().norm(dim=-1, keepdim=True)
        v_unit = last_v.float() / (v_norms + 1e-10)
        y_v = butterfly_rotate(v_unit)
        indices_v = torch.searchsorted(boundaries_v, y_v.contiguous()).clamp(0, 3)
        q1, q2, q3, q4 = indices_v.split(indices_v.shape[-1] // 4, dim=-1)
        ref_v_q = (q1 + q2 * 4 + q3 * 16 + q4 * 64).to(torch.uint8)
        ref_v_norm = v_norms.squeeze(-1)

        # Compare the last token (the one we wrote)
        runtime_k_last = blocked_k[0, hist_len:hist_len + 1]
        runtime_v_last = blocked_v[0, hist_len:hist_len + 1]
        runtime_k_meta_last = blocked_ksz[0, hist_len:hist_len + 1]
        runtime_v_meta_last = blocked_vsz[0, hist_len:hist_len + 1, :, 0]

        # Reference is already for the last token only
        ref_k_last = ref_k_q
        ref_v_last = ref_v_q
        ref_v_meta_last = ref_v_norm

        # Verify K packed data
        torch.testing.assert_close(runtime_k_last, ref_k_last,
                                   msg='K packed last-token runtime vs python mismatch')
        # Verify V packed data
        torch.testing.assert_close(runtime_v_last, ref_v_last,
                                   msg='V packed last-token runtime vs python mismatch')
        # Verify K meta (larger tolerance due to FP16 precision differences)
        # Use only absolute tolerance to avoid issues with small relative values
        torch.testing.assert_close(runtime_k_meta_last.float(), ref_k_meta.float(), atol=0.01, rtol=0,
                                   msg='K meta last-token runtime vs python mismatch')
        # Verify V meta
        torch.testing.assert_close(runtime_v_meta_last.float(), ref_v_meta_last.float(), atol=0.01, rtol=0,
                                   msg='V meta last-token runtime vs python mismatch')

        print('fill_kv_cache quant42 vs Python reference: all checks passed')


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestFillKVCacheBlockedFP8(TestFillKVCache):

    @pytest.fixture(autouse=True, scope='class')
    def initialize(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        yield

    @pytest.fixture
    def scale_fmt(self, request):
        yield request.param

    @pytest.fixture
    def quant_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def num_heads(self):
        yield 4

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture
    def block_size(self):
        yield 64

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def cu_seqlen_q(self, q_start_loc, q_seq_length):
        batch_size = q_start_loc.size(0)
        cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32).cuda()
        cu_seqlen[1:] = q_start_loc + q_seq_length
        return cu_seqlen

    @pytest.fixture
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim, quant_dtype):
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim)
        yield torch.full(shape, 0, dtype=quant_dtype).cuda()

    @pytest.fixture
    def v_caches(self, k_caches):
        yield torch.zeros_like(k_caches)

    @pytest.fixture
    def ks_caches(self, batch_size, max_num_blocks, block_size, num_heads, head_dim, group_size):
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim // group_size)
        yield torch.full(shape, 0.0).cuda()

    @pytest.fixture
    def vs_caches(self, ks_caches):
        yield torch.ones_like(ks_caches)

    @pytest.fixture
    def gt(self, k_states, v_states, group_size, quant_dtype, scale_fmt):
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
        batch_size = k_states.size(0)
        num_heads = k_states.size(1)
        head_dim = k_states.size(2)

        k_states = k_states.flatten(0, -2)
        v_states = v_states.flatten(0, -2)

        quant_k, quant_ks = quant_fp8(k_states, group_size=group_size, dtype=quant_dtype, scale_fmt=scale_fmt)
        quant_v, quant_vs = quant_fp8(v_states, group_size=group_size, dtype=quant_dtype, scale_fmt=scale_fmt)

        quant_k = quant_k.view(batch_size, num_heads, head_dim)
        quant_ks = quant_ks.view(batch_size, num_heads, head_dim // group_size)
        quant_v = quant_v.view(batch_size, num_heads, head_dim)
        quant_vs = quant_vs.view(batch_size, num_heads, head_dim // group_size)

        yield quant_k, quant_ks, quant_v, quant_vs

    def uncache(self, k_caches, ks_caches, v_caches, vs_caches, cu_seqlen_q, kv_seqlens, block_offsets):
        batch_size = block_offsets.size(0)
        out_k = []
        out_ks = []
        out_v = []
        out_vs = []

        q_seqlens = cu_seqlen_q[1:] - cu_seqlen_q[:-1]

        for bidx in range(batch_size):
            seqlen = q_seqlens[bidx].item()
            kv_len = kv_seqlens[bidx].item()
            start = kv_len - seqlen
            end = kv_len

            k = k_caches[block_offsets[bidx]].reshape(-1, k_caches.size(-2), k_caches.size(-1))
            ks = ks_caches[block_offsets[bidx]].reshape(-1, ks_caches.size(-2), ks_caches.size(-1))
            v = v_caches[block_offsets[bidx]].reshape(-1, v_caches.size(-2), v_caches.size(-1))
            vs = vs_caches[block_offsets[bidx]].reshape(-1, vs_caches.size(-2), vs_caches.size(-1))

            out_k.append(k[start:end])
            out_ks.append(ks[start:end])
            out_v.append(v[start:end])
            out_vs.append(vs[start:end])

        out_k = torch.cat(out_k, dim=0)
        out_ks = torch.cat(out_ks, dim=0)
        out_v = torch.cat(out_v, dim=0)
        out_vs = torch.cat(out_vs, dim=0)
        return out_k, out_ks, out_v, out_vs

    @pytest.mark.parametrize('scale_fmt', [None, 'ue8m0'], indirect=True)
    @pytest.mark.parametrize(['seq_lens', 'history_lens'], [
        ((1, 1, 1, 1), (1, 128, 256, 200)),
        ((1, 64, 128, 50), (1, 128, 256, 200)),
    ],
                             indirect=True)
    def test_fill_kv_cache(self, k_states, v_states, k_caches, v_caches, ks_caches, vs_caches, block_offsets,
                           cu_seqlen_q, kv_seq_length, max_q_seq_length, gt, group_size, scale_fmt):
        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8

        fill_kv_cache_blocked_fp8(k_states,
                                  v_states,
                                  k_caches,
                                  v_caches,
                                  ks_caches,
                                  vs_caches,
                                  cu_seqlen_q,
                                  kv_seq_length,
                                  max_q_seq_length,
                                  block_offsets=block_offsets,
                                  group_size=group_size,
                                  scale_fmt=scale_fmt)

        gt_k, gt_ks, gt_v, gt_vs = gt

        # uncache
        out_k, out_ks, out_v, out_vs = self.uncache(k_caches, ks_caches, v_caches, vs_caches, cu_seqlen_q,
                                                    kv_seq_length, block_offsets)

        out_k = out_k / out_k.max()
        gt_k = gt_k.float()
        gt_k = gt_k / gt_k.max()

        out_v = out_v.float()
        out_v = out_v / out_v.max()
        gt_v = gt_v.float()
        gt_v = gt_v / gt_v.max()

        torch.testing.assert_close(out_k, gt_k)
        torch.testing.assert_close(out_ks, gt_ks)
        torch.testing.assert_close(out_v, gt_v)
        torch.testing.assert_close(out_vs, gt_vs)
