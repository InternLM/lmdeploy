
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import (
    _get_lloyd_max_codebook,
    _get_rotation_matrix,
)


def _div_up(a, b):
    return (a + b - 1) // b


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

    # Rescale by original norms
    x_hat = x_hat * norms.unsqueeze(-1)

    return x_hat


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

        out_k = out_k.float()
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
