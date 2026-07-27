import pytest
import torch


def _make_A(M, K, group_size, out_dtype, device):
    quant_A = torch.randn(M, K // group_size, group_size, dtype=torch.float32, device=device)
    # -1 ~ 1
    quant_A = quant_A * 2 - 1
    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
    quant_A *= scaling
    quant_A = quant_A.to(out_dtype).to(torch.float32)

    # create scale and A
    scale = torch.randn(M, K // group_size, dtype=torch.float32, device=device)
    scale /= fmax
    A = quant_A * scale[..., None]

    A = A.reshape(M, K)
    quant_A = quant_A.reshape(M, K).to(out_dtype)
    scale = scale.T.contiguous().T
    return A, quant_A, scale


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestDSIndex:

    @pytest.fixture
    def num_heads(self):
        yield 64

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture
    def block_size(self):
        yield 64

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def q_seqlens(self, request):
        yield request.param

    @pytest.fixture
    def kv_seqlens(self, request):
        yield request.param

    @pytest.fixture
    def k_seqlens(self, kv_seqlens, device):
        yield torch.tensor(kv_seqlens, dtype=torch.int32, device=device)

    @pytest.fixture
    def cu_seqlen_q(self, q_seqlens, device):
        yield torch.tensor([0] + list(q_seqlens), dtype=torch.int32, device=device).cumsum(0)

    @pytest.fixture
    def cu_seqlen_kv(self, kv_seqlens, device):
        yield torch.tensor([0] + list(kv_seqlens), dtype=torch.int32, device=device).cumsum(0)

    @pytest.fixture
    def query(self, q_seqlens, num_heads, head_dim, device):
        total_len = sum(q_seqlens)
        fp_q, q, q_s = _make_A(total_len * num_heads, head_dim, head_dim, out_dtype=torch.float8_e4m3fn, device=device)
        fp_q = fp_q.view(total_len, num_heads, head_dim)
        q = q.view(total_len, num_heads, head_dim)
        q_s = q_s.view(total_len, num_heads)
        yield fp_q, q, q_s

    @pytest.fixture
    def q(self, query):
        yield query[1]

    @pytest.fixture
    def q_s(self, query):
        yield query[2]

    @pytest.fixture
    def key(self, kv_seqlens, head_dim):
        total_len = sum(kv_seqlens)
        fp_k, k, k_s = _make_A(total_len, head_dim, head_dim, out_dtype=torch.float8_e4m3fn, device='cuda')
        fp_k = fp_k.view(total_len, head_dim)
        k = k.view(total_len, head_dim)
        k_s = k_s.view(total_len)
        yield fp_k, k, k_s

    @pytest.fixture
    def k(self, key):
        yield key[1]

    @pytest.fixture
    def k_s(self, key):
        yield key[2]

    @pytest.fixture
    def cache_key(self, k, k_s, kv_seqlens, block_size, head_dim):
        batch_size = len(kv_seqlens)
        max_num_blocks = (max(kv_seqlens) + block_size - 1) // block_size

        # get block offsets
        batch_ids = torch.arange(batch_size, device='cuda') * max_num_blocks
        block_ids = torch.arange(max_num_blocks, device='cuda')
        block_offsets = (batch_ids[:, None] + block_ids[None, :])

        k_cache = torch.zeros((max_num_blocks * batch_size * block_size, head_dim),
                              dtype=torch.float8_e4m3fn,
                              device='cuda')
        k_s_cache = torch.zeros((max_num_blocks * batch_size * block_size), dtype=torch.float32, device='cuda')

        k = k.split(kv_seqlens, dim=0)
        k_s = k_s.split(kv_seqlens, dim=0)
        for i in range(batch_size):
            size = k[i].size(0)
            start = i * max_num_blocks * block_size
            end = start + size
            k_cache[start:end] = k[i]
            k_s_cache[start:end] = k_s[i]

        k_cache = k_cache.view(batch_size * max_num_blocks, block_size, head_dim)
        k_s_cache = k_s_cache.view(batch_size * max_num_blocks, block_size)

        yield k_cache, k_s_cache, block_offsets

    @pytest.fixture
    def k_cache(self, cache_key):
        yield cache_key[0]

    @pytest.fixture
    def k_s_cache(self, cache_key):
        yield cache_key[1]

    @pytest.fixture
    def block_offset(self, cache_key):
        yield cache_key[2]

    @pytest.mark.parametrize('q_seqlens', [(1, 1, 1, 1), (1024, 2048, 1024, 1)], indirect=True)
    @pytest.mark.parametrize('kv_seqlens', [(2048, 4096, 1024, 128)], indirect=True)
    def test_fp8_index(self, q, q_s, k_cache, k_s_cache, cu_seqlen_q, k_seqlens, block_offset):
        # gt requires tilelang, so this test just ensure the kernel works
        from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index
        fp8_index(q, q_s, k_cache, k_s_cache, cu_seqlen_q, k_seqlens, block_offset)

    def test_fp8_index_trim_causal_tail_with_raw_lengths(self, num_heads, head_dim, block_size, device):
        """Trimmed V4-style causal scoring must preserve every visible
        score."""
        from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index

        q_seqlens = (17, 9)
        raw_kv_seqlens = (65, 33)
        compress_ratio = 4
        k_seqlens_host = tuple(raw_len // compress_ratio for raw_len in raw_kv_seqlens)
        batch_size = len(q_seqlens)
        total_q = sum(q_seqlens)
        total_k = sum(k_seqlens_host)
        max_num_blocks = (max(k_seqlens_host) + block_size - 1) // block_size

        _, q, q_s = _make_A(total_q * num_heads, head_dim, head_dim,
                            out_dtype=torch.float8_e4m3fn, device=device)
        q = q.view(total_q, num_heads, head_dim)
        q_s = q_s.view(total_q, num_heads)
        _, k, k_s = _make_A(total_k, head_dim, head_dim,
                            out_dtype=torch.float8_e4m3fn, device=device)
        k_s = k_s.view(total_k)

        k_cache = torch.zeros((batch_size * max_num_blocks, block_size, head_dim),
                              dtype=torch.float8_e4m3fn, device=device)
        k_s_cache = torch.zeros((batch_size * max_num_blocks, block_size),
                                dtype=torch.float32, device=device)
        block_ids = torch.arange(max_num_blocks, device=device)
        batch_ids = torch.arange(batch_size, device=device) * max_num_blocks
        block_offset = batch_ids[:, None] + block_ids[None, :]

        k_start = 0
        for batch, k_len in enumerate(k_seqlens_host):
            cache_start = batch * max_num_blocks * block_size
            cache_end = cache_start + k_len
            k_cache.view(-1, head_dim)[cache_start:cache_end] = k[k_start:k_start + k_len]
            k_s_cache.view(-1)[cache_start:cache_end] = k_s[k_start:k_start + k_len]
            k_start += k_len

        cu_seqlen_q = torch.tensor([0] + list(q_seqlens), dtype=torch.int32,
                                   device=device).cumsum(0)
        k_seqlens = torch.tensor(k_seqlens_host, dtype=torch.int32, device=device)
        raw_kv_seqlens_t = torch.tensor(raw_kv_seqlens, dtype=torch.int64, device=device)
        max_k_seqlen = max_num_blocks * block_size

        full_scores, row_k_seqlens = fp8_index(q, q_s, k_cache, k_s_cache,
                                               cu_seqlen_q, k_seqlens, block_offset,
                                               max_q_seqlen=max(q_seqlens),
                                               max_k_seqlen=max_k_seqlen,
                                               causal=True,
                                               raw_k_seqlens=raw_kv_seqlens_t,
                                               compress_ratio=compress_ratio,
                                               return_row_k_seqlens=True)
        trimmed_scores, trimmed_row_k_seqlens = fp8_index(q, q_s, k_cache, k_s_cache,
                                                          cu_seqlen_q, k_seqlens, block_offset,
                                                          max_q_seqlen=max(q_seqlens),
                                                          max_k_seqlen=max_k_seqlen,
                                                          causal=True,
                                                          raw_k_seqlens=raw_kv_seqlens_t,
                                                          compress_ratio=compress_ratio,
                                                          return_row_k_seqlens=True,
                                                          trim_causal_tail=True)

        expected_lens = []
        for q_len, raw_k_len, k_len in zip(q_seqlens, raw_kv_seqlens, k_seqlens_host):
            start_pos = raw_k_len - q_len
            expected_lens.extend(
                min(max((start_pos + q_pos + 1) // compress_ratio, 0), k_len)
                for q_pos in range(q_len))
        expected_lens = torch.tensor(expected_lens, dtype=torch.int32, device=device)

        assert torch.equal(row_k_seqlens, expected_lens)
        assert torch.equal(trimmed_row_k_seqlens, expected_lens)
        for row, row_len in enumerate(expected_lens.tolist()):
            if row_len > 0:
                torch.testing.assert_close(trimmed_scores[row, :row_len],
                                           full_scores[row, :row_len],
                                           rtol=0,
                                           atol=0)

        deep_gemm = pytest.importorskip('deep_gemm')
        from lmdeploy.pytorch.backends.cuda.v4_compressor import _get_v4_packed_index_cache_views
        from lmdeploy.pytorch.consts import v4_packed_index_cache_shape

        packed_cache = torch.zeros(batch_size * max_num_blocks,
                                   *v4_packed_index_cache_shape(block_size, head_dim),
                                   dtype=torch.uint8,
                                   device=device)
        packed_values, packed_scales = _get_v4_packed_index_cache_views(packed_cache, head_dim)
        packed_values.copy_(k_cache)
        packed_scales.squeeze(-1).copy_(k_s_cache)

        repeats = torch.tensor(q_seqlens, dtype=torch.int64, device=device)
        page_table = torch.repeat_interleave(block_offset.to(torch.int32), repeats,
                                             dim=0, output_size=total_q)
        context_lens = expected_lens.unsqueeze(-1)
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, block_size, deep_gemm.get_num_sms())
        deepgemm_scores = deep_gemm.fp8_paged_mqa_logits(
            q.view(total_q, 1, num_heads, head_dim),
            packed_cache,
            q_s,
            context_lens,
            page_table,
            schedule,
            max_k_seqlen,
            False)
        for row, row_len in enumerate(expected_lens.tolist()):
            if row_len > 0:
                torch.testing.assert_close(deepgemm_scores[row, :row_len],
                                           trimmed_scores[row, :row_len],
                                           rtol=1e-3,
                                           atol=1e-3)
