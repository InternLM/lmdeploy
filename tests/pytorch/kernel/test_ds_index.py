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
