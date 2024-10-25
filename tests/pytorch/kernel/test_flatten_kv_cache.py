import pytest
import torch


def _div_up(a, b):
    return (a + b - 1) // b


class TestFlattenKVCache:

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
    def k_caches(self, batch_size, max_num_blocks, block_size, num_heads,
                 head_dim):
        shape = (batch_size * max_num_blocks, block_size, num_heads, head_dim)
        yield torch.rand(shape).cuda()

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
    def gt(self, k_caches, v_caches, kv_lens, block_offsets, block_size,
           num_heads, out_size, head_dim):
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

    def test_flatten_kv_cache(self, k_caches, v_caches, kv_seqlens,
                              block_offsets, out_size, gt):
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import \
            flatten_kv_cache

        k_states, v_states = flatten_kv_cache(k_caches,
                                              v_caches,
                                              kv_seqlens,
                                              block_offsets,
                                              out_size=out_size)

        torch.testing.assert_close(k_states, gt[0])
        torch.testing.assert_close(v_states, gt[1])
