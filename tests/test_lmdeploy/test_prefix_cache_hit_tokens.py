import numpy as np

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.paging.block_trie import BlockTrie


class _FakeAllocator:

    def update_access_time(self, blocks):
        return None

    def add_ref_count(self, blocks, count):
        return None


class _FakeBlockManager:

    def __init__(self):
        self.allocator = _FakeAllocator()
        self.num_gpu_blocks = 16


class _FakeLogicalBlocks(list):

    def __init__(self, num_blocks: int = 8):
        super().__init__(list(range(num_blocks)))
        self.last_shared_node = None

    def append(self, blocks):
        self.extend(blocks)


class _FakeSeq:

    def __init__(self, tokens: list[int], block_size: int = 4):
        self.adapter_name = 'default'
        self.logical_blocks = _FakeLogicalBlocks()
        self.history_cache = np.array(tokens, dtype=np.int64)
        self.prefix_cache_hit_tokens = 0
        self._num_valid_ids = len(tokens)
        self._block_size = block_size

    @property
    def num_valid_ids(self):
        return self._num_valid_ids

    @property
    def num_all_ids(self):
        return self._num_valid_ids

    def set_step(self, step: int):
        self._step = step


def test_block_trie_match_disabled_sets_zero():
    cache_config = CacheConfig(
        max_batches=4,
        block_size=4,
        num_cpu_blocks=0,
        num_gpu_blocks=16,
        enable_prefix_caching=False,
    )
    trie = BlockTrie(cache_config, _FakeBlockManager())
    seq = _FakeSeq(list(range(8)))
    trie.match(seq)
    assert seq.prefix_cache_hit_tokens == 0
