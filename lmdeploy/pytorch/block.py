# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from dataclasses import dataclass

import numpy as np


class LogicalTokenBlock:
    """Logical block used to count tokens per block."""

    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size

        self.num_tokens = 0

    def get_num_empty_slots(self):
        """get num empty slots."""
        return self.block_size - self.num_tokens

    def is_empty(self):
        """is empty."""
        return self.num_tokens == 0

    def is_full(self):
        """is full."""
        return self.num_tokens == self.block_size

    def append_tokens(self, num_tokens: int = 1):
        """append tokens."""
        assert num_tokens <= self.get_num_empty_slots()
        self.num_tokens += num_tokens


def _div_up(x, n):
    """perform div up."""
    return (x + n - 1) // n


def _round_up(x, n):
    """perform round up."""
    return _div_up(x, n) * n


class LogicalTokenBlocks:
    """Logical blocks."""
    ALLOC_SIZE = 128

    def __init__(self, block_size: int):
        self._block_size = block_size
        reserve_size = _round_up(block_size, self.ALLOC_SIZE)
        self._blocks = np.zeros((reserve_size, ), dtype=np.int64)
        self._last_block_size = 0
        self._num_real = 0

    def reserve(self, size: int):
        """reserve cache size."""
        num_blocks = self._blocks.size
        if num_blocks >= size:
            return
        reserve_size = _round_up(size - num_blocks, self.ALLOC_SIZE)
        self._blocks = np.pad(self._blocks, (0, reserve_size))

    def __setitem__(self, *args, **kwargs):
        """set values."""
        return self.get_real_blocks().__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        """get values."""
        return self.get_real_blocks().__getitem__(*args, **kwargs)

    def get_real_blocks(self):
        """get logical blocks."""
        return self._blocks[:self._num_real]

    def append(self, blocks: np.ndarray):
        """append blocks."""
        num_blocks = len(blocks)
        self.reserve(num_blocks + self._num_real)
        slice_start = self._num_real
        slice_end = slice_start + num_blocks
        self._num_real += num_blocks
        self.__setitem__(slice(slice_start, slice_end), blocks)

    def num_required_blocks(self, num_tokens: int):
        """get num required blocks."""
        if self._last_block_size == 0:
            remain_tokens = num_tokens
        else:
            next_block_size = min(num_tokens,
                                  self._block_size - self._last_block_size)
            remain_tokens = num_tokens - next_block_size
        return _div_up(remain_tokens, self._block_size)

    def add_tokens(self, num_tokens: int):
        """add tokens."""
        total_tokens = self.num_tokens() + num_tokens
        self._last_block_size = total_tokens % self._block_size
        if self._last_block_size == 0:
            self._last_block_size = self._block_size

    def num_tokens(self):
        """get num tokens."""
        return max(
            0, self._num_real - 1) * self._block_size + self._last_block_size

    def __len__(self):
        """get length."""
        return self._num_real

    def reshape_by_tokens(self, num_tokens: int):
        """resize logical blocks by num tokens."""
        assert num_tokens <= self.num_tokens()
        self._num_real = _div_up(num_tokens, self._block_size)
        self._last_block_size = num_tokens % self._block_size
        if self._last_block_size == 0:
            self._last_block_size = self._block_size

    def reset(self):
        """reset."""
        self.reshape_by_tokens(0)

    def get_block_size(self):
        """get block size."""
        return self._block_size

    def last_block_size(self):
        """get last block size."""
        return self._last_block_size

    def clone(self):
        """clone logical blocks."""
        ret = LogicalTokenBlocks(self.get_block_size())
        ret.append(self[:])
        ret.add_tokens(self.num_tokens())
        return ret


@dataclass
class PhysicalTokenBlock:
    """Physical block used to schedule key value cache."""

    device: str
    block_id: int
    block_size: int
    ref_count: int = 0
