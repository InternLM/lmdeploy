# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch


@dataclass
class CompressedCacheSlot:
    """Per-sequence compressed cache state for a single layer."""

    window_kv: torch.Tensor | None = None
    compressed_kv: torch.Tensor | None = None
    index_kv: torch.Tensor | None = None
    kv_state: torch.Tensor | None = None
    score_state: torch.Tensor | None = None


class CompressedCacheEngine:
    """Dynamic compressed cache storage for DeepSeek-V4.

    This cache is intentionally separate from lmdeploy's paged KV / state cache. It stores per-sequence history whose
    effective length depends on each layer's compression ratio and current decoded context length.
    """

    def __init__(self,
                 window_size: int,
                 compress_ratio: int,
                 head_dim: int,
                 overlap: bool = False,
                 state_dim: int | None = None):
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.state_dim = state_dim if state_dim is not None else head_dim
        self.overlap = overlap
        self._slots: dict[int, CompressedCacheSlot] = {}

    def get_slot(self, slot: int) -> CompressedCacheSlot:
        """Get or create slot."""
        if slot not in self._slots:
            self._slots[slot] = CompressedCacheSlot()
        return self._slots[slot]

    def reset_slot(self, slot: int):
        """Clear slot."""
        self._slots.pop(slot, None)

    def get_window(self, slot: int) -> torch.Tensor | None:
        """Get window kv."""
        return self.get_slot(slot).window_kv

    def get_compressed(self, slot: int) -> torch.Tensor | None:
        """Get compressed kv."""
        return self.get_slot(slot).compressed_kv

    def get_index(self, slot: int) -> torch.Tensor | None:
        """Get index kv."""
        return self.get_slot(slot).index_kv

    def set_window(self, slot: int, kv: torch.Tensor):
        """Replace window kv."""
        if kv is None:
            self.get_slot(slot).window_kv = None
            return
        if kv.size(0) > self.window_size:
            cutoff = kv.size(0) % self.window_size
            ring = kv.new_empty(self.window_size, kv.size(-1))
            ring[cutoff:self.window_size], ring[:cutoff] = kv[-self.window_size:].split([self.window_size - cutoff,
                                                                                         cutoff],
                                                                                        dim=0)
            kv = ring
        self.get_slot(slot).window_kv = kv

    def append_window(self, slot: int, kv: torch.Tensor):
        """Append to the sliding-window kv cache."""
        if kv is None or kv.numel() == 0:
            return
        cache = self.get_slot(slot).window_kv
        cache = kv if cache is None else torch.cat([cache, kv], dim=0)
        if cache.size(0) > self.window_size:
            cache = cache[-self.window_size:]
        self.get_slot(slot).window_kv = cache

    def update_window(self, slot: int, pos: int, kv: torch.Tensor):
        """Update a single position in the sliding-window kv cache."""
        if kv is None or kv.numel() == 0:
            return
        cache_slot = self.get_slot(slot)
        cache = cache_slot.window_kv
        if cache is None:
            cache = kv.new_zeros(self.window_size, kv.size(-1))
        elif cache.size(0) < self.window_size and pos >= cache.size(0):
            new_cache = kv.new_zeros(self.window_size, kv.size(-1))
            new_cache[:cache.size(0)] = cache
            cache = new_cache
        cache[pos] = kv
        cache_slot.window_kv = cache

    def append_compressed(self, slot: int, kv: torch.Tensor):
        """Append compressed kv entries."""
        if kv is None or kv.numel() == 0:
            return
        cache = self.get_slot(slot).compressed_kv
        cache = kv if cache is None else torch.cat([cache, kv], dim=0)
        self.get_slot(slot).compressed_kv = cache

    def append_index(self, slot: int, kv: torch.Tensor):
        """Append index kv entries."""
        if kv is None or kv.numel() == 0:
            return
        cache = self.get_slot(slot).index_kv
        cache = kv if cache is None else torch.cat([cache, kv], dim=0)
        self.get_slot(slot).index_kv = cache

    def ensure_states(self, slot: int, num_rows: int, device: torch.device, dtype: torch.dtype):
        """Ensure fixed-size scratch state exists for the slot."""
        cache_slot = self.get_slot(slot)
        if cache_slot.kv_state is None or cache_slot.kv_state.size(0) != num_rows:
            cache_slot.kv_state = torch.zeros((num_rows, self.state_dim), dtype=torch.float32, device=device)
        if cache_slot.score_state is None or cache_slot.score_state.size(0) != num_rows:
            cache_slot.score_state = torch.full((num_rows, self.state_dim), float('-inf'), dtype=torch.float32,
                                                device=device)

    def get_states(self, slot: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get compressor scratch states."""
        cache_slot = self.get_slot(slot)
        return cache_slot.kv_state, cache_slot.score_state
