# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata


class TritonV4AttentionImpl:
    """DeepSeek V4 attention wrapper over the official sparse attention kernel.

    This implementation deliberately keeps history in named block caches and
    only materializes the working set required by one decode step. The actual
    sparse attention math still delegates to the official `sparse_attn` kernel.
    """

    def __init__(self, head_size: int, scale: float, window_size: int, compress_ratio: int, kernel_mod):
        self.head_size = head_size
        self.scale = scale
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.kernel_mod = kernel_mod

    @staticmethod
    def _gather_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, positions: torch.Tensor,
                              block_size: int):
        if positions.numel() == 0:
            return cache.new_empty((*positions.shape, cache.size(-1)))
        safe_positions = positions.clamp(min=0)
        block_idx = torch.div(safe_positions, block_size, rounding_mode='floor').long()
        max_block_idx = block_offsets.size(1)
        valid = (positions >= 0) & (block_idx < max_block_idx)
        safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
        block_off = torch.remainder(safe_positions, block_size).long()
        phys_blocks = block_offsets.gather(1, safe_block_idx).long()
        gathered = cache[phys_blocks, block_off]
        return torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()))

    def _get_compressed_scratch(self, decode_scratch: dict[str, torch.Tensor] | None, batch_size: int,
                                max_width: int, device: torch.device):
        if self.compress_ratio == 4:
            key = 'selected_compressed_kv_r4'
        else:
            key = 'selected_compressed_kv_r128'
        if decode_scratch is not None and key in decode_scratch:
            buffer = decode_scratch[key]
            if buffer.size(0) >= batch_size and buffer.size(1) == max_width:
                return buffer[:batch_size]
        return torch.empty((batch_size, max_width, self.head_size), dtype=torch.bfloat16, device=device)

    def _get_full_scratch(self, decode_scratch: dict[str, torch.Tensor] | None, batch_size: int, total_width: int,
                          device: torch.device):
        if self.compress_ratio == 4:
            key = 'selected_full_kv_r4'
        else:
            key = 'selected_full_kv_r128'
        if decode_scratch is not None and key in decode_scratch:
            buffer = decode_scratch[key]
            if buffer.size(0) >= batch_size and buffer.size(1) == total_width:
                return buffer[:batch_size]
        return torch.empty((batch_size, total_width, self.head_size), dtype=torch.bfloat16, device=device)

    def forward_decode(self,
                       query: torch.Tensor,
                       window_kv_state: torch.Tensor,
                       attn_sink: torch.Tensor,
                       attn_metadata: V4AttentionMetadata,
                       block_size: int,
                       compressed_kv_cache: torch.Tensor | None = None,
                       decode_scratch: dict[str, torch.Tensor] | None = None):
        block_offsets = attn_metadata.block_offsets.long()
        bsz = query.size(0)

        if decode_scratch is not None and 'selected_window_kv' in decode_scratch:
            window_kv = decode_scratch['selected_window_kv'][:bsz]
        else:
            window_kv = torch.empty((bsz, self.window_size, self.head_size), dtype=torch.bfloat16, device=query.device)
        safe_positions = attn_metadata.window_positions.clamp(min=0)
        gathered_window = window_kv_state.gather(1, safe_positions.unsqueeze(-1).expand(-1, -1, self.head_size))
        gathered_window = torch.where(attn_metadata.window_positions.unsqueeze(-1) >= 0, gathered_window,
                                      gathered_window.new_zeros(()))
        window_kv.copy_(gathered_window)

        full_kv = window_kv
        if compressed_kv_cache is not None and attn_metadata.compressed_positions is not None:
            max_width = attn_metadata.compressed_positions.size(1)
            compressed_kv = self._get_compressed_scratch(decode_scratch, bsz, max_width, query.device)
            compressed_kv.copy_(self._gather_cache_entries(compressed_kv_cache,
                                                           block_offsets,
                                                           attn_metadata.compressed_positions,
                                                           block_size))
            full_kv = self._get_full_scratch(decode_scratch, bsz, self.window_size + max_width, query.device)
            full_kv[:, :self.window_size] = window_kv
            full_kv[:, self.window_size:self.window_size + max_width] = compressed_kv

        return self.kernel_mod.sparse_attn(query, full_kv, attn_sink, attn_metadata.topk_indices.int(), self.scale)


class TritonV4AttentionBuilder:
    """Builder for DeepSeek V4 sparse attention."""

    @staticmethod
    def build(head_size: int,
              scale: float,
              window_size: int,
              compress_ratio: int,
              kernel_mod,
              **kwargs) -> TritonV4AttentionImpl:
        return TritonV4AttentionImpl(head_size=head_size,
                                     scale=scale,
                                     window_size=window_size,
                                     compress_ratio=compress_ratio,
                                     kernel_mod=kernel_mod)
