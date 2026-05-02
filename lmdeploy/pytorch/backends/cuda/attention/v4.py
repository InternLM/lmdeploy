# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata


class TritonV4AttentionImpl:
    """DeepSeek V4 attention using batched FlashMLA sparse decode."""

    def __init__(self, head_size: int, scale: float, window_size: int, compress_ratio: int):
        self.head_size = head_size
        self.scale = scale
        self.window_size = window_size
        import flash_mla
        self.flash_mla = flash_mla

    @staticmethod
    def _pad_query_heads(query: torch.Tensor, attn_sink: torch.Tensor):
        num_heads = query.size(2)
        if num_heads in (64, 128):
            return query, attn_sink, num_heads
        if num_heads < 64:
            padded_heads = 64
        elif num_heads < 128:
            padded_heads = 128
        else:
            raise RuntimeError(f'Unsupported h_q for FlashMLA sparse decode: {num_heads}')

        pad_heads = padded_heads - num_heads
        query = torch.nn.functional.pad(query, (0, 0, 0, pad_heads))
        attn_sink = torch.nn.functional.pad(attn_sink, (0, pad_heads))
        return query, attn_sink, num_heads

    @staticmethod
    def _pad_sparse_indices(indices: torch.Tensor | None, block: int = 128):
        if indices is None:
            return None
        topk = indices.size(-1)
        padded_topk = ((topk + block - 1) // block) * block
        if padded_topk == topk:
            return indices
        pad = padded_topk - topk
        return torch.nn.functional.pad(indices, (0, pad), value=-1)

    def forward_decode(self,
                       query: torch.Tensor,
                       window_kv_fp8_state: torch.Tensor,
                       attn_sink: torch.Tensor,
                       attn_metadata: V4AttentionMetadata,
                       block_size: int,
                       compressed_kv_fp8_cache: torch.Tensor | None = None):
        bsz = query.size(0)

        extra_k_cache = window_kv_fp8_state.view(bsz, self.window_size, 1, -1)

        extra_indices = attn_metadata.extra_indices_in_kvcache
        batch_offsets = torch.arange(bsz, device=extra_indices.device, dtype=torch.int32).view(-1, 1, 1) * self.window_size
        extra_indices = torch.where(extra_indices >= 0, extra_indices + batch_offsets, extra_indices)

        padded_indices = self._pad_sparse_indices(attn_metadata.indices_in_kvcache)
        padded_extra_indices = self._pad_sparse_indices(extra_indices)
        padded_query, padded_sink, original_heads = self._pad_query_heads(query, attn_sink)

        k_cache = compressed_kv_fp8_cache.unsqueeze(2) if compressed_kv_fp8_cache is not None else extra_k_cache

        sched_meta, _ = self.flash_mla.get_mla_metadata()

        output, _ = self.flash_mla.flash_mla_with_kvcache(
            padded_query,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=self.head_size,
            tile_scheduler_metadata=sched_meta,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=padded_indices.to(torch.int32),
            attn_sink=padded_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=padded_extra_indices.to(torch.int32),
            topk_length=attn_metadata.topk_length.to(torch.int32),
            extra_topk_length=attn_metadata.extra_topk_length.to(torch.int32),
        )
        return output[:, :, :original_heads]

    def forward_prefill(self,
                        query: torch.Tensor,
                        flat_kv: torch.Tensor,
                        attn_sink: torch.Tensor,
                        topk_indices: torch.Tensor):
        """Prefill attention using ``flash_mla_sparse_fwd``."""
        topk_indices = self._pad_sparse_indices(topk_indices).to(torch.int32)

        num_heads = query.size(1)
        target = 64 if num_heads < 64 else (128 if num_heads < 128 else num_heads)
        if target != num_heads:
            pad = target - num_heads
            query = torch.nn.functional.pad(query, (0, 0, 0, pad))
            attn_sink = torch.nn.functional.pad(attn_sink, (0, pad))

        out = self.flash_mla.flash_mla_sparse_fwd(
            query, flat_kv, topk_indices,
            sm_scale=self.scale, attn_sink=attn_sink)
        return out[0][:, :num_heads]


class TritonV4AttentionBuilder:
    """Builder for DeepSeek V4 sparse attention."""

    @staticmethod
    def build(head_size: int,
              scale: float,
              window_size: int,
              compress_ratio: int,
              **kwargs) -> TritonV4AttentionImpl:
        return TritonV4AttentionImpl(head_size=head_size,
                                     scale=scale,
                                     window_size=window_size,
                                     compress_ratio=compress_ratio)
