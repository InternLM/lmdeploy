# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.attention import V4AttentionMetadata


class V4Attention(nn.Module):
    """DeepSeek V4 cache-aware attention wrapper."""

    def __init__(self, head_size: int, scale: float, window_size: int, compress_ratio: int, **kwargs):
        super().__init__()
        layer_backend = get_backend()
        impl_builder = layer_backend.get_layer_impl_builder(OpType.V4Attention)
        self.impl = impl_builder.build(head_size=head_size,
                                       scale=scale,
                                       window_size=window_size,
                                       compress_ratio=compress_ratio,
                                       **kwargs)

    def forward_decode(self,
                       query: torch.Tensor,
                       window_kv_fp8_state: torch.Tensor,
                       attn_sink: torch.Tensor,
                       attn_metadata: V4AttentionMetadata,
                       block_size: int,
                       compressed_kv_fp8_cache: torch.Tensor | None = None):
        return self.impl.forward_decode(query,
                                        window_kv_fp8_state,
                                        attn_sink,
                                        attn_metadata,
                                        block_size,
                                        compressed_kv_fp8_cache=compressed_kv_fp8_cache)

    def forward_prefill(self,
                        query: torch.Tensor,
                        flat_kv: torch.Tensor,
                        attn_sink: torch.Tensor,
                        topk_indices: torch.Tensor):
        return self.impl.forward_prefill(query, flat_kv, attn_sink, topk_indices)
