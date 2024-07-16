# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from .base import AttentionMetadata


class Attention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
        **kwargs,
    ):
        from .selector import get_attn_backend
        super().__init__()
        attn_backend = get_attn_backend()
        impl_cls = attn_backend.get_impl_cls()

        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            v_head_size,
            alibi_scale,
            sliding_window,
            **kwargs,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        inplace: bool = True,
    ) -> torch.Tensor:
        return self.impl.forward(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata=attn_metadata,
            inplace=inplace,
        )
