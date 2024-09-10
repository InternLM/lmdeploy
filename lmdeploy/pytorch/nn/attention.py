# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..backends import OpType, get_backend
from ..backends.attention import AttentionMetadata
from .utils import get_distribute_size, get_world_rank


class Attention(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = None,
        replicate_kv: bool = False,
        **kwargs,
    ):
        super().__init__()
        num_heads, num_kv_heads = self._update_num_heads(
            num_heads, num_kv_heads, replicate_kv)

        layer_backend = get_backend()
        impl_builder = layer_backend.get_layer_impl_builder(OpType.Attention)

        self.impl = impl_builder.build(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            **kwargs,
        )

    def _update_num_heads(self, num_heads: int, num_kv_heads: int,
                          replicate_kv: bool):
        """update heads."""
        world_size, rank = get_world_rank()
        num_heads = get_distribute_size(num_heads, world_size, rank)
        if not replicate_kv:
            num_kv_heads = get_distribute_size(num_kv_heads, world_size, rank)
        return num_heads, num_kv_heads

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
        """forward."""
        return self.impl.forward(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata=attn_metadata,
            inplace=inplace,
        )
