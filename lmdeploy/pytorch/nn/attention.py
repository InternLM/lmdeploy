# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.distributed import get_tp_world_rank

from ..backends import OpType, get_backend
from ..backends.attention import AttentionMetadata
from .utils import get_distribute_size


def _update_num_heads(num_heads: int, num_kv_heads: int):
    """Update heads."""
    world_size, rank = get_tp_world_rank('attn')
    num_heads = get_distribute_size(num_heads, world_size, rank)
    num_kv_heads = get_distribute_size(num_kv_heads, world_size, rank)
    return num_heads, num_kv_heads


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
        logit_softcapping: float = 0.0,
        causal: bool = True,
        use_flash_mla: bool = False,
        learnable_sink: bool = False,
        block_sparse_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if v_head_size is None:
            v_head_size = head_size
        self.origin_num_heads = num_heads
        num_heads, num_kv_heads = _update_num_heads(num_heads, num_kv_heads)
        self.num_heads = num_heads

        layer_backend = get_backend()
        impl_builder = layer_backend.get_layer_impl_builder(OpType.PagedAttention)

        self.impl = impl_builder.build(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            use_flash_mla=use_flash_mla,
            learnable_sink=learnable_sink,
            block_sparse_size=block_sparse_size,
            **kwargs,
        )

        if alibi:
            self.alibi_ready = False
        else:
            self.alibi_ready = True

    def _lazy_init(self, device):
        """Lazy init."""
        if not self.alibi_ready:
            _, rank = get_tp_world_rank('attn')
            start = self.num_heads * rank
            end = start + self.num_heads
            alibi_slopes = self.impl.make_alibi_slopes(start,
                                                       end,
                                                       self.origin_num_heads,
                                                       alibi_scale=1,
                                                       dtype=torch.float32,
                                                       device=device)
            self.impl.set_alibi_slopes(alibi_slopes)
            self.alibi_ready = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        s_aux: torch.Tensor = None,
        nsa_indices: torch.Tensor = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        """forward."""
        self._lazy_init(query.device)

        kwargs = dict()
        if nsa_indices is not None:
            kwargs['nsa_indices'] = nsa_indices
        if s_aux is not None:
            kwargs['learnable_sink'] = s_aux
        return self.impl.forward(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata=attn_metadata,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            inplace=inplace,
            **kwargs,
        )

    @staticmethod
    def update_meta_flashmla(attn_metadata: AttentionMetadata, num_attention_heads):
        get_backend().update_meta_flashmla(attn_metadata, num_attention_heads)


class FlashAttention(nn.Module):
    """Flash attention w/o paging."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_dim: int = None,
        causal: bool = True,
        sliding_window: int = None,
        logit_softcapping: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if v_head_dim is None:
            v_head_dim = head_dim
        num_heads, num_kv_heads = _update_num_heads(num_heads, num_kv_heads)

        layer_backend = get_backend()

        impl_builder = layer_backend.get_layer_impl_builder(OpType.FlashAttention)

        self.impl = impl_builder.build(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_dim=v_head_dim,
            causal=causal,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            **kwargs,
        )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                q_start_loc: torch.Tensor,
                q_seqlens: torch.Tensor,
                kv_start_loc: torch.Tensor = None,
                kv_seqlens: torch.Tensor = None,
                max_q_seqlen: int = None) -> torch.Tensor:
        """forward."""

        if max_q_seqlen is None:
            max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))

        if kv_start_loc is None and kv_seqlens is None:
            kv_start_loc = q_start_loc
            kv_seqlens = q_seqlens

        assert kv_start_loc is not None
        assert kv_seqlens is not None

        return self.impl.forward(
            query,
            key,
            value,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,
            max_q_seqlen=max_q_seqlen,
        )
