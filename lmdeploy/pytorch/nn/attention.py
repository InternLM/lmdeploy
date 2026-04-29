# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.utils import get_logger

from ..backends import OpType, get_backend
from ..backends.attention import AttentionMetadata
from .utils import get_distribute_size

logger = get_logger('lmdeploy')
_DEFAULT_FP8_SCALE_WARNED = False


def _is_normal_fp8_quant_policy(quant_policy: QuantPolicy):
    """Return whether quant_policy uses scalar-scale FP8 KV cache."""
    return quant_policy in (QuantPolicy.FP8, QuantPolicy.FP8_E5M2)


def _get_fp8_dtype(quant_policy: QuantPolicy):
    """Get the torch FP8 dtype for normal FP8 KV cache."""
    if quant_policy == QuantPolicy.FP8:
        return torch.float8_e4m3fn
    if quant_policy == QuantPolicy.FP8_E5M2:
        return torch.float8_e5m2
    raise ValueError(f'Not a normal FP8 quant policy: {quant_policy}')


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
        scale_device = kwargs.get('device', None)
        self.register_buffer('k_scale', torch.ones((), dtype=torch.float32, device=scale_device))
        self.register_buffer('v_scale', torch.ones((), dtype=torch.float32, device=scale_device))
        self.calculate_kv_scales = False

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

    def set_calculate_kv_scales(self, enabled: bool):
        """Set one-shot scalar FP8 KV scale calculation."""
        self.calculate_kv_scales = enabled

    @torch.no_grad()
    def finalize_kv_scales(self, quant_policy: QuantPolicy):
        """Finalize loaded scalar FP8 KV scales before inference."""
        global _DEFAULT_FP8_SCALE_WARNED
        if not _is_normal_fp8_quant_policy(quant_policy) or self.calculate_kv_scales:
            return

        if _DEFAULT_FP8_SCALE_WARNED or quant_policy != QuantPolicy.FP8:
            return
        if self.k_scale.item() == 1.0 and self.v_scale.item() == 1.0:
            logger.warning('Using normal FP8 E4M3 KV cache with default k_scale=v_scale=1.0. '
                           'This matches vLLM no-calibration behavior but may affect accuracy.')
            _DEFAULT_FP8_SCALE_WARNED = True

    def has_pending_kv_scale_calculation(self):
        """Return whether this layer still needs one-shot KV scale
        calculation."""
        return self.calculate_kv_scales

    def _effective_kv_scales(self):
        """Return scalar K/V scales."""
        return self.k_scale, self.v_scale

    @torch.no_grad()
    def _maybe_calculate_kv_scales(self, key: torch.Tensor, value: torch.Tensor, quant_policy: QuantPolicy):
        """Calculate scalar FP8 KV scales once, then freeze them."""
        if not self.calculate_kv_scales or not _is_normal_fp8_quant_policy(quant_policy):
            return
        fp8_max = torch.finfo(_get_fp8_dtype(quant_policy)).max
        min_scale = torch.tensor(1e-6, dtype=torch.float32, device=key.device)
        k_scale = torch.maximum(key.abs().max().to(torch.float32) / fp8_max, min_scale)
        v_scale = torch.maximum(value.abs().max().to(torch.float32) / fp8_max, min_scale)
        self.k_scale.copy_(k_scale)
        self.v_scale.copy_(v_scale)
        self.calculate_kv_scales = False

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

        quant_policy = attn_metadata.quant_policy
        if key is not None and value is not None:
            self._maybe_calculate_kv_scales(key, value, quant_policy)
        if _is_normal_fp8_quant_policy(quant_policy):
            k_scale, v_scale = self._effective_kv_scales()
        else:
            k_scale = None
            v_scale = None

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
            k_scale=k_scale,
            v_scale=v_scale,
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
