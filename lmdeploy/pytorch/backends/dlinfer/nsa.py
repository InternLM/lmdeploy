# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lmdeploy.pytorch.consts import DSA_INDEXER_K_CACHE_NAME
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequest
from lmdeploy.pytorch.kernels.dlinfer import fill_kv_cache, lightning_indexer
from lmdeploy.utils import get_logger

from ..nsa import NSAIndexFP8BuildSpec, NSAIndexFP8Impl, NSAIndexMeta

logger = get_logger('lmdeploy')


@dataclass
class DlinferNSAIndexMeta(NSAIndexMeta):
    """Dlinfer-specific NSA metadata for Ascend cache writes."""

    kv_start_indices: Tensor | None = None


class DlinferNSAIndexBF16(NSAIndexFP8Impl):
    """Ascend BF16 implementation of the NSA indexer interface."""

    def __init__(self, topk: int, softmax_scale: float, block_size: int,
                 fill: int):
        super().__init__()
        self.topk = topk
        self.softmax_scale = softmax_scale
        self.block_size = block_size

    def get_block_cache_requests(self, geometry: BlockCacheGeometry,
                                 head_dim: int) -> tuple[BlockCacheRequest, ...]:
        """Request the Ascend Lightning Indexer K cache.

        Unlike the CUDA implementation, the Ascend kernel consumes a native
        BF16 cache row. The contiguous block layout therefore stores one row
        as ``[kernel_block_size, head_dim]``; ``_forward_index`` adds the
        singleton head axis expected by ``lightning_indexer``.
        """
        if geometry.logical_block_size != geometry.kernel_block_size:
            raise ValueError(
                'Ascend DSA indexer cache requires equal logical and kernel '
                f'block sizes, got {geometry.logical_block_size} and '
                f'{geometry.kernel_block_size}.')
        request = BlockCacheRequest(
            name=DSA_INDEXER_K_CACHE_NAME,
            shape=(geometry.kernel_block_size, 1, head_dim),
            dtype=torch.bfloat16,
            per_row_contiguous=True,
        )
        return (request, )

    def get_step_metadata(self, attn_metadata) -> DlinferNSAIndexMeta:
        """Build the per-step metadata consumed by the Lightning Indexer."""
        if attn_metadata is None:
            raise RuntimeError('Ascend NSA metadata is required.')

        if isinstance(attn_metadata, DlinferNSAIndexMeta):
            return attn_metadata

        q_seqlens = getattr(attn_metadata, 'q_seqlens', None)
        kv_seqlens = getattr(attn_metadata, 'kv_seqlens', None)
        cu_seqlens_q = getattr(attn_metadata, 'cu_seqlens_q', None)
        if q_seqlens is None or kv_seqlens is None or cu_seqlens_q is None:
            raise RuntimeError(
                'Ascend NSA metadata is missing sequence lengths.')

        return DlinferNSAIndexMeta(
            cu_seqlen_q=cu_seqlens_q,
            q_seqlens=q_seqlens,
            k_seqlens=kv_seqlens,
            cu_seqlen_k=getattr(attn_metadata, 'cu_seqlens_k', None),
            block_offset=attn_metadata.block_offsets,
            # Lightning Indexer consumes the per-request lengths directly;
            # unlike CUDA top-k, it does not need a flattened causal-length
            # vector.  Avoid deriving host scalars here so this path remains
            # safe during Ascend graph replay.
            indexer_kv_seqlens=getattr(attn_metadata, 'indexer_kv_seqlens', None),
            max_q_seqlen=getattr(attn_metadata, 'max_q_seqlen', None),
            max_kv_seqlen=getattr(
                attn_metadata, 'max_kv_seq_len',
                getattr(attn_metadata, 'max_kv_seqlen', None)),
            kv_flatten_size=getattr(attn_metadata, 'kv_flatten_size', None),
            block_size=getattr(attn_metadata, 'block_size', self.block_size),
            is_decoding=getattr(attn_metadata, 'is_decoding', False),
            kv_start_indices=getattr(attn_metadata, 'kv_start_indices', None),
        )

    def _forward_index(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        indexer_k_cache: Tensor,
        meta: DlinferNSAIndexMeta,
    ) -> Tensor:
        if meta.kv_start_indices is None:
            raise RuntimeError(
                'Ascend NSA metadata is missing kv_start_indices')
        if meta.cu_seqlen_q is None:
            raise RuntimeError('Ascend NSA metadata is missing cu_seqlen_q')

        k = k.unsqueeze(-2)
        fill_kv_cache(
            k,
            k,
            indexer_k_cache,
            indexer_k_cache,
            meta.kv_start_indices,
            k_scales_zeros=(),
            v_scales_zeros=(),
            quant_bits=0,
        )

        actual_q = meta.cu_seqlen_q[1:]
        actual_k = meta.k_seqlens
        block_offset = meta.block_offset
        indices = lightning_indexer(
            q.contiguous(),
            indexer_k_cache,
            weights.to(dtype=q.dtype).contiguous(),
            actual_seq_lengths_query=actual_q,
            actual_seq_lengths_key=actual_k,
            block_table=block_offset,
            sparse_count=self.topk,
        )
        if indices.dim() != 3 or indices.size(1) != 1:
            raise RuntimeError(
                'Ascend Lightning Indexer returned an unexpected shape: '
                f'{tuple(indices.shape)}')
        return indices.squeeze(1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        indexer_k_cache: Tensor,
        attn_metadata=None,
        meta: DlinferNSAIndexMeta | None = None,
    ) -> Tensor:
        if meta is None:
            meta = self.get_step_metadata(attn_metadata)
        return self._forward_index(q, k, weights, indexer_k_cache, meta)

    @staticmethod
    def _rotate_partial(x: Tensor, cos: Tensor, sin: Tensor, rope_dim: int,
                        interleaved: bool) -> Tensor:
        """Rotate the leading RoPE slice with the Ascend custom op."""
        x = x.contiguous()
        op = getattr(getattr(torch.ops, "_C_ascend", None),
                     "inplace_partial_rotary_mul", None)
        if op is not None and x.dtype in (torch.bfloat16, torch.float16):
            view = x.unsqueeze(1) if x.ndim == 3 else x.unsqueeze(1).unsqueeze(1)
            rope_cos = cos.reshape(cos.shape[0], 1, 1, cos.shape[-1])
            rope_sin = sin.reshape(sin.shape[0], 1, 1, sin.shape[-1])
            op(view, rope_cos, rope_sin,
               rotary_mode="interleave" if interleaved else "half",
               partial_slice=[0, rope_dim])
            return x

        rope = x[..., :rope_dim]
        tail = x[..., rope_dim:]
        if interleaved:
            even = rope[..., 0::2]
            odd = rope[..., 1::2]
            c = cos[..., :even.shape[-1]]
            ss = sin[..., :even.shape[-1]]
            rotated = torch.stack((even * c - odd * ss,
                                   even * ss + odd * c), dim=-1).flatten(-2)
        else:
            half = rope.shape[-1] // 2
            first, second = rope[..., :half], rope[..., half:]
            rotated = torch.cat((-second, first), dim=-1) * sin
            rotated = torch.cat((first, second), dim=-1) * cos + rotated
        return torch.cat((rotated, tail), dim=-1)

    def forward_fused(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        norm_weight: Tensor,
        norm_bias: Tensor,
        cos: Tensor,
        sin: Tensor,
        indexer_k_cache: Tensor,
        norm_eps: float,
        head_gate_scale: float,
        rope_interleaved: bool,
        attn_metadata=None,
        meta: DlinferNSAIndexMeta | None = None,
    ) -> Tensor:
        """Run fused projection outputs through Ascend indexer preparation."""
        if q.ndim != 3 or k.ndim != 2 or weights.ndim != 2:
            raise ValueError(
                f"Unexpected fused indexer shapes q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}, weights={tuple(weights.shape)}")
        if cos.ndim != 2 or sin.shape != cos.shape:
            raise ValueError("Fused indexer RoPE metadata must be 2-D and matched")
        if q.shape[0] != k.shape[0] or q.shape[0] != weights.shape[0]:
            raise ValueError("Fused indexer token dimensions do not match")
        k = F.layer_norm(k.float(), (k.shape[-1],), norm_weight, norm_bias,
                          norm_eps).to(dtype=k.dtype)
        rope_dim = cos.shape[-1]
        q = self._rotate_partial(q, cos, sin, rope_dim, rope_interleaved)
        k = self._rotate_partial(k, cos, sin, rope_dim, rope_interleaved)
        if meta is None:
            meta = self.get_step_metadata(attn_metadata)
        return self._forward_index(q, k, weights * head_gate_scale,
                                   indexer_k_cache, meta)


def _build_nsa_index_fp8(spec: NSAIndexFP8BuildSpec) -> NSAIndexFP8Impl:
    """Build the Ascend BF16 fallback for the NSA indexer."""
    logger.warning('Ascend backend does not support FP8 indexer; '
                   'falling back to BF16 indexer.')
    return DlinferNSAIndexBF16(spec.top_k, spec.softmax_scale,
                               spec.block_size, spec.fill)
