# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.backends.compressor import V4CompressorMetadata
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager


class V4Compressor(nn.Module):
    """DeepSeek V4 compressor wrapper."""

    def __init__(self, compress_ratio: int, overlap: bool, head_dim: int):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.V4Compressor)
        self.impl = impl_builder.build(
            compress_ratio=compress_ratio,
            overlap=overlap,
            head_dim=head_dim)

    def score_and_fill_state(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        state_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        meta = self._build_metadata(attn_metadata)
        return self.impl.score_and_fill_state(
            kv, score, ape, kv_state, score_state, state_ids, meta)

    def write_compressed_kv(
        self,
        compressed_kv: torch.Tensor,
        kv_cache: torch.Tensor | None,
        attn_metadata: AttentionMetadata,
        fp8_cache: torch.Tensor | None = None,
        kv_scale_cache: torch.Tensor | None = None,
    ) -> None:
        meta = self._build_metadata(attn_metadata)
        self.impl.write_compressed_kv(
            compressed_kv, kv_cache, meta,
            fp8_cache=fp8_cache,
            kv_scale_cache=kv_scale_cache)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl.rotate_activation(x)

    def _build_metadata(self, attn_metadata: AttentionMetadata) -> V4CompressorMetadata:
        step_ctx = get_step_ctx_manager().current_context()
        cache_config = step_ctx.cache_config
        is_decoding = attn_metadata.is_decoding
        max_kv_seqlen = (cache_config.block_size * cache_config.num_gpu_blocks
                         if is_decoding else step_ctx.max_kv_seqlen)
        return V4CompressorMetadata(
            cu_q_seqlens=attn_metadata.cu_seqlens_q,
            kv_seqlens=attn_metadata.kv_seqlens,
            block_offsets=attn_metadata.block_offsets,
            block_size=cache_config.block_size,
            max_kv_seqlen=max_kv_seqlen,
        )
