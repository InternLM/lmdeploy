# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.cuda.v4_compressor import (
    fill_compress_state,
    fill_compressed_kv,
    score_kv,
)

from ..compressor import BaseV4Compressor, BaseV4CompressorBuilder, V4CompressorMetadata


class TritonV4CompressorImpl(BaseV4Compressor):

    def __init__(self, compress_ratio: int, overlap: bool, head_dim: int) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        self.head_dim = head_dim

    def score_and_fill_state(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        state_ids: torch.Tensor,
        meta: V4CompressorMetadata,
    ) -> torch.Tensor:
        compressed_kv = kv.new_zeros(kv.size(0), self.head_dim)
        score_kv(kv, score, ape, kv_state, score_state, state_ids,
                 meta.cu_q_seqlens, meta.kv_seqlens, compressed_kv,
                 self.overlap, meta.max_kv_seqlen)
        fill_compress_state(kv, score, ape, kv_state, score_state, state_ids,
                            meta.cu_q_seqlens, meta.kv_seqlens)
        return compressed_kv

    def write_compressed_kv(
        self,
        compressed_kv: torch.Tensor,
        kv_cache: torch.Tensor | None,
        meta: V4CompressorMetadata,
        fp8_cache: torch.Tensor | None = None,
        kv_scale_cache: torch.Tensor | None = None,
    ) -> None:
        fill_compressed_kv(
            compressed_kv, kv_cache,
            meta.cu_q_seqlens, meta.kv_seqlens,
            meta.block_offsets, self.compress_ratio, meta.block_size,
            meta.max_kv_seqlen,
            fp8_cache=fp8_cache,
            kv_scale_cache=kv_scale_cache)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=x.size(-1)**-0.5)


class TritonV4CompressorBuilder(BaseV4CompressorBuilder):

    @staticmethod
    def build(compress_ratio: int, overlap: bool, head_dim: int) -> BaseV4Compressor:
        return TritonV4CompressorImpl(
            compress_ratio=compress_ratio,
            overlap=overlap,
            head_dim=head_dim)
