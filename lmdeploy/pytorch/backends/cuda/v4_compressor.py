# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping

import torch

from lmdeploy.pytorch.consts import (
    V4_COMPRESSED_KV_R4_CACHE_NAME,
    V4_COMPRESSED_KV_R128_CACHE_NAME,
    V4_INDEX_KV_R4_CACHE_NAME,
    V4_INDEX_SCALE_BYTES,
    V4_PACKED_TOKEN_DIM,
    v4_packed_index_cache_shape,
)
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequest
from lmdeploy.pytorch.kernels.cuda.v4_compressor import (
    fill_compressed_kv,
    score_and_fill_state_decode,
    score_and_fill_state_prefill,
)

from ..compressor import BaseV4Compressor, BaseV4CompressorBuilder, V4CompressorMetadata


def _get_v4_packed_index_cache_views(index_cache: torch.Tensor,
                                     head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 value and FP32 scale views of a packed V4 index cache."""
    if index_cache.dtype != torch.uint8:
        raise TypeError(f'Packed V4 index cache must be uint8, got {index_cache.dtype}.')
    if index_cache.dim() != 4 or index_cache.size(2) != 1:
        raise ValueError('Packed V4 index cache must have shape [num_blocks, entries, 1, head_dim + 4].')
    if index_cache.size(-1) != head_dim + V4_INDEX_SCALE_BYTES:
        raise ValueError(f'Packed V4 index cache last dim must be {head_dim + V4_INDEX_SCALE_BYTES}, '
                         f'got {index_cache.size(-1)}.')

    num_blocks = index_cache.size(0)
    entries_per_block = index_cache.size(1)
    flat = index_cache.view(num_blocks, -1)
    value_bytes = entries_per_block * head_dim
    scale_bytes = entries_per_block * V4_INDEX_SCALE_BYTES
    values = flat[:, :value_bytes].view(torch.float8_e4m3fn).view(
        num_blocks, entries_per_block, head_dim)
    scales = flat[:, value_bytes:value_bytes + scale_bytes].view(torch.float32).view(
        num_blocks, entries_per_block, 1)
    return values, scales


class TritonV4CompressorImpl(BaseV4Compressor):

    def __init__(self,
                 compress_ratio: int,
                 overlap: bool,
                 head_dim: int,
                 is_indexer: bool = False) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        self.head_dim = head_dim
        self.is_indexer = is_indexer
        self.block_cache_name = self._resolve_block_cache_name()

    def _resolve_block_cache_name(self) -> str:
        if self.is_indexer:
            if self.compress_ratio != 4:
                raise ValueError('The V4 indexer compressor requires compress_ratio=4.')
            return V4_INDEX_KV_R4_CACHE_NAME
        try:
            return {
                4: V4_COMPRESSED_KV_R4_CACHE_NAME,
                128: V4_COMPRESSED_KV_R128_CACHE_NAME,
            }[self.compress_ratio]
        except KeyError as e:
            raise ValueError(f'Unsupported V4 compression ratio: {self.compress_ratio}.') from e

    def get_block_cache_requests(self, geometry: BlockCacheGeometry) -> tuple[BlockCacheRequest, ...]:
        """Describe the CUDA compressor's contiguous per-consumer cache."""
        entries_per_block = geometry.kernel_block_size // self.compress_ratio
        if self.is_indexer:
            request = BlockCacheRequest(
                name=self.block_cache_name,
                shape=v4_packed_index_cache_shape(entries_per_block, self.head_dim),
                dtype=torch.uint8,
                per_row_contiguous=True,
            )
            return (request, )

        request = BlockCacheRequest(
            name=self.block_cache_name,
            shape=(entries_per_block, V4_PACKED_TOKEN_DIM),
            dtype=torch.float8_e4m3fn,
            per_row_contiguous=True,
        )
        return (request, )

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
        is_decoding = kv.size(0) == state_ids.size(0)
        if is_decoding:
            score_and_fill_state_decode(
                kv, score, ape, kv_state, score_state, state_ids,
                meta.cu_q_seqlens, meta.kv_seqlens, compressed_kv,
                self.overlap)
        else:
            score_and_fill_state_prefill(
                kv, score, ape, kv_state, score_state, state_ids,
                meta.cu_q_seqlens, meta.kv_seqlens, compressed_kv,
                self.overlap, meta.max_q_seqlen)
        return compressed_kv

    def write_compressed_kv(
        self,
        compressed_kv: torch.Tensor,
        block_caches: Mapping[str, torch.Tensor],
        meta: V4CompressorMetadata,
    ) -> None:
        cache = block_caches[self.block_cache_name]
        kv_cache = None
        kv_scale_cache = None
        fp8_cache = None
        if self.is_indexer:
            kv_cache = cache
            kv_cache, kv_scale_cache = _get_v4_packed_index_cache_views(kv_cache, compressed_kv.size(-1))
        else:
            fp8_cache = cache
        fill_compressed_kv(
            compressed_kv, kv_cache,
            meta.cu_q_seqlens, meta.kv_seqlens,
            meta.block_offsets, self.compress_ratio, meta.block_size,
            meta.max_q_seqlen,
            fp8_cache=fp8_cache,
            kv_scale_cache=kv_scale_cache)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=x.size(-1)**-0.5)


class TritonV4CompressorBuilder(BaseV4CompressorBuilder):

    @staticmethod
    def build(compress_ratio: int,
              overlap: bool,
              head_dim: int,
              is_indexer: bool = False) -> BaseV4Compressor:
        return TritonV4CompressorImpl(
            compress_ratio=compress_ratio,
            overlap=overlap,
            head_dim=head_dim,
            is_indexer=is_indexer)
