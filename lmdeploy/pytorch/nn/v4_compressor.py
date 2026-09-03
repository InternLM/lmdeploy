# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.compressor import V4CompressorMetadata
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheBinding, BlockCacheRequestContext


class V4Compressor(nn.Module):
    """DeepSeek V4 compressor wrapper."""

    def __init__(self,
                 compress_ratio: int,
                 overlap: bool,
                 head_dim: int,
                 is_indexer: bool = False):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.V4Compressor)
        self.impl = impl_builder.build(
            compress_ratio=compress_ratio,
            overlap=overlap,
            head_dim=head_dim,
            is_indexer=is_indexer)
        self._block_cache_bindings: dict[str, BlockCacheBinding] = {}

    def get_block_cache_requests(self, context: BlockCacheRequestContext):
        """Return block-cache requirements from the selected implementation."""
        return self.impl.get_block_cache_requests(context.geometry)

    def bind_block_cache(self, binding: BlockCacheBinding):
        """Retain one logical cache row assigned by the worker collector."""
        self._block_cache_bindings[binding.cache_name] = binding

    def resolve_block_caches(self, cache_view: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Resolve this compressor's logical bindings to physical tensors."""
        if not self._block_cache_bindings:
            raise RuntimeError('The V4 compressor block cache has not been bound.')
        get_row = getattr(cache_view, 'row', None)
        if get_row is not None:
            return {
                name: get_row(name, binding.consumer_row)
                for name, binding in self._block_cache_bindings.items()
            }
        return {
            name: cache_view[name][binding.consumer_row]
            for name, binding in self._block_cache_bindings.items()
        }

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
        return self.impl.score_and_fill_state(
            kv, score, ape, kv_state, score_state, state_ids, meta)

    def write_compressed_kv(
        self,
        compressed_kv: torch.Tensor,
        block_caches: Mapping[str, torch.Tensor],
        meta: V4CompressorMetadata,
    ) -> None:
        self.impl.write_compressed_kv(compressed_kv, block_caches, meta)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl.rotate_activation(x)
