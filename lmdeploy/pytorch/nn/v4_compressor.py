# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.compressor import V4CompressorBuildSpec, V4CompressorMetadata
from lmdeploy.pytorch.models.patch import get_build_model_context


class V4Compressor(nn.Module):
    """DeepSeek V4 compressor wrapper."""

    def __init__(self, compress_ratio: int, overlap: bool, head_dim: int):
        super().__init__()
        self.impl = get_backend().build_op(
            V4CompressorBuildSpec(
                compress_ratio=compress_ratio,
                overlap=overlap,
                head_dim=head_dim,
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

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
        kv_cache: torch.Tensor | None,
        meta: V4CompressorMetadata,
        fp8_cache: torch.Tensor | None = None,
        kv_scale_cache: torch.Tensor | None = None,
    ) -> None:
        self.impl.write_compressed_kv(
            compressed_kv, kv_cache, meta,
            fp8_cache=fp8_cache,
            kv_scale_cache=kv_scale_cache)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl.rotate_activation(x)
