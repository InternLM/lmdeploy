# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ..backends import LayerType, get_backend
from ..backends.rotary_embedding import EmbeddingType


def build_rotary_embedding(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        emb_type: EmbeddingType = EmbeddingType.Default) -> nn.Module:
    backend = get_backend()

    builder = backend.get_layer_impl_builder(LayerType.RotaryEmbedding)
    return builder.build(dim, max_position_embeddings, base, scaling_factor,
                         emb_type)


class ApplyRotaryEmb(nn.Module):

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(LayerType.ApplyRotaryEmb)
        self.impl = builder.build()

    def forward(self, query, key, cos, sin, inplace: bool = True):
        return self.impl.forward(query, key, cos, sin, inplace)
