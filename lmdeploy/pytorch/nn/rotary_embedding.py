# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from ..backends import OpType, get_backend
from ..backends.rotary_embedding import (Llama3Parameters,
                                         LongRoPEScalingParameters, RopeType,
                                         YarnParameters)


def build_rotary_embedding(dim: int,
                           max_position_embeddings: int = 2048,
                           base: int = 10000,
                           scaling_factor: float = 1.0,
                           yarn_params: YarnParameters = None,
                           longrope_params: LongRoPEScalingParameters = None,
                           llama3_params: Llama3Parameters = None,
                           emb_type: RopeType = RopeType.Default) -> nn.Module:
    """build rotary embedding op."""
    backend = get_backend()

    builder = backend.get_layer_impl_builder(OpType.RotaryEmbedding)
    return builder.build(dim,
                         max_position_embeddings,
                         base,
                         scaling_factor,
                         yarn_params=yarn_params,
                         longrope_params=longrope_params,
                         llama3_params=llama3_params,
                         emb_type=emb_type)


class ApplyRotaryEmb(nn.Module):
    """apply rotary embedding."""

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.ApplyRotaryEmb)
        self.impl = builder.build()

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True):
        """forward."""
        return self.impl.forward(query, key, cos, sin, inplace)
