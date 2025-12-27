# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import torch
from torch import nn


class Embedding(nn.Embedding):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=dtype, device=device, **kwargs)
        self.orig_dtype = self.weight.dtype

    def update_weight_dtype(self, dtype):
        """Update weight dtype."""
        if self.weight.dtype != dtype:
            self.weight.data = self.weight.data.to(dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = super().forward(input)
        if out.dtype != self.orig_dtype:
            out = out.to(dtype=self.orig_dtype)
        return out
