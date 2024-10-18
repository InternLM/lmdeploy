# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.cuda import multinomial_sampling

from ..multinomial_sampling import (MultinomialSamplingBuilder,
                                    MultinomialSamplingImpl)


class TritonMultinomialSamplingImpl(MultinomialSamplingImpl):

    def forward(self,
                scores: torch.Tensor,
                seeds: torch.LongTensor,
                offsets: torch.LongTensor,
                indices: torch.Tensor = None):
        """forward."""
        return multinomial_sampling(scores, seeds, offsets, indices)


class TritonMultinomialSamplingBuilder(MultinomialSamplingBuilder):
    """triton multinomial sampling builder."""

    def build():
        """build."""
        return TritonMultinomialSamplingImpl()
