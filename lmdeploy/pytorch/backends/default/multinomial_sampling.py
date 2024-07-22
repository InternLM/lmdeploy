# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..multinomial_sampling import (MultinomialSamplingBuilder,
                                    MultinomialSamplingImpl)


class DefaultMultinomialSamplingImpl(MultinomialSamplingImpl):

    def forward(scores: torch.Tensor,
                seeds: torch.LongTensor,
                offsets: torch.LongTensor,
                indices: torch.Tensor = None):
        sampled_index = torch.multinomial(scores,
                                          num_samples=1,
                                          replacement=True)
        outputs = torch.gather(indices, dim=1, index=sampled_index)
        return outputs.view(-1)


class DefaultMultinomialSamplingBuilder(MultinomialSamplingBuilder):

    def build():
        return DefaultMultinomialSamplingImpl()
