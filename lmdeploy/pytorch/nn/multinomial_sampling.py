# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..backends import OpType, get_backend


def multinomial_sampling(scores: torch.Tensor,
                         seeds: torch.LongTensor,
                         offsets: torch.LongTensor,
                         indices: torch.Tensor = None):
    """multinomial sampling op."""
    impl_builder = get_backend().get_layer_impl_builder(
        OpType.MultinomialSampling)
    return impl_builder.build().forward(scores, seeds, offsets, indices)
