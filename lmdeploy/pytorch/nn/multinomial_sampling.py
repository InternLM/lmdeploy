# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..backends import get_backend
from ..backends.multinomial_sampling import MultinomialSamplingBuildSpec

_MULTINOMIAL_SAMPLING_SPEC = MultinomialSamplingBuildSpec()


def multinomial_sampling(scores: torch.Tensor,
                         seeds: torch.LongTensor,
                         offsets: torch.LongTensor,
                         indices: torch.Tensor = None):
    """Multinomial sampling op."""
    return get_backend().build_op(_MULTINOMIAL_SAMPLING_SPEC).forward(scores, seeds, offsets, indices)
