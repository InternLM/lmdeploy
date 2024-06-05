# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .dispatcher import FunctionDispatcher


def _multinomial_sampling_api(scores: torch.Tensor,
                              seeds: torch.LongTensor,
                              offsets: torch.LongTensor,
                              indices: torch.Tensor = None):
    """multinomial sampling."""
    ...


multinomial_sampling = FunctionDispatcher('multinomial_sampling').make_caller(
    _multinomial_sampling_api)
