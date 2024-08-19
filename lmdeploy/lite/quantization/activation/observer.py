# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.lite.utils.global_avail import GlobalAvailMixin


class ActivationObserver(GlobalAvailMixin):
    """A class to observe and record the max, min, mean, absolute max, and
    absolute mean value of a given tensor.

    Also keeps track of the number of batches observed.
    """
    observed = False

    def __init__(self, dim: int) -> None:
        """Constructor for ActivationObserver.

        Args:
            dim : Dimension of the tensor
        """
        self.dim = dim
        self.max_val = torch.full((dim, ), -torch.inf, dtype=torch.float16)
        self.min_val = torch.full((dim, ), torch.inf, dtype=torch.float16)
        self.absmax_val = torch.full((dim, ), 0, dtype=torch.float16)
        self.absmean_val = torch.full((dim, ), 0, dtype=torch.float16)
        self.mean_val = torch.full((dim, ), 0, dtype=torch.float16)
        self.num_batches_tracked = 0
        self.value = None
        self.ratio = None
        self.num_ratio_tracked = 0

    @classmethod
    def disable(cls):
        """To avoid recomputation in search scale process."""
        cls.observed = True

    @classmethod
    def enable(cls):
        """To avoid recomputation in search scale process."""
        cls.observed = False

    @torch.no_grad()
    def observe(self, x: torch.Tensor, save_input: bool = False) -> None:
        """Function to observe the input tensor and update the max, min, mean,
        absolute max, absolute mean values and number of batches tracked.

        Args:
            x : Input tensor
        """
        if self.observed:
            return
        if len(x.shape) != 3:
            return
        assert x.size(2) == self.dim
        cur_val = x.flatten(0, 1)
        cur_max = cur_val.max(0)[0].cpu()
        cur_min = cur_val.min(0)[0].cpu()
        cur_mean = cur_val.mean(0).cpu()

        cur_abs = cur_val.abs()
        cur_absmax = cur_abs.max(0)[0].cpu()
        cur_absmean = cur_abs.mean(0).cpu()

        self.max_val = torch.maximum(self.max_val, cur_max)
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.absmax_val = torch.maximum(self.absmax_val, cur_absmax)
        if save_input:
            self.value = x

        # Update mean and absmean value with accumulated sum divided
        # by total number of batches
        self.mean_val = (
            (self.mean_val * self.num_batches_tracked + cur_mean) /
            (self.num_batches_tracked + 1))
        self.absmean_val = (
            (self.absmean_val * self.num_batches_tracked + cur_absmean) /
            (self.num_batches_tracked + 1))

        # Increment the count of batches tracked
        self.num_batches_tracked += 1

    @torch.no_grad()
    def save_ratio(self, ratio: float) -> None:
        if self.ratio is None:
            self.ratio = 0
        self.ratio = (self.ratio * self.num_ratio_tracked +
                      ratio) / (self.num_ratio_tracked + 1)
        self.num_ratio_tracked += 1
