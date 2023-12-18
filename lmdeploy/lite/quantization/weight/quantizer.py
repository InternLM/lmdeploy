# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Optional

import torch

from lmdeploy.lite.utils import (QParams, cal_qparams_per_channel_absmax,
                                 cal_qparams_per_channel_minmax,
                                 cal_qparams_per_group_absmax,
                                 cal_qparams_per_group_minmax,
                                 cal_qparams_per_tensor_absmax,
                                 cal_qparams_per_tensor_minmax, precise_round)
from lmdeploy.lite.utils.global_avail import GlobalAvailMixin


class WeightQuantizer(GlobalAvailMixin):
    """A class for performing weight quantization of neural networks.

    The WeightQuantizer class provides various methods to quantize the weights
    of a neural network. This helps in reducing the memory requirements and
    computational complexity of the model, potentially offering faster
    inference and lower power consumption.

    Attributes:
        bits (int): The bit width for quantization.
        symmetry (bool): If True, use absmax scaling; if False,
            use min-max scaling.
        granularity (str): The granularity of quantization. Available options
            are 'per_channel', 'per_tensor', and 'per_group'.
        group_size (Optional[int]): If using 'per_group' quantization, this is
            the number of channels in each group.

    Example:

        # Instantiate the weight quantizer with specific quantization settings
        quantizer = WeightQuantizer(bits=8,
                                     symmetry=True,
                                     granularity='per_tensor')

        # Calculate the quantization parameters for given weights
        qparams = quantizer.calculate_qparams(weights)

        # Perform fake quantization on the weights
        quantized_weights = quantizer.fake_quant(weights, qparams)
    """

    CAL_FUNC_MAP: Dict[str, Dict[str, Callable]] = {
        'per_group': {
            'absmax': cal_qparams_per_group_absmax,
            'minmax': cal_qparams_per_group_minmax,
        },
        'per_channel': {
            'absmax': cal_qparams_per_channel_absmax,
            'minmax': cal_qparams_per_channel_minmax,
        },
        'per_tensor': {
            'absmax': cal_qparams_per_tensor_absmax,
            'minmax': cal_qparams_per_tensor_minmax,
        },
    }

    def __init__(self,
                 bits: int,
                 symmetry: bool,
                 granularity: str,
                 group_size: Optional[int] = -1):

        assert bits in [4, 8], "The 'bits' argument must be either 4 or 8."
        self.bits = bits

        if granularity not in ['per_channel', 'per_tensor', 'per_group']:
            raise NotImplementedError(
                "The 'granularity' argument must be one of 'per_channel', "
                "'per_tensor', or 'per_group'.")

        self.granularity = granularity

        if self.granularity == 'per_group':
            assert group_size > 0, \
                "The 'group_size' argument must be greater than 0."

        self.group_size = group_size

        # If symmetry is True, use absmax to compute scales
        # If symmetry is False, use minmax to compute scales and zeor-points
        self.symmetry = symmetry
        self.observer = 'absmax' if symmetry else 'minmax'

    def calculate_qparams(self, weight: torch.Tensor) -> QParams:
        """Calculate the quantization parameters for the given weight tensor.

        Args:
            weight (torch.Tensor): The weight tensor with shape
                (out_features, in_features).

        Returns:
            QParams: A namedtuple containing 'scales' and 'zero_points'.
        """

        cal_func = self.CAL_FUNC_MAP[self.granularity][self.observer]
        if self.granularity == 'per_group':
            return cal_func(weight, self.bits, self.group_size)
        else:
            return cal_func(weight, self.bits)

    def quant(self,
              weight: torch.Tensor,
              qparams: Optional[QParams] = None,
              real: bool = False) -> torch.Tensor:
        """Perform fake quantization on the given weight tensor.

        Args:
            weight (torch.Tensor): The weight tensor with shape
                (out_features, in_features).
            qparams (Optional[QParams]): A namedtuple containing 'scales'
                and 'zero_points'.
            real (bool): If True, return the tensor with quantized type.

        Returns:
            torch.Tensor: The fake quantized weight tensor.
        """

        float_w = weight.float()

        if qparams is None:
            qparams = self.calculate_qparams(float_w)

        scales = qparams.scales
        zero_points = qparams.zero_points

        out_c, in_c = weight.shape

        # Reshape the weights if using per_group quantization
        # per tensor scales shape: [1]
        # per channel scales shape: [out_c, 1]
        # per group scales shape: [out_c, in_c//group_size, 1]
        if len(scales.shape) > 2:
            # scales shape: [out_c, in_c//group_size, 1]
            float_w = float_w.reshape(out_c, scales.shape[1], -1)

        if zero_points is None:
            assert self.symmetry
            real_qweight = (float_w / scales).round()
            fake_qweight = real_qweight * scales

        else:
            assert not self.symmetry

            real_qweight = precise_round(
                (float_w - float_w.min(-1, keepdim=True)[0]) / scales)
            fake_qweight = (real_qweight - zero_points) * scales

        if len(scales.shape) > 2:
            real_qweight = real_qweight.reshape(out_c, in_c)
            fake_qweight = fake_qweight.reshape(out_c, in_c)

        if real:
            return real_qweight.to(torch.int32)
        else:
            return fake_qweight.to(weight.dtype)
