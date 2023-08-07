# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Type, TypeVar, Union

import torch
from torch import nn


class WeightOnlyQLinear(nn.Module):
    """This class implements weight only quantization linear.

    Args:
            w_bit (int): number of bits for quantization.
            symmetry (bool): If true, use symmetric quantization,
                otherwise use asymmetric quantization.
            group_size (int): size of the quantization group.
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            bias (Tensor, optional): Learnable bias. Defaults to None.
    """

    def __init__(self,
                 w_bit: int,
                 symmetry: bool,
                 group_size: int,
                 in_features: int,
                 out_features: int,
                 bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()

        if w_bit not in [2, 4, 8]:
            raise NotImplementedError('Only 2,4,8 bit are supported for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        w_pack_oc = out_features // (32 // self.w_bit)
        w_inc = in_features
        weight = torch.zeros((w_inc, w_pack_oc), dtype=torch.int32)
        self.register_buffer('qweight', weight)

        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

        s_inc = in_features // self.group_size
        s_oc = out_features
        scales = torch.zeros((s_inc, s_oc), dtype=torch.float16)
        self.register_buffer('scales', scales)

        if not symmetry:
            z_inc = in_features // self.group_size
            z_oc = out_features // (32 // self.w_bit)
            zeros = torch.zeros((z_inc, z_oc), dtype=torch.int32)
            self.register_buffer('qzeros', zeros)
        else:
            self.qzeros = None

    @classmethod
    def from_linear(
            cls: Type['WeightOnlyQLinear'],
            linear: nn.Linear,
            quantizer: TypeVar('Quantizer'),
            awq_layout: Union[str, bool] = False) -> 'WeightOnlyQLinear':
        """Create a WeightOnlyQLinear object from a PyTorch Linear object.

        Args:
            linear (nn.Linear): PyTorch Linear object.
            quantizer (Quantizer): Object that handles quantization.
            awq_layout (str or bool, optional): AWQ layout. Defaults to False.

        Returns:
            WeightOnlyQLinear: A WeightOnlyQLinear object.
        """
        device = linear.weight.device

        w_bit = quantizer.bits
        pack_num = 32 // w_bit
        if awq_layout == 'awq':
            assert w_bit == 4
            pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            pack_order = torch.arange(pack_num)
        group_size = quantizer.group_size
        symmetry = quantizer.symmetry

        in_features = linear.in_features
        out_features = linear.out_features
        bias = linear.bias

        qlinear = cls(w_bit, symmetry, group_size, in_features, out_features,
                      bias)

        qparams = quantizer.calculate_qparams(linear.weight)
        i32_w = quantizer.quant(linear.weight, qparams, real=True)
        i32_w = i32_w.t().contiguous()

        pack_int_w = torch.zeros_like(qlinear.qweight).to(device)

        for col in range(pack_int_w.shape[1]):
            for i in range(pack_num):
                pack_int_w_col = i32_w[:, col * pack_num + pack_order[i]]
                pack_int_w[:, col] |= pack_int_w_col << (i * w_bit)

        qlinear.qweight = pack_int_w
        qlinear.scales = qparams.scales.clone().half()

        if qparams.zero_points is not None:
            zeros = qparams.zero_points.to(torch.int32).to(device)
            zeros = zeros.reshape(out_features,
                                  in_features // group_size).t().contiguous()
            pack_int_zeros = torch.zeros_like(qlinear.qzeros).to(device)

            for col in range(pack_int_zeros.shape[1]):
                for i in range(pack_num):
                    qzero_col = zeros[:, col * pack_num + pack_order[i]]
                    pack_int_zeros[:, col] |= qzero_col << (i * w_bit)
            qlinear.qzeros = pack_int_zeros

        qlinear.to('cpu')

        return qlinear
