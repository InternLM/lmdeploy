# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List

import torch


def identity(x):
    return x


def to_half(x: torch.Tensor):
    return x.to(torch.half)


def to_float(x: torch.Tensor):
    return x.to(torch.float)


def to_fp8(x: torch.Tensor):
    assert x.dtype == torch.uint8
    return x.view(dtype=torch.float8_e4m3fn)


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


class Parameter:
    KEY = ()

    @classmethod
    def take(cls, keys: List[str]):
        if not any(k.endswith(cls.KEYS[0]) for k in keys):
            return False
        xs = []
        for k in keys:
            if any(k.endswith(p) for p in cls.KEYS):
                xs.append(k)
        for x in xs:
            keys.remove(x)
        return True

    @abstractmethod
    def __call__(cls, f, g, i):
        pass


class QuantWeightOnly(Parameter):
    KEYS = '.qweight', '.scales', '.qzeros'

    def __call__(self, f, g, i):
        f(i, g('qweight'), 'qweight', pack_u4_row)
        f(i, g('scales'), 'scales', to_half, apply_gs=['w2'])
        f(i, g('qzeros'), 'zeros', to_half, apply_gs=['w2'])


class WeightScaleInv(Parameter):
    KEYS = '.weight_scale_inv', '.weight'

    # TODO: flag any operations crossing the quant blocks as illegal
    def __call__(self, f, g, i):
        f(i, g('weight_scale_inv'), 'scales', to_float, apply_gs=['w1', 'w3', 'w2'])
        f(i, g('weight'), 'weight', identity)


class CompressedWeight(Parameter):
    KEYS = '.weight_packed', '.weight_scale', '.weight_zero_point'

    def __call__(self, f, g, i):
        f(i, g('weight_packed'), 'qweight', pack_u4_row)
        f(i, g('weight_scale'), 'scales', to_half, apply_gs=['w2'])
        f(i, g('weight_zero_point'), 'zeros', to_half, apply_gs=['w2'])


class Mxfp4Weight(Parameter):
    KEYS = '.blocks', '.scales'

    def __call__(self, f, g, i):
        f(i, g('blocks'), 'weight', pack_u4_row)
        f(i, g('scales'), 'scales', identity, apply_gs=['w2'])


class Weight(Parameter):
    KEYS = '.weight',

    def __call__(self, f, g, i):
        f(i, g('weight'), 'weight', identity)


class Bias(Parameter):
    KEYS = '.bias',

    def __call__(self, f, g, i):
        f(i, g('bias'), 'bias', identity)


class PLora(Parameter):
    KEYS = '.Plora_A.weight', '.Plora_B.weight'

    def __call__(self, f, g, i):
        f(i, g('Plora_A.weight'), 'lora_a.weight', identity)
        f(i, g('Plora_B.weight'), 'lora_b.weight', identity)


def get_params(keys: List[str], bias=0):
    ps = []
    if PLora.take(keys):
        ps.append(PLora())
    if QuantWeightOnly.take(keys):
        ps.append(QuantWeightOnly())
    if WeightScaleInv.take(keys):
        ps.append(WeightScaleInv())
    if CompressedWeight.take(keys):
        ps.append(CompressedWeight())
    if Mxfp4Weight.take(keys):
        ps.append(Mxfp4Weight())
    if Weight.take(keys):
        ps.append(Weight())
    if bias and Bias.take(keys):
        ps.append(Bias())
    return ps
