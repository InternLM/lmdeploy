# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

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
    assert x.dtype == torch.uint8, f'x.dtype: {x.dtype}'
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


def generate_zero_point(scales):
    """Synthesize symmetric int4 zero-points from exported scale shapes."""
    return tuple(torch.full(s.shape, 8, dtype=torch.uint8) for s in scales)


class Parameter:
    KEY = ()

    @classmethod
    def take(cls, keys: list[str]):
        if not any(k.endswith(cls.KEYS[0]) for k in keys):
            return False
        xs = []
        for k in keys:
            if any(k.endswith(p) for p in cls.KEYS):
                xs.append(k)
        for x in xs:
            keys.remove(x)
        return xs

    @abstractmethod
    def __call__(cls, f, g, i):
        pass


class QuantWeightOnly(Parameter):
    AWQ_KEYS = '.qweight', '.scales', '.qzeros'
    COMPRESSED_KEYS = '.weight_packed', '.weight_scale', '.weight_zero_point'
    KEYS = AWQ_KEYS + COMPRESSED_KEYS

    @classmethod
    def take(cls, keys: list[str]):
        if any(k.endswith(cls.AWQ_KEYS[0]) for k in keys):
            suffixes = cls.AWQ_KEYS
        elif any(k.endswith(cls.COMPRESSED_KEYS[0]) for k in keys):
            suffixes = cls.COMPRESSED_KEYS
        else:
            return False

        xs = []
        for k in keys:
            if any(k.endswith(p) for p in suffixes):
                xs.append(k)
        for x in xs:
            keys.remove(x)
        return xs

    def __init__(self, xs):
        self.compressed_tensors = any(key.endswith(self.COMPRESSED_KEYS[0]) for key in xs)
        self.has_zero_point = any(key.endswith(self.COMPRESSED_KEYS[2]) for key in xs)

    def _get(self, g, kind: str):
        if not self.compressed_tensors:
            return g(kind)

        mapping = {
            'qweight': 'weight_packed',
            'scales': 'weight_scale',
            'qzeros': 'weight_zero_point',
        }
        return g(mapping[kind])

    def __call__(self, f, g, i):
        f(i, self._get(g, 'qweight'), 'qweight', pack_u4_row)
        scales = self._get(g, 'scales')
        f(i, scales, 'scales', to_half, apply_gs=['w2'])
        if self.compressed_tensors and not self.has_zero_point:
            zeros = generate_zero_point(scales)
        else:
            zeros = self._get(g, 'qzeros')
        f(i, zeros, 'zeros', to_half, apply_gs=['w2'])


class WeightScaleInv(Parameter):
    KEYS = '.weight_scale_inv', '.weight'

    # TODO: flag any operations crossing the quant blocks as illegal
    def __call__(self, f, g, i):
        f(i, g('weight_scale_inv'), 'scales', to_float, apply_gs=['w1', 'w3', 'w2'])
        f(i, g('weight'), 'weight', identity)


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


def get_params(keys: list[str], bias=0):
    ps = []
    if PLora.take(keys):
        ps.append(PLora())
    xs = QuantWeightOnly.take(keys)
    if xs:
        ps.append(QuantWeightOnly(xs))
    if WeightScaleInv.take(keys):
        ps.append(WeightScaleInv())
    if Mxfp4Weight.take(keys):
        ps.append(Mxfp4Weight())
    if Weight.take(keys):
        ps.append(Weight())
    if bias and Bias.take(keys):
        ps.append(Bias())
    return ps
