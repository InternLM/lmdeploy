# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch

from .target_model.base import BaseOutputModel, BaseReader


def permute_v2(x: torch.Tensor, size_per_head: int = 128):
    """
        Contract: x.size(-1) is output dims
    """

    assert x.size(-1) > 1

    output_dims = x.size(-1)
    head_num = output_dims // size_per_head

    return x.view(-1, head_num, 2,
                  size_per_head // 2).transpose(2, 3).reshape(x.shape)


def merge_qkv_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int):
    """
        Contract: x.size(-1) is output dims
    """

    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkv = torch.cat(tuple(map(reshape, (q, k, v))), dim=-1)

    qkv = qkv.view(-1, qkv.size(-1) * tp)
    if q.dim() == 1:
        qkv.squeeze_()

    return qkv


def identity(x):
    return x


def transpose(x):
    return x.t() if x is not None else x


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


def pad_out_dims(x: torch.Tensor, dims: int):
    pad = dims - x.size(-1)
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, pad), 'constant', 0)


def pad_in_dims(x: torch.Tensor, dims: int):
    pad = dims - x.size(0)
    assert x.dim() == 2
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad), 'constant', 0)


class BaseExporter(ABC):

    _attn = 'layers.{0}.attention.{1}.{2}'
    _ffn = 'layers.{0}.feed_forward.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.tensor_para_size
        self.head_dim = model.model_config.size_per_head
        self.inter_size = model.model_config.inter_size

    def export_attn(self, idx: int, qkvo, kind: str, pack_fn=identity):
        if all(x is None for x in qkvo):
            return
        is_lora_a, is_lora_b = self.get_lora_flags(kind)
        q, k, v, o = map(transpose, qkvo)
        if self.model.permute_qk:
            q = permute_v2(q, self.head_dim)
            k = permute_v2(k, self.head_dim)
        qkv = merge_qkv_v2(q, k, v, self.tp)
        if o is None and q.dim() == 1:
            o = torch.zeros_like(q)
        qkv = pack_fn(qkv)
        o = pack_fn(o)
        self.model.save_split(qkv,
                              self._attn.format(idx, 'w_qkv', kind),
                              split_dim=-1,
                              copy=is_lora_a)
        self.model.save_split(o,
                              self._attn.format(idx, 'wo', kind),
                              split_dim=0,
                              copy=is_lora_b)

    def export_ffn(self, idx: int, w123, kind: str, pack_fn=identity, g=1):
        is_lora_a, is_lora_b = self.get_lora_flags(kind)
        w1, w2, w3 = map(transpose, w123)

        if not is_lora_a:
            w1 = pad_out_dims(w1, self.inter_size)
            w3 = pad_out_dims(w3, self.inter_size)
        if not is_lora_b:
            w2 = pad_in_dims(w2, self.inter_size // g)

        w1, w2, w3 = map(pack_fn, (w1, w2, w3))
        self.model.save_split(w1,
                              self._ffn.format(idx, 'w1', kind),
                              split_dim=-1,
                              copy=is_lora_a)
        self.model.save_split(w3,
                              self._ffn.format(idx, 'w3', kind),
                              split_dim=-1,
                              copy=is_lora_a)
        self.model.save_split(w2,
                              self._ffn.format(idx, 'w2', kind),
                              split_dim=0,
                              copy=is_lora_b)

    # split out dims -> copy A, split-out-dims B (qkv, w1, w3)
    # split  in dims -> split-in-dims A,  copy B (  o, w2)
    def get_lora_flags(self, kind: str):
        return ('lora_a' in kind, 'lora_b' in kind)

    @abstractmethod
    def export(self, r: BaseReader, idx: int):
        pass


class WeightExporter(BaseExporter):

    def export(self, r: BaseReader, i: int):
        self.export_attn(i, r.attn(i), 'weight')
        self.export_attn(i, r.attn_bias(i), 'bias')
        self.export_ffn(i, r.ffn(i), 'weight')


class LayerNormExporter(BaseExporter):

    def export(self, r: BaseReader, i: int):
        attn_norm = r.attn_norm(i)
        ffn_norm = r.ffn_norm(i)
        self.model.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.model.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')


class QuantWeightExporter(BaseExporter):

    def __init__(self, model: BaseOutputModel, pack_fn):
        super().__init__(model)
        self.pack_fn = pack_fn
        self.group_size = model.tm_config.group_size

    def export(self, r: BaseReader, i: int):

        def to_half(x: torch.Tensor):
            return x.to(torch.half)

        self.export_attn(i, r.attn(i), 'qweight', self.pack_fn)
        self.export_attn(i, r.attn_bias(i), 'bias', to_half)
        self.export_attn(i, r.attn_scale(i), 'scales', to_half)
        self.export_attn(i, r.attn_zero(i), 'zeros', to_half)
        self.export_ffn(i, r.ffn(i), 'qweight', self.pack_fn)
        self.export_ffn(i, r.ffn_scale(i), 'scales', to_half, self.group_size)
        self.export_ffn(i, r.ffn_zero(i), 'zeros', to_half, self.group_size)


class PLoraExporter(BaseExporter):

    def export_attn_lora_a(self, idx: int, ws, kind: str):
        is_lora_a, is_lora_b = self.get_lora_flags(kind)
        qkv, o = map(transpose, ws)
        self.model.save_split(qkv,
                              self._attn.format(idx, 'w_qkv', kind),
                              split_dim=-1,
                              copy=is_lora_a)
        self.model.save_split(o,
                              self._attn.format(idx, 'wo', kind),
                              split_dim=0,
                              copy=is_lora_b)

    def export(self, r: BaseReader, i: int):
        self.export_attn_lora_a(i, r.attn_lora_a(i), 'lora_a.weight')
        self.export_attn(i, r.attn_lora_b(i), 'lora_b.weight')
        self.export_ffn(i, r.ffn_lora_a(i), 'lora_a.weight')
        self.export_ffn(i, r.ffn_lora_b(i), 'lora_b.weight')


def get_exporter_factory(weight_type, lora_type):

    def get_exporters(model: BaseOutputModel):
        exporters = [LayerNormExporter(model)]

        if weight_type == 'int4':
            exporters.append(QuantWeightExporter(model, pack_u4_row))
        else:
            exporters.append(WeightExporter(model))

        if lora_type == 'plora':
            exporters.append(PLoraExporter(model))

        return exporters

    return get_exporters
