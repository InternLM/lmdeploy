# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch

from .target_model.base import BaseOutputModel, BaseReader, TurbomindModelConfig
from functools import partial


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

    # _attn = 'layers.{0}.attention.{1}.{2}'
    # _ffn = 'layers.{0}.feed_forward.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model

    #     self.tp = model.cfg.tensor_para_size
    #     self.head_dim = model.cfg.size_per_head
    #     self.inter_size = model.cfg.inter_size
    #     self.expert_num = model.cfg.expert_num

    # def export_attn(self, idx: int, qkvo, kind: str, pack_fn=identity):
    #     if all(x is None for x in qkvo):
    #         return
    #     is_lora_a, is_lora_b = self.get_lora_flags(kind)
    #     q, k, v, o = map(transpose, qkvo)
    #     if self.model.permute_qk:
    #         q = permute_v2(q, self.head_dim)
    #         k = permute_v2(k, self.head_dim)
    #     qkv = merge_qkv_v2(q, k, v, self.tp)
    #     if o is None and q.dim() == 1:
    #         o = torch.zeros_like(q)
    #     qkv = pack_fn(qkv)
    #     o = pack_fn(o)
    #     self.model.save_split(qkv,
    #                           self._attn.format(idx, 'w_qkv', kind),
    #                           split_dim=-1,
    #                           copy=is_lora_a)
    #     self.model.save_split(o,
    #                           self._attn.format(idx, 'wo', kind),
    #                           split_dim=0,
    #                           copy=is_lora_b)

    # def _export_ffn(self, fmt: str, idx: int, w123, kind: str, pack_fn=identity, g=1):
    #     is_lora_a, is_lora_b = self.get_lora_flags(kind)
    #     w1, w2, w3 = map(transpose, w123)

    #     if not is_lora_a:
    #         w1 = pad_out_dims(w1, self.inter_size)
    #         w3 = pad_out_dims(w3, self.inter_size)
    #     if not is_lora_b:
    #         w2 = pad_in_dims(w2, self.inter_size // g)

    #     w1, w2, w3 = map(pack_fn, (w1, w2, w3))
    #     self.model.save_split(w1,
    #                           fmt.format(idx, 'w1', kind),
    #                           split_dim=-1,
    #                           copy=is_lora_a)
    #     self.model.save_split(w3,
    #                           fmt.format(idx, 'w3', kind),
    #                           split_dim=-1,
    #                           copy=is_lora_a)
    #     self.model.save_split(w2,
    #                           fmt.format(idx, 'w2', kind),
    #                           split_dim=0,
    #                           copy=is_lora_b)
        
    # def export_ffn(self, *args, **kwargs):
    #     self._export_ffn(self._ffn, *args, **kwargs)

    # def get_lora_flags(self, kind: str):
    #     return ('lora_a' in kind, 'lora_b' in kind)

    @abstractmethod
    def export(self, r: BaseReader, idx: int):
        pass


# class WeightExporter(BaseExporter):

#     def export(self, r: BaseReader, i: int):
#         print (r.get_attn())
#         assert 0
#         self.export_attn(i, r.attn(i), 'weight')
#         self.export_attn(i, r.attn_bias(i), 'bias')
#         # self.export_ffn(i, r.ffn(i), 'weight')

class LayerNorm(BaseExporter):

    def export(self, i: int, r: BaseReader):
        attn_norm = r.attn_norm(i)
        ffn_norm = r.ffn_norm(i)
        self.model.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.model.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')


# class QuantWeightExporter(BaseExporter):

#     def __init__(self, model: BaseOutputModel, pack_fn):
#         super().__init__(model)
#         self.pack_fn = pack_fn
#         self.group_size = model.cfg.group_size

#     def export(self, r: BaseReader, i: int):

#         def to_half(x: torch.Tensor):
#             return x.to(torch.half)

#         self.export_attn(i, r.attn(i), 'qweight', self.pack_fn)
#         self.export_attn(i, r.attn_bias(i), 'bias', to_half)
#         self.export_attn(i, r.attn_scale(i), 'scales', to_half)
#         self.export_attn(i, r.attn_zero(i), 'zeros', to_half)
#         self.export_ffn(i, r.ffn(i), 'qweight', self.pack_fn)
#         self.export_ffn(i, r.ffn_scale(i), 'scales', to_half, self.group_size)
#         self.export_ffn(i, r.ffn_zero(i), 'zeros', to_half, self.group_size)


# class PLoraExporter(BaseExporter):

#     def export_attn_lora_a(self, idx: int, ws, kind: str):
#         is_lora_a, is_lora_b = self.get_lora_flags(kind)
#         qkv, o = map(transpose, ws)
#         self.model.save_split(qkv,
#                               self._attn.format(idx, 'w_qkv', kind),
#                               split_dim=-1,
#                               copy=is_lora_a)
#         self.model.save_split(o,
#                               self._attn.format(idx, 'wo', kind),
#                               split_dim=0,
#                               copy=is_lora_b)

#     def export(self, r: BaseReader, i: int):
#         self.export_attn_lora_a(i, r.attn_lora_a(i), 'lora_a.weight')
#         self.export_attn(i, r.attn_lora_b(i), 'lora_b.weight')
#         self.export_ffn(i, r.ffn_lora_a(i), 'lora_a.weight')
#         self.export_ffn(i, r.ffn_lora_b(i), 'lora_b.weight')


# class MoeExporter(BaseExporter):

#     _moe_ffn_expert = 'layers.[0].moe_ffn.experts.{0}.[1].[2]'
#     _moe_ffn_gate = 'layers.{0}.moe_ffn.gate.{1}'

#     def __init__(self, model: BaseOutputModel):
#         super().__init__(model)
#         self.expert_num = model.cfg.expert_num

#     def export_moe_ffn(self, e: int, *args, **kwargs):
#         # write expert id and replace [] with {}
#         fmt = self._moe_ffn_expert.format(e).replace('[', '{').replace(']','}')
#         self._export_ffn(fmt, *args, **kwargs)

#     def export(self, r: BaseReader, i: int):
#         for e in range(self.expert_num):
#             self.export_moe_ffn(e, i, r.moe_ffn_expert(e, i), 'weight')

#         gate = transpose(r.moe_ffn_gate(i))
#         self.model.save_split(gate, self._moe_ffn_gate.format(i, 'weight'))


# def _get_exporter_factory(weight_type, lora_type, is_moe):

#     print (weight_type, lora_type, is_moe)

#     def get_exporters(model: BaseOutputModel):
#         exporters = [LayerNormExporter(model)]

#         if weight_type == 'int4':
#             exporters.append(QuantWeightExporter(model, pack_u4_row))
#         else:
#             exporters.append(WeightExporter(model))

#         if is_moe:
#             exporters.append(MoeExporter(model))

#         if lora_type == 'plora':
#             exporters.append(PLoraExporter(model))

#         return exporters

#     return get_exporters


def to_half(x: torch.Tensor):
    return x.to(torch.half)


def export_quant_weight_only(f, g, i):
    f(i, g('qweight'), 'qweight', pack_u4_row)
    f(i, g('scales'), 'scales', to_half, apply_gs=True)
    f(i, g('qzeros'), 'zeros', to_half, apply_gs=True)


def export_weight(f, g, i):
    f(i, g('weight'), 'weight', identity)


def export_bias(f, g, i):
    f(i, g('bias'), 'bias', identity)


def export_plora(f, g, i):
    """
    feed_forward.\w+.([\w|.]+)
    """
    f(i, g('Plora_A.weight'), 'lora_a.weight', identity)
    f(i, g('Plora_B.weight'), 'lora_b.weight', identity)


def get_weight_exporters(kinds, bias=0):
    e = []
    if 'qweight' in kinds:
        e.append(export_quant_weight_only)
    if 'weight' in kinds:
        e.append(export_weight)
    if bias and 'bias' in kinds:
        e.append(export_bias)
    if 'Plora_A.weight' in kinds:
        e.append(export_plora)
    return e


# split out dims -> copy A, split-out-dims B (qkv, w1, w3)
# split  in dims -> split-in-dims A,  copy B (  o, w2)
def get_lora_flags(kind: str):
    return ('lora_a' in kind, 'lora_b' in kind)


class Ffn:
    """
    requires
        r.ffn(i, kind)
    """

    _ffn = 'layers.{0}.feed_forward.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.cfg.tensor_para_size
        self.inter_size = model.cfg.inter_size
        self.group_size = max(1, model.cfg.group_size)

    def _export(self, fmt: str, idx: int, w123, kind: str, pack_fn=identity, apply_gs=False):
        is_lora_a, is_lora_b = get_lora_flags(kind)
        w1, w2, w3 = map(transpose, w123)

        if not is_lora_a:
            w1 = pad_out_dims(w1, self.inter_size)
            w3 = pad_out_dims(w3, self.inter_size)
        if not is_lora_b:
            group_size = self.group_size if apply_gs else 1
            w2 = pad_in_dims(w2, self.inter_size // group_size)

        w1, w2, w3 = map(pack_fn, (w1, w2, w3))
        self.model.save_split(w1,
                              fmt.format(idx, 'w1', kind),
                              split_dim=-1,
                              copy=is_lora_a)
        self.model.save_split(w3,
                              fmt.format(idx, 'w3', kind),
                              split_dim=-1,
                              copy=is_lora_a)
        self.model.save_split(w2,
                              fmt.format(idx, 'w2', kind),
                              split_dim=0,
                              copy=is_lora_b)
        
    def export(self, i: int, r: BaseReader):
        for e in get_weight_exporters(r.ffn(i, None)):
            e(partial(self._export, self._ffn), partial(r.ffn, i), i)


class MoeFfn(Ffn):
    """
    requires
        r.moe_ffn_expert(e, i, kind)
        r.moe_ffn_gate(i)
    """

    _moe_ffn_expert = 'layers.[0].moe_ffn.experts.{0}.[1].[2]'
    _moe_ffn_gate = 'layers.{0}.moe_ffn.gate.{1}'

    def __init__(self, model: BaseOutputModel):
        super().__init__(model)
        self.expert_num = model.cfg.expert_num

    def export(self, i: int, r: BaseReader):
        for compose in get_weight_exporters(r.moe_ffn_expert()):
            for e in range(self.expert_num):
                fmt = self._moe_ffn_expert.format(e).replace('[', '{').replace(']','}')
                compose(partial(self._export, fmt), partial(r.moe_ffn_expert, e, i), i)

        gate = transpose(r.moe_ffn_gate(i))
        self.model.save_split(gate, self._moe_ffn_gate.format(i, 'weight'))


class Attn:

    """
    requires
        r.attn(i, kind)
    """

    _attn = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.cfg.tensor_para_size
        self.head_dim = model.cfg.size_per_head

    def _reorder_and_merge(self, qkvo):
        q, k, v, o = map(transpose, qkvo)
        # reorder output dim for tm's rotary embedding layout
        if self.model.permute_qk:
            q = permute_v2(q, self.head_dim)
            k = permute_v2(k, self.head_dim)
        qkv = merge_qkv_v2(q, k, v, self.tp)
        # zero bias for `wo` when `w_qkv` has bias but `wo` doesn't
        if o is None and q.dim() == 1:
            o = torch.zeros_like(q)
        return qkv, o

    def _export(self, idx: int, qkvo, kind: str, pack_fn=identity, **kwargs):
        if all(x is None for x in qkvo):
            return
        is_lora_a, is_lora_b = get_lora_flags(kind)
        if is_lora_a:
            qkv, o = map(transpose, qkvo)
        else:
            qkv, o = self._reorder_and_merge(qkvo)
        self.model.save_split(pack_fn(qkv),
                              self._attn.format(idx, 'w_qkv', kind),
                              split_dim=-1,
                              copy=is_lora_a)
        self.model.save_split(pack_fn(o),
                              self._attn.format(idx, 'wo', kind),
                              split_dim=0,
                              copy=is_lora_b)

    def export(self, i: int, r: BaseReader):
        for e in get_weight_exporters(r.attn(i, None), bias=1):
            e(self._export, partial(r.attn, i), i)


class Misc(BaseExporter):
    """
    requires
        r.tok_embeddings()
        r.norm_weight()
        r.output_weight()
    """
    def export(self, i: int, r: BaseReader):
        """Export embedding, norm, output weight."""
        emb = r.tok_embeddings()
        norm_weight = r.norm_weight()
        output_weight = r.output_weight()

        def pad_weight(tensor):
            pad_size = None
            vocab_size = self.model.cfg.vocab_size
            tp = self.model.cfg.tensor_para_size
            if vocab_size % tp != 0:
                pad_size = (vocab_size + tp - 1) // tp * tp - vocab_size

            if pad_size is None:
                return tensor
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size),
                                           'constant', 0)

        if emb is not None:
            emb = pad_weight(emb)
            self.model.export_weight(emb, 'tok_embeddings.weight')
        if norm_weight is not None:
            self.model.export_weight(norm_weight, 'norm.weight')
        if output_weight is not None:
            output_weight = pad_weight(output_weight)
            self.model.export_weight(output_weight, 'output.weight')


class Transformer:
    def __init__(self, model: BaseOutputModel):
        self.model = model
        ffn = MoeFfn if model.cfg.expert_num else Ffn
        modules = [Attn, LayerNorm, ffn]
        self.modules = [c(model) for c in modules]
        self.misc = Misc(model)

    def __call__(self, i: int, r: BaseReader):
        if i >= 0:
            for m in self.modules:
                m.export(i, r)
            return 1
        else:
            self.misc.export(i, r)

def get_exporter_factory(*args):
    return Transformer
