# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from functools import partial

import torch

from .parameter import get_params
from .source_model.base import BaseReader
from .target_model.base import BaseOutputModel


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


def transpose(x):
    return x.t() if x is not None else x


def pad_out_dims(x: torch.Tensor, dims: int):
    pad = dims - x.size(-1)
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, pad), 'constant', 0)


def pad_in_dims(x: torch.Tensor, dims: int):
    pad = dims - x.size(0)
    assert x.dim() == 2
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad), 'constant', 0)


# split out dims -> copy A, split-out-dims B (qkv, w1, w3)
# split  in dims -> split-in-dims A,  copy B (  o, w2)
def get_lora_flags(kind: str):
    return ('lora_a' in kind, 'lora_b' in kind)


class Module(ABC):

    def __init__(self, model: BaseOutputModel):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractmethod
    def apply(self, idx: int, r: BaseReader):
        pass


class LayerNorm(Module):

    def apply(self, i: int, r: BaseReader):
        attn_norm = r.attn_norm(i)
        ffn_norm = r.ffn_norm(i)
        self.model.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.model.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')


class Ffn(Module):
    """
    requires:
        r.ffn(i, kind)
    """

    _ffn = 'layers.{0}.feed_forward.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.tensor_para_size
        self.inter_size = model.model_config.inter_size
        self.group_size = max(1, model.model_config.group_size)

    def _export(self,
                fmt: str,
                idx: int,
                w123,
                kind: str,
                pack_fn,
                apply_gs=False):
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

    def apply(self, i: int, r: BaseReader):
        for e in get_params(r.ffn(i, None)):
            e(partial(self._export, self._ffn), partial(r.ffn, i), i)


class MoeFfn(Ffn):
    """
    requires:
        r.moe_ffn_expert(e, i, kind)
        r.moe_ffn_gate(i)
        r.moe_ffn_shared_gate(i)
    """

    _moe_ffn_expert = 'layers.{0}.moe_ffn.experts.E.{1}.{2}'
    _moe_ffn_gate = 'layers.{0}.moe_ffn.gate.weight'
    _moe_ffn_shared_gate = 'layers.{0}.moe_ffn.shared_gate.weight'

    def __init__(self, model: BaseOutputModel):
        super().__init__(model)
        self.expert_num = model.model_config.expert_num
        self.inter_size = model.model_config.expert_inter_size
        self.shared_gate = model.model_config.moe_shared_gate

    def apply(self, i: int, r: BaseReader):
        for p in get_params(r.moe_ffn_expert()):
            for e in range(self.expert_num):
                fmt = self._moe_ffn_expert.replace('E', str(e))
                p(partial(self._export, fmt), partial(r.moe_ffn_expert, e, i),
                  i)

        gate = transpose(r.moe_ffn_gate(i))
        self.model.save_split(gate, self._moe_ffn_gate.format(i))

        if self.shared_gate:
            shared_gate = transpose(r.moe_ffn_shared_gate(i))
            # print(shared_gate)
            self.model.save_split(shared_gate,
                                  self._moe_ffn_shared_gate.format(i))


class Attn(Module):
    """
    requires:
        r.attn(i, kind)
    """

    _attn = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.tensor_para_size
        self.head_dim = model.model_config.size_per_head
        self.attn_bias = model.model_config.attn_bias

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

    def _export(self, idx: int, qkvo, kind: str, pack_fn, **kwargs):
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

    def apply(self, i: int, r: BaseReader):
        for e in get_params(r.attn(i, None), bias=self.attn_bias):
            e(self._export, partial(r.attn, i), i)


class Misc(Module):
    """
    requires:
        r.tok_embeddings()
        r.norm_weight()
        r.output_weight()
    """

    def apply(self, i: int, r: BaseReader):
        """Export embedding, norm, output weight."""
        emb = r.tok_embeddings()
        norm_weight = r.norm_weight()
        output_weight = r.output_weight()

        def pad_weight(tensor):
            pad_size = None
            vocab_size = self.model.model_config.vocab_size
            tp = self.model.tensor_para_size
            if vocab_size % tp != 0:
                pad_size = (vocab_size + tp - 1) // tp * tp - vocab_size

            if pad_size is None:
                return tensor
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size),
                                           'constant', 0)

        if emb is not None:
            emb = pad_weight(emb)
            self.model.save_split(emb, 'tok_embeddings.weight', split_dim=1)
        if norm_weight is not None:
            self.model.export_weight(norm_weight, 'norm.weight')
        if output_weight is not None:
            output_weight = pad_weight(output_weight)
            self.model.save_split(output_weight, 'output.weight', split_dim=0)


class Transformer:

    def __init__(self, model: BaseOutputModel):
        self.model = model
        modules = [Attn, LayerNorm]
        if model.model_config.inter_size:
            modules.append(Ffn)
        if model.model_config.expert_num:
            modules.append(MoeFfn)
        self.modules = [c(model) for c in modules]
        self.misc = Misc(model)

    def __call__(self, i: int, r: BaseReader):
        if i >= 0:
            for m in self.modules:
                m(i, r)
            return 1
        else:
            self.misc(i, r)
