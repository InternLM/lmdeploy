# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from functools import partial

import torch

from .parameter import get_params, identity
from .source_model.base import BaseReader
from .target_model.base import BaseOutputModel


def permute_v2(x: torch.Tensor, size_per_head: int = 128):
    """
        Contract: x.size(-1) is output dims
    """

    assert x.size(-1) > 1

    output_dims = x.size(-1)
    head_num = output_dims // size_per_head

    return x.view(-1, head_num, 2, size_per_head // 2).transpose(2, 3).reshape(x.shape)


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
    if x.dim() == 1:  # 1-dim object does not have input dim (e.g. bias)
        return x
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
        self.tp = model.mlp_tp_size
        # inter_sizes in config are padded and may be different from what's
        # in the weights
        self.inter_size = model.model_config.inter_size
        self.group_size = max(1, model.model_config.group_size)

    def _export(self, inter_size: int, fmt: str, idx: int, w123, kind: str, pack_fn, apply_gs=[], **kwargs):
        is_lora_a, is_lora_b = get_lora_flags(kind)
        w1, w2, w3 = map(transpose, w123)

        gs1 = self.group_size if 'w1' in apply_gs else 1
        w1 = pad_out_dims(w1, inter_size // gs1)

        gs3 = self.group_size if 'w3' in apply_gs else 1
        w3 = pad_out_dims(w3, inter_size // gs3)

        gs2 = self.group_size if 'w2' in apply_gs else 1
        w2 = pad_in_dims(w2, inter_size // gs2)

        w1, w2, w3 = map(pack_fn, (w1, w2, w3))
        self.model.save_split(w1, fmt.format(idx, 'w1', kind), split_dim=-1, split_num=self.tp, copy=is_lora_a)
        self.model.save_split(w3, fmt.format(idx, 'w3', kind), split_dim=-1, split_num=self.tp, copy=is_lora_a)
        self.model.save_split(w2, fmt.format(idx, 'w2', kind), split_dim=0, split_num=self.tp, copy=is_lora_b)

    def apply(self, i: int, r: BaseReader):
        if i >= len(self.inter_size) or not self.inter_size[i]:
            return
        keys = r.ffn(i, None)

        for e in get_params(keys):
            e(partial(self._export, self.inter_size[i], self._ffn), partial(r.ffn, i), i)


class MoeFfn(Ffn):
    """
    requires:
        r.moe_ffn_expert(e, i, kind)
        r.moe_ffn_gate(i)
        r.moe_ffn_shared_gate(i)
    """

    _moe_ffn_expert = 'layers.{0}.moe_ffn.experts.E.{1}.{2}'
    _moe_ffn_gate = 'layers.{0}.moe_ffn.gate.{1}'
    _moe_ffn_shared_gate = 'layers.{0}.moe_ffn.shared_gate.weight'

    def __init__(self, model: BaseOutputModel):
        super().__init__(model)
        self.expert_num = model.model_config.expert_num
        self.inter_size = model.model_config.expert_inter_size
        self.shared_gate = model.model_config.moe_shared_gate

    def apply(self, i: int, r: BaseReader):
        if i >= len(self.expert_num) or self.expert_num[i] == 0:
            return

        # Export expert weights with outer loop over experts (not params)
        # to ensure each expert's full weight set is grouped together
        for e in range(self.expert_num[i]):
            for p in get_params(r.moe_ffn_expert(), 1):
                fmt = self._moe_ffn_expert.replace('E', str(e))
                p(partial(self._export, self.inter_size, fmt), partial(r.moe_ffn_expert, e, i), i)

        # router
        gate = transpose(r.moe_ffn_gate(i, 'weight'))
        self.model.save_split(gate, self._moe_ffn_gate.format(i, 'weight'))
        bias = r.moe_ffn_gate(i, 'bias')
        if bias is not None:
            self.model.save_split(bias, self._moe_ffn_gate.format(i, 'bias'))

        # Export score_correction_bias for noaux_tc routing (GLM 4.7 Flash)
        correction_bias = getattr(r, 'moe_ffn_gate_correction_bias', None)
        if callable(correction_bias):
            correction = correction_bias(i)
            if correction is not None:
                self.model.save_split(correction, self._moe_ffn_gate.format(i, 'score_correction_bias'))

        if self.shared_gate:
            shared_gate = transpose(r.moe_ffn_shared_gate(i))
            self.model.save_split(shared_gate, self._moe_ffn_shared_gate.format(i))


class Attn(Module):
    """
    requires:
        r.attn(i, kind)
    """

    _attn = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.attn_tp_size
        self.head_dim = model.model_config.size_per_head
        self.attn_bias = model.model_config.attn_bias
        self.qk_norm = model.model_config.qk_norm
        self.attn_sink = model.model_config.attn_sink
        self.group_size = max(1, model.model_config.group_size)

    def _reorder_and_merge(self, qkvo, gs: int):
        q, k, v, o = qkvo
        # reorder output dim for tm's rotary embedding layout
        if self.model.permute_qk:
            if gs == 1:
                q = permute_v2(q, self.head_dim)
                k = permute_v2(k, self.head_dim)
            else:
                assert gs % self.head_dim == 0
        qkv = merge_qkv_v2(q, k, v, self.tp)
        # zero bias for `wo` when `w_qkv` has bias but `wo` doesn't
        if o is None and q.dim() == 1:
            o = torch.zeros_like(q)
        return qkv, o

    def _repeat_kv(self, qkvo, gs: int, kind: str):
        """Replicate kv."""
        q, k, v, o = qkvo
        head_dim = self.model.model_config.size_per_head // gs
        kv_head_num = self.model.model_config.kv_head_num // self.model.repeat_kv
        hidden_dim = self.model.model_config.hidden_units

        def _repeat(x):
            n = self.model.repeat_kv

            x = x.reshape(-1, kv_head_num, head_dim)
            x = x.repeat(1, 1, n)
            x = x.reshape(-1, kv_head_num * n * head_dim)

            return x

        k, v = map(_repeat, (k, v))

        if kind == 'bias':
            if o is None:
                o = torch.zeros(hidden_dim, dtype=q.dtype, device=q.device)
            q, k, v, o = map(torch.squeeze, (q, k, v, o))

        return (q, k, v, o)

    def _export(self, idx: int, qkvo, kind: str, pack_fn, apply_gs=[], **kwargs):
        if all(x is None for x in qkvo):
            return
        is_lora_a, is_lora_b = get_lora_flags(kind)
        assert not (is_lora_a or is_lora_b)

        qkvo = tuple(map(transpose, qkvo))

        gs = self.group_size if ('w1' in apply_gs) else 1

        if self.model.repeat_kv:
            qkvo = self._repeat_kv(qkvo, gs, kind)

        qkv, o = self._reorder_and_merge(qkvo, gs)

        self.model.save_split(pack_fn(qkv),
                              self._attn.format(idx, 'w_qkv', kind),
                              split_dim=-1,
                              split_num=self.tp,
                              copy=is_lora_a)
        self.model.save_split(pack_fn(o),
                              self._attn.format(idx, 'wo', kind),
                              split_dim=0,
                              split_num=self.tp,
                              copy=is_lora_b)

    def apply(self, i: int, r: BaseReader):
        for e in get_params(r.attn(i, None), bias=self.attn_bias):
            e(self._export, partial(r.attn, i), i)
        if self.qk_norm:
            q, k = r.qk_norm(i)
            if self.model.permute_qk:
                q = permute_v2(q, self.head_dim)
                k = permute_v2(k, self.head_dim)
            self.model.save_split(q, self._attn.format(i, 'q_norm', '')[:-1])
            self.model.save_split(k, self._attn.format(i, 'k_norm', '')[:-1])
        if self.attn_sink:
            sinks = r.attn_sinks(i)
            self.model.save_split(sinks, self._attn.format(i, 'sinks', '')[:-1], split_dim=-1, split_num=self.tp)


class MLA(Module):
    """
    requires:
        r.mla(i, kind)
        r.mla_norm(i)
    """

    _mla = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model

    def _export(self, idx: int, xs, kind: str, pack_fn, **kwargs):
        if all(x is None for x in xs):
            return
        q_a, q_b, q, kv_a, kv_b, o = xs

        cfg = self.model.model_config
        head_num = cfg.head_num
        kv_lora_rank = cfg.kv_lora_rank
        qk_rope_dim = cfg.qk_rope_dim
        size_per_head = cfg.size_per_head
        v_head_dim = cfg.v_head_dim

        # ========== MLA Weight Folding for Dimension Mismatch ==========
        # When kv_lora_rank != qk_nope_dim (e.g., GLM 4.7 Flash: 512 != 512+64=576),
        # fold the kc/vc compression/decompression BMMs into q_b_proj/o_proj weights
        # at conversion time to avoid runtime overhead.
        if kind == 'weight' and kv_lora_rank and q is None and q_b is not None and kv_b is not None and o is not None:
            if not (torch.is_floating_point(q_b) and torch.is_floating_point(kv_b) and torch.is_floating_point(o)):
                raise ValueError('MLA weight folding requires floating-point attention weights.')

            orig_q_head_dim = q_b.size(0) // head_num
            orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
            orig_kv_dim_total = kv_b.size(0) // head_num
            orig_v_head_dim = o.size(1) // head_num
            actual_orig_qk_nope_dim = orig_kv_dim_total - orig_v_head_dim

            if abs(orig_qk_nope_dim - actual_orig_qk_nope_dim) > 1:
                raise ValueError(f'Dimension mismatch: inferred qk_nope from q_b ({orig_qk_nope_dim}) != '
                                 f'inferred from kv_b ({actual_orig_qk_nope_dim})')

            orig_qk_nope_dim = actual_orig_qk_nope_dim
            target_nope_dim = size_per_head - qk_rope_dim
            target_v_head_dim = v_head_dim

            if orig_qk_nope_dim != target_nope_dim or orig_v_head_dim != target_v_head_dim:
                if target_nope_dim != kv_lora_rank or target_v_head_dim != kv_lora_rank:
                    raise ValueError(f'MLA folding expects v_head_dim and nope_dim to equal kv_lora_rank, '
                                     f'got nope={target_nope_dim}, v_head={target_v_head_dim}, rank={kv_lora_rank}')

                if kv_b.size(1) != kv_lora_rank:
                    raise ValueError(f'kv_b_proj second dim must equal kv_lora_rank for MLA folding, '
                                     f'got {kv_b.size(1)} != {kv_lora_rank}')

                # Split kv_b into kc and vc
                kv_b_per_head = kv_b.reshape(head_num, orig_qk_nope_dim + orig_v_head_dim, kv_lora_rank)
                kc_w = kv_b_per_head[:, :orig_qk_nope_dim, :]
                vc_w = kv_b_per_head[:, orig_qk_nope_dim:, :]

                # Fold kc into q_b_proj
                q_b_per_head = q_b.reshape(head_num, orig_q_head_dim, q_b.size(1))
                q_nope_w = q_b_per_head[:, :orig_qk_nope_dim, :]
                q_rope_w = q_b_per_head[:, orig_qk_nope_dim:, :]
                q_nope_expanded = torch.bmm(kc_w.transpose(1, 2), q_nope_w)
                q_b_folded = torch.cat([q_nope_expanded, q_rope_w], dim=1)
                q_b = q_b_folded.reshape(head_num * size_per_head, q_b.size(1))

                # Fold vc into o_proj
                o_per_head = o.reshape(o.size(0), head_num, orig_v_head_dim)
                o_folded = torch.bmm(o_per_head.permute(1, 0, 2), vc_w)
                o = o_folded.permute(1, 0, 2).reshape(o.size(0), head_num * kv_lora_rank)

                # Set kv_b to identity (kc/vc are now absorbed)
                eye = torch.eye(kv_lora_rank, dtype=kv_b.dtype, device=kv_b.device)
                kv_b = torch.cat([eye, eye], dim=0).repeat(head_num, 1)
        # ========== End MLA Weight Folding ==========

        # Transpose after folding
        q_a, q_b, q, kv_a, kv_b, o = map(transpose, (q_a, q_b, q, kv_a, kv_b, o))

        if q is not None:
            q_b = q

        # Pad o_proj to size_per_head if present
        if o is not None:
            o = o.reshape(head_num, v_head_dim, -1)
            o = torch.nn.functional.pad(o, (0, 0, 0, size_per_head - v_head_dim, 0, 0))
            o = o.view(head_num * size_per_head, cfg.hidden_units)

        tp = self.model.attn_tp_size

        # Export MLA weights (handle None for folded-away tensors)
        if q_a is not None:
            self.model.save_split(pack_fn(q_a), self._mla.format(idx, 'q_a_proj', kind))
        q_b_name = 'q_proj' if q_a is None else 'q_b_proj'
        if q_b is not None:
            self.model.save_split(pack_fn(q_b), self._mla.format(idx, q_b_name, kind), split_dim=-1, split_num=tp)
        if kv_a is not None:
            self.model.save_split(pack_fn(kv_a), self._mla.format(idx, 'kv_a_proj', kind))
        if kv_b is not None:
            self.model.save_split(pack_fn(kv_b), self._mla.format(idx, 'kv_b_proj', kind), split_dim=-1, split_num=tp)
        if o is not None:
            self.model.save_split(pack_fn(o), self._mla.format(idx, 'wo', kind), split_dim=0, split_num=tp)

    _layernorm = 'layers.{0}.attention.{1}_a_layernorm'

    def apply(self, i: int, r: BaseReader):

        for f in get_params(r.attn(i, None), bias=False):
            f(self._export, partial(r.mla, i), i)

        q, k = r.mla_norm(i)
        if q is not None:
            self.model.save_split(q, self._layernorm.format(i, 'q'))
        self.model.save_split(k, self._layernorm.format(i, 'kv'))


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

        def pad_weight(tensor: torch.Tensor, tp: int):
            pad_size = None
            vocab_size = self.model.model_config.vocab_size
            if vocab_size % tp != 0:
                pad_size = (vocab_size + tp - 1) // tp * tp - vocab_size
            if pad_size is None:
                return tensor
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), 'constant', 0)

        tp = self.model.attn_tp_size * self.model.attn_cp_size
        if emb is not None:
            emb = pad_weight(emb, tp=tp)
            self.model.save_split(emb, 'tok_embeddings.weight', split_dim=1, split_num=tp)
        if norm_weight is not None:
            self.model.export_weight(norm_weight, 'norm.weight')
        if output_weight is not None:
            output_weight = pad_weight(output_weight, tp=tp)
            # transpose
            self.model.save_split(output_weight.t(), 'output.weight', split_dim=1, split_num=tp)


class Transformer:

    def __init__(self, model: BaseOutputModel):
        self.model = model
        modules = [LayerNorm]
        if model.model_config.kv_lora_rank:
            modules.append(MLA)
        else:
            modules.append(Attn)
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
