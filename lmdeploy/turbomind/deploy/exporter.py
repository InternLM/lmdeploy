# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import numpy as np
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


def get_qqq_perms(group_size: int, param: str):
    if param == 'qweight':
        perm = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                        4 * (i % 4), 4 * (i % 4) + 1, 4 * (i % 4) + 2,
                        4 * (i % 4) + 3
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm.extend([p + 256 * j for p in perm1])

        perm = np.array(perm)
        if group_size == -1:
            interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        perm = perm.reshape((-1, 8))[:, interleave].ravel()
        perm = torch.from_numpy(perm)
        return perm
    elif param == 'scales_zeros':
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm = torch.from_numpy(np.array(scale_perm))
        return scale_perm
    elif param == 'scales_channel':
        scale_perm_single = []
        for i in range(4):
            scale_perm_single.extend(
                [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        scale_perm_single = torch.from_numpy(np.array(scale_perm_single))
        return scale_perm_single
    else:
        raise ValueError('Invalid parameter!')


def qqq_pack(torch_tensor: torch.Tensor,
             param: str,
             group_size: int,
             tile: int = 16):
    assert torch_tensor is not None
    if param == 'qweight':
        assert torch_tensor.dim() == 2
        infeatures, outfeatures = torch_tensor.shape[0], torch_tensor.shape[1]
        _perm = get_qqq_perms(group_size, param)
        org_device = torch_tensor.device
        # permute and pack weight
        w = torch_tensor.reshape((
            infeatures // tile,
            tile,
            outfeatures // tile,
            tile,
        ))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((infeatures // tile, outfeatures * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        if group_size != -1 and group_size < infeatures:
            for i in range(8):
                q |= res[:, i::8] << 4 * i
        else:
            for i in range(8):
                q |= (res[:, i::8] & 0xF) << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(org_device)
        return q
    elif param == 'scales_zeros':
        # permute scales
        outfeatures = torch_tensor.shape[-1]
        _scale_perm = get_qqq_perms(group_size, param)
        s_group = torch_tensor.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        s_group = s_group.reshape((-1, outfeatures)).contiguous()
        return s_group
    elif param == 'scales_channel':
        outfeatures = torch_tensor.shape[-1]
        _scale_perm_single = get_qqq_perms(group_size, param)
        s_channel = torch_tensor.reshape(
            (-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s_channel = s_channel.reshape((-1, outfeatures)).contiguous()
        return s_channel
    else:
        raise ValueError('Invalid parameter!')


def qqq_unpack(torch_tensor: torch.Tensor,
               param: str,
               group_size: int,
               tile: int = 16,
               wbits: int = 4):
    assert torch_tensor is not None
    if param == 'qweight':
        assert torch_tensor.dim() == 2
        pack_factor = 32 // wbits
        infeatures = torch_tensor.shape[0] * tile
        outfeatures = torch_tensor.shape[1] * pack_factor // tile
        org_device = torch_tensor.device
        wf = torch.tensor(list(range(0, 32, 4)),
                          dtype=torch.int32).unsqueeze(0).to(org_device)
        # unpack weight
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(torch_tensor, 2).expand(-1, -1, 32 // wbits),
            wf.unsqueeze(0),
        )
        weight = torch.bitwise_and(weight, (2**wbits) - 1)
        weight = weight.reshape(weight.shape[0],
                                weight.shape[1] * weight.shape[2])

        _perm = get_qqq_perms(group_size, param)
        _perm_inv = torch.argsort(_perm)

        weight = weight.reshape(-1, _perm.numel())[:, _perm_inv]
        weight = weight.reshape((
            infeatures // tile,
            outfeatures // tile,
            tile,
            tile,
        ))
        weight = weight.permute((0, 2, 1, 3))
        weight = weight.reshape((infeatures, outfeatures))
        return weight
    elif param == 'scales_zeros':
        outfeatures = torch_tensor.shape[-1]
        _scale_perm = get_qqq_perms(group_size, param)
        _scale_perm_inv = torch.argsort(_scale_perm)
        s_group = torch_tensor.reshape(
            -1, len(_scale_perm))[:, _scale_perm_inv].reshape(-1, outfeatures)
        return s_group
    elif param == 'scales_channel':
        outfeatures = torch_tensor.shape[-1]
        _scale_perm_single = get_qqq_perms(group_size, param)
        _scale_perm_single_inv = torch.argsort(_scale_perm_single)
        s_channel = torch_tensor.reshape(
            -1, len(_scale_perm_single))[:, _scale_perm_single_inv].reshape(
                -1, outfeatures)
        return s_channel
    else:
        raise ValueError('Invalid parameter!')


def qqq_permute_qk(torch_tensor: torch.Tensor,
                   param: str,
                   group_size: int,
                   size_per_head: int = 128):
    unp_tensor = qqq_unpack(torch_tensor, param, group_size)
    dim = unp_tensor.shape[-1]
    n_heads = dim // size_per_head
    perm_tensor = unp_tensor.view(-1, n_heads, 2,
                                  dim // n_heads // 2).transpose(2, 3).reshape(
                                      -1, dim)
    p_tensor = qqq_pack(perm_tensor, param, group_size)
    return p_tensor


class BaseExporter(ABC):

    _attn = 'layers.{0}.attention.{1}.{2}'
    _ffn = 'layers.{0}.feed_forward.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.cfg.tensor_para_size
        self.head_dim = model.cfg.size_per_head
        self.inter_size = model.cfg.inter_size

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
        self.group_size = model.cfg.group_size

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


class QQQWeightExporter(BaseExporter):

    def __init__(self, model: BaseOutputModel):
        super().__init__(model)
        self.group_size = model.cfg.group_size

    def export_attn(self, idx: int, qkvo, kind: str, pack_fn=identity):
        if all(x is None for x in qkvo):
            return
        is_lora_a, is_lora_b = self.get_lora_flags(kind)
        # q, k, v, o = map(transpose, qkvo)
        q, k, v, o = qkvo
        # TODO(HandH1998): verify tp > 1
        if self.model.permute_qk:
            if kind == 'bias':
                q = permute_v2(q, self.head_dim)
                k = permute_v2(k, self.head_dim)
            else:
                q = qqq_permute_qk(q, kind, self.group_size)
                k = qqq_permute_qk(k, kind, self.group_size)
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
        w1, w2, w3 = w123

        # TODO(HandH1998): seems unnecessary
        # if not is_lora_a:
        #     w1 = pad_out_dims(w1, self.inter_size)
        #     w3 = pad_out_dims(w3, self.inter_size)
        # if not is_lora_b:
        #     w2 = pad_in_dims(w2, self.inter_size // g)

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

    def export(self, r: BaseReader, i: int):

        def to_half(x: torch.Tensor):
            return x.to(torch.half)

        def to_float(x: torch.Tensor):
            return x.to(torch.float)

        self.export_attn(i, r.attn(i), 'qweight')
        self.export_attn(i, r.attn_bias(i), 'bias', to_half)
        self.export_attn(i, r.attn_scale_channel(i), 'scales_channel',
                         to_float)
        self.export_ffn(i, r.ffn(i), 'qweight')
        self.export_ffn(i, r.ffn_scale_channel(i), 'scales_channel', to_float)
        if self.group_size != -1:
            self.export_attn(i, r.attn_scale_group(i), 'scales_zeros', to_half)
            self.export_ffn(i, r.ffn_scale_group(i), 'scales_zeros', to_half)


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


def get_exporter_factory(quantization, lora_type):

    def get_exporters(model: BaseOutputModel):
        exporters = [LayerNormExporter(model)]

        if quantization in ['awq', 'gptq']:
            exporters.append(QuantWeightExporter(model, pack_u4_row))
        elif quantization == 'qqq':
            exporters.append(QQQWeightExporter(model))
        else:
            exporters.append(WeightExporter(model))

        if lora_type == 'plora':
            exporters.append(PLoraExporter(model))

        return exporters

    return get_exporters
