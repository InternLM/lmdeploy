# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys

import torch

import lmdeploy

from ..source_model.base import BaseInputModel, BaseReader
from .base import (OUTPUT_MODELS, BaseOutputModel, TurbomindModelConfig,
                   merge_qkv, permute)

# import _turbomind as _tm
# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402


def transpose_qk_s4(src: torch.Tensor, group_size):
    assert src.is_contiguous()
    dst = torch.zeros_like(src)
    _tm.transpose_qk_s4_k_m8(src, dst,
                             src.size(-1) * 8, src.size(0), group_size)
    return dst


def fuse_w1_w3_s4(w1_qw: torch.Tensor, w1_qz: torch.Tensor, w1_s: torch.Tensor,
                  w3_qw: torch.Tensor, w3_qz: torch.Tensor,
                  w3_s: torch.Tensor):

    def fuse(a: torch.Tensor, b: torch.Tensor):
        ab = torch.cat((a, b)).contiguous()
        _ab = torch.zeros_like(ab)
        _tm.fuse_w1_w3_s4_k_m8(ab, _ab, a.size(-1) * 8, a.size(0))
        return _ab.view(a.size(0), -1)

    w13_qw = fuse(w1_qw, w3_qw)
    w13_qz = fuse(w1_qz, w3_qz)

    w13_s = torch.cat((w1_s, w3_s)).view(2, w1_s.size(0), -1)
    w13_s = w13_s.permute(1, 2, 0).contiguous().view(w1_s.size(0), -1)

    return w13_qw, w13_qz, w13_s


def convert_s4(qw: torch.Tensor, qz: torch.Tensor, s: torch.Tensor,
               group_size: int):
    assert qw.is_contiguous()
    assert qz.is_contiguous()
    assert s.is_contiguous()
    _qw = torch.zeros_like(qw)
    _sz = torch.zeros_like(s, dtype=torch.int32)  # half2
    _ws = torch.zeros_like(s)
    _tm.convert_s4_k_m8(_qw, _sz, _ws, qw, s, qz,
                        qw.size(-1) * 8, qw.size(0), group_size)
    return _qw, _sz


def tp_m_s4(x: torch.Tensor, tp: int):
    return x.view(x.size(0) // 32, tp, -1, 128).permute(0, 2, 3,
                                                        1).contiguous()


def get_cuda_tensor(tensors):
    """Get cuda tensor."""
    result = map(lambda x: x.cuda() if x is not None else x, tensors)
    return (*result, )


def transpose_tensor(tensors):
    """Get cuda tensor."""
    result = map(lambda x: x.cuda().t() if x is not None else x, tensors)
    return (*result, )


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


@OUTPUT_MODELS.register_module(name='w4')
class TurbomindW4Model(BaseOutputModel):
    """Export to turbomind w4a16 format."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        super().__init__(input_model, cfg, to_file, out_dir)

    def get_config(self, cfg: TurbomindModelConfig):
        """Get turbomind config."""
        final_cfg = super().get_config(cfg).__dict__

        # attn_bias, inter_size
        visit = False
        attn_bias = 0
        for bin in self.input_model.bins():
            for i in range(bin.start_layer_id, bin.end_layer_id):
                visit = True
                w1s, _, _ = bin.ffn_scale(i)
                inter_size = w1s.shape[0]
                qb, _, _, _ = bin.attn_bias(i)
                if qb is not None:
                    attn_bias = 1
                break
            if visit:
                break
        final_cfg.update(dict(attn_bias=attn_bias, inter_size=inter_size))
        return TurbomindModelConfig.from_dict(final_cfg)

    def export_transformer_block(self, bin: BaseReader, i: int):
        """Export transformer layer i."""
        tp = self.cfg.tensor_para_size
        size_per_head = self.cfg.size_per_head
        # attn
        q_w, k_w, v_w, o_w = transpose_tensor(bin.attn(i))
        q_z, k_z, v_z, o_z = transpose_tensor(bin.attn_zero(i))
        q_s, k_s, v_s, o_s = transpose_tensor(bin.attn_scale(i))

        # print(q_w.shape, k_w.shape, q_z.shape, k_z.shape, q_s.shape, k_s.shape)

        if self.permute_qk:
            q_w = permute(q_w, size_per_head)
            k_w = permute(k_w, size_per_head)
            q_z = permute(q_z, size_per_head)
            k_z = permute(k_z, size_per_head)
            q_s = permute(q_s, size_per_head)
            k_s = permute(k_s, size_per_head)

        # print(q_w.shape, k_w.shape, q_z.shape, q_z.shape, q_s.shape, k_s.shape)

        qkv_w = merge_qkv(q_w, k_w, v_w, tp, dim=2)
        qkv_z = merge_qkv(q_z, k_z, v_z, tp, dim=2)
        qkv_s = merge_qkv(q_s, k_s, v_s, tp, dim=2)


        qkv_z = qkv_z.to(qkv_s.dtype)
        qkv_w = pack_u4_row(qkv_w)

        self.save_split(qkv_w, f'layers.{i}.attention.w_qkv.qweight', -1)
        self.save_split(qkv_s, f'layers.{i}.attention.w_qkv.scales', -1)
        self.save_split(qkv_z, f'layers.{i}.attention.w_qkv.zeros', -1)

        o_z = o_z.to(o_s.dtype)
        o_w = pack_u4_row(o_w)

        self.save_split(o_w, f'layers.{i}.attention.wo.qweight', 0)
        self.save_split(o_s, f'layers.{i}.attention.wo.scales', 0)
        self.save_split(o_z, f'layers.{i}.attention.wo.zeros', 0)

        q_b, k_b, v_b, o_b = transpose_tensor(bin.attn_bias(i))
        if q_b is not None:
            if self.permute_qk:
                q_b = permute(q_b, size_per_head)
                k_b = permute(k_b, size_per_head)
            else:
                q_b = q_b[None, :]
                k_b = k_b[None, :]
            qkv_b = merge_qkv(q_b, k_b, v_b, tp, dim=1)
            self.save_split(qkv_b, f'layers.{i}.attention.w_qkv.bias', -1)
            self.save_split(o_b, f'layers.{i}.attention.wo.bias', copy=True)

        # ffn weights
        w1_w, w2_w, w3_w = transpose_tensor(bin.ffn(i))
        w1_z, w2_z, w3_z = transpose_tensor(bin.ffn_zero(i))
        w1_s, w2_s, w3_s = transpose_tensor(bin.ffn_scale(i))

        w1_w = pack_u4_row(w1_w)
        w3_w = pack_u4_row(w3_w)
        w2_w = pack_u4_row(w2_w)

        self.save_split(w1_w, f'layers.{i}.feed_forward.w1.qweight', -1)
        self.save_split(w3_w, f'layers.{i}.feed_forward.w3.qweight', -1)
        self.save_split(w2_w, f'layers.{i}.feed_forward.w2.qweight', 0)

        self.save_split(w1_s, f'layers.{i}.feed_forward.w1.scales', -1)
        self.save_split(w3_s, f'layers.{i}.feed_forward.w3.scales', -1)
        self.save_split(w2_s, f'layers.{i}.feed_forward.w2.scales', 0)

        w1_z = w1_z.to(w1_s.dtype)
        w3_z = w3_z.to(w3_s.dtype)
        w2_z = w2_z.to(w2_s.dtype)

        self.save_split(w1_z, f'layers.{i}.feed_forward.w1.zeros', -1)
        self.save_split(w3_z, f'layers.{i}.feed_forward.w3.zeros', -1)
        self.save_split(w2_z, f'layers.{i}.feed_forward.w2.zeros', 0)

        # norm
        attn_norm = bin.attn_norm(i)
        ffn_norm = bin.ffn_norm(i)
        self.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')
