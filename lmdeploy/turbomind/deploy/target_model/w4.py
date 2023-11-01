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
                inter_size = w1s.shape[-1]
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
        group_size = self.cfg.group_size
        tp = self.cfg.tensor_para_size
        size_per_head = self.cfg.size_per_head
        # attn
        q_qw, k_qw, v_qw, o_qw = get_cuda_tensor(bin.attn(i))
        q_qz, k_qz, v_qz, o_qz = get_cuda_tensor(bin.attn_zero(i))
        q_s, k_s, v_s, o_s = get_cuda_tensor(bin.attn_scale(i))

        q_qw = transpose_qk_s4(q_qw, group_size)
        k_qw = transpose_qk_s4(k_qw, group_size)
        q_qz = transpose_qk_s4(q_qz, group_size)
        k_qz = transpose_qk_s4(k_qz, group_size)
        q_s = permute(q_s, size_per_head)
        k_s = permute(k_s, size_per_head)

        qkv_qw = merge_qkv(q_qw, k_qw, v_qw, tp, dim=2)
        qkv_qz = merge_qkv(q_qz, k_qz, v_qz, tp, dim=2)
        qkv_s = merge_qkv(q_s, k_s, v_s, tp, dim=2)

        qkv_qw, qkv_sz = convert_s4(qkv_qw, qkv_qz, qkv_s, group_size)
        qkv_qw = tp_m_s4(qkv_qw, tp)
        self.save_split(qkv_qw, f'layers.{i}.attention.w_qkv.qweight', -1)
        self.save_split(qkv_sz, f'layers.{i}.attention.w_qkv.scales_zeros', -1)

        o_qw, o_sz = convert_s4(o_qw, o_qz, o_s, group_size)
        self.save_split(o_qw, f'layers.{i}.attention.wo.qweight', 0)
        self.save_split(o_sz, f'layers.{i}.attention.wo.scales_zeros', 0)

        q_b, k_b, v_b, o_b = get_cuda_tensor(bin.attn_bias(i))
        if q_b is not None:
            q_b = permute(q_b, size_per_head)
            k_b = permute(k_b, size_per_head)
            qkv_b = merge_qkv(q_b, k_b, v_b, tp, dim=1)
            self.save_split(qkv_b, f'layers.{i}.attention.w_qkv.bias', -1)
            self.save_split(o_b, f'layers.{i}.attention.wo.bias', copy=True)

        # ffn weights
        w1_qw, w2_qw, w3_qw = get_cuda_tensor(bin.ffn(i))
        w1_qz, w2_qz, w3_qz = get_cuda_tensor(bin.ffn_zero(i))
        w1_s, w2_s, w3_s = get_cuda_tensor(bin.ffn_scale(i))

        w13_qw, w13_qz, w13_s = fuse_w1_w3_s4(w1_qw, w1_qz, w1_s, w3_qw, w3_qz,
                                              w3_s)
        w13_qw, w13_sz = convert_s4(w13_qw, w13_qz, w13_s, group_size)
        w13_qw = tp_m_s4(w13_qw, tp)
        self.save_split(w13_qw, f'layers.{i}.feed_forward.w13.qweight', -1)
        self.save_split(w13_sz, f'layers.{i}.feed_forward.w13.scales_zeros',
                        -1)

        w2_qw, w2_sz = convert_s4(w2_qw, w2_qz, w2_s, group_size)
        self.save_split(w2_qw, f'layers.{i}.feed_forward.w2.qweight', 0)
        self.save_split(w2_sz, f'layers.{i}.feed_forward.w2.scales_zeros', 0)

        # norm
        attn_norm = bin.attn_norm(i)
        ffn_norm = bin.ffn_norm(i)
        self.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')
