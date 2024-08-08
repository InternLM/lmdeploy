# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from ..source_model.base import BaseInputModel, BaseReader
from .base import (OUTPUT_MODELS, BaseOutputModel, TurbomindModelConfig,
                   merge_qkv, permute)


def transpose_tensor(input: List[torch.Tensor]):
    """Transpose tensor."""
    output = [x.cuda().t() for x in input]
    return output


@OUTPUT_MODELS.register_module(name=['plora'])
class TurbomindPloraModel(BaseOutputModel):
    """Export to turbomind fp16 format."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 out_dir: str = ''):
        super().__init__(input_model, cfg, out_dir)

    def get_config(self, cfg: TurbomindModelConfig):
        """Get turbomind config."""
        final_cfg = super().get_config(cfg).__dict__

        # attn_bias, inter_size
        visit = False
        attn_bias = 0
        for bin in self.input_model.bins():
            for i in range(bin.start_layer_id, bin.end_layer_id):
                visit = True
                w1, _, _ = bin.ffn(i)
                inter_size = w1.t().shape[-1]
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
        assert bin.start_layer_id <= i < bin.end_layer_id
        tp = self.cfg.tensor_para_size
        size_per_head = self.cfg.size_per_head
        # attn
        qw, kw, vw, ow = bin.attn(i)
        qw, kw, vw, ow = transpose_tensor([qw, kw, vw, ow])
        qw = permute(qw, size_per_head)
        kw = permute(kw, size_per_head)
        qkv_w = merge_qkv(qw, kw, vw, tp, dim=2)
        self.save_split(qkv_w, f'layers.{i}.attention.w_qkv.weight', -1)
        self.save_split(ow, f'layers.{i}.attention.wo.weight', 0)
        qb, kb, vb, ob = bin.attn_bias(i)
        # attn lora_a
        lora_a_qkv, lora_a_o = bin.attn_lora_a(i)
        lora_a_qkv, lora_a_o = transpose_tensor([lora_a_qkv, lora_a_o])
        # print(lora_a_qkv.shape, lora_a_o.shape)
        self.save_split(lora_a_qkv,
                        f'layers.{i}.attention.w_qkv.lora_a.weight',
                        copy=True)
        self.save_split(lora_a_o, f'layers.{i}.attention.wo.lora_a.weight', 0)
        # attn lora_b
        lora_b_qw, lora_b_kw, lora_b_vw, lora_b_ow = bin.attn_lora_b(i)
        lora_b_qw, lora_b_kw, lora_b_vw, lora_b_ow = transpose_tensor(
            [lora_b_qw, lora_b_kw, lora_b_vw, lora_b_ow])
        lora_b_qw = permute(lora_b_qw, size_per_head)
        lora_b_kw = permute(lora_b_kw, size_per_head)
        lora_b_qkv_w = merge_qkv(lora_b_qw, lora_b_kw, lora_b_vw, tp, dim=2)
        self.save_split(lora_b_qkv_w,
                        f'layers.{i}.attention.w_qkv.lora_b.weight', -1)
        self.save_split(lora_b_ow,
                        f'layers.{i}.attention.wo.lora_b.weight',
                        copy=True)

        if qb is not None:
            qb, kb, vb, ob = transpose_tensor([qb, kb, vb, ob])
            qb = permute(qb, size_per_head)
            kb = permute(kb, size_per_head)
            qkv_b = merge_qkv(qb, kb, vb, tp, dim=1)
            self.save_split(qkv_b, f'layers.{i}.attention.w_qkv.bias', -1)
            self.save_split(ob, f'layers.{i}.attention.wo.bias', copy=True)
        # ffn
        w1, w2, w3 = bin.ffn(i)
        w1, w2, w3 = transpose_tensor([w1, w2, w3])
        self.save_split(w1, f'layers.{i}.feed_forward.w1.weight', -1)
        self.save_split(w3, f'layers.{i}.feed_forward.w3.weight', -1)
        self.save_split(w2, f'layers.{i}.feed_forward.w2.weight', 0)
        # # ffn lora_a
        lora_a_w1, lora_a_w2, lora_a_w3 = bin.ffn_lora_a(i)
        lora_a_w1, lora_a_w2, lora_a_w3 = transpose_tensor(
            [lora_a_w1, lora_a_w2, lora_a_w3])
        # print('lora_a_w1', lora_a_w1.shape, lora_a_w2.shape, lora_a_w3.shape)
        self.save_split(lora_a_w1,
                        f'layers.{i}.feed_forward.w1.lora_a.weight',
                        copy=True)
        self.save_split(lora_a_w3,
                        f'layers.{i}.feed_forward.w3.lora_a.weight',
                        copy=True)
        self.save_split(lora_a_w2, f'layers.{i}.feed_forward.w2.lora_a.weight',
                        0)
        # # ffn lora_b
        lora_b_w1, lora_b_w2, lora_b_w3 = bin.ffn_lora_b(i)
        lora_b_w1, lora_b_w2, lora_b_w3 = transpose_tensor(
            [lora_b_w1, lora_b_w2, lora_b_w3])
        # print('lora_b_w1', lora_b_w1.shape, lora_b_w2.shape, lora_b_w3.shape)
        self.save_split(lora_b_w1, f'layers.{i}.feed_forward.w1.lora_b.weight',
                        -1)
        self.save_split(lora_b_w3, f'layers.{i}.feed_forward.w3.lora_b.weight',
                        -1)
        self.save_split(lora_b_w2,
                        f'layers.{i}.feed_forward.w2.lora_b.weight',
                        copy=True)

        # norm
        attn_norm = bin.attn_norm(i)
        ffn_norm = bin.ffn_norm(i)
        self.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')
