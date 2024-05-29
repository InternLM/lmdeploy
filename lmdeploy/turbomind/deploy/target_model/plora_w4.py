# Copyright (c) OpenMMLab. All rights reserved.
from ..source_model.base import BaseInputModel, BaseReader
from .base import OUTPUT_MODELS, TurbomindModelConfig, merge_qkv, permute
from .plora import TurbomindPloraModel, transpose_tensor
from .w4 import convert_s4, get_cuda_tensor, tp_m_s4, transpose_qk_s4


@OUTPUT_MODELS.register_module(name=['plora-w4'])
class TurbomindPloraW4Model(TurbomindPloraModel):
    """Export to turbomind plora w4 format."""

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
        assert bin.start_layer_id <= i < bin.end_layer_id
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

        w1_qw, w1_sz = convert_s4(w1_qw, w1_qz, w1_s, group_size)
        w3_qw, w3_sz = convert_s4(w3_qw, w3_qz, w3_s, group_size)
        self.save_split(w1_qw, f'layers.{i}.feed_forward.w1.qweight', -1)
        self.save_split(w1_sz, f'layers.{i}.feed_forward.w1.scales_zeros', -1)
        self.save_split(w3_qw, f'layers.{i}.feed_forward.w3.qweight', -1)
        self.save_split(w3_sz, f'layers.{i}.feed_forward.w3.scales_zeros', -1)

        w2_qw, w2_sz = convert_s4(w2_qw, w2_qz, w2_s, group_size)
        self.save_split(w2_qw, f'layers.{i}.feed_forward.w2.qweight', 0)
        self.save_split(w2_sz, f'layers.{i}.feed_forward.w2.scales_zeros', 0)

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

        # # ffn lora_a
        lora_a_w1, lora_a_w2, lora_a_w3 = bin.ffn_lora_a(i)
        lora_a_w1, lora_a_w2, lora_a_w3 = transpose_tensor(
            [lora_a_w1, lora_a_w2, lora_a_w3])
        # print('lora_a_w1', lora_a_w1.shape, lora_a_w2.shape, lora_a_w3.shape)
        self.save_split(lora_a_w2, f'layers.{i}.feed_forward.w2.lora_a.weight',
                        0)
        self.save_split(lora_a_w1,
                        f'layers.{i}.feed_forward.w1.lora_a.weight',
                        copy=True)
        self.save_split(lora_a_w3,
                        f'layers.{i}.feed_forward.w3.lora_a.weight',
                        copy=True)
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
