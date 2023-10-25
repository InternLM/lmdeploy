# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
from lmdeploy.turbomind.deploy.source_model.qwen import (QwenModel,
                                                         QwenWeightFileMgr)


def fp32tofp16_tensors(tensors: torch.Tensor):
    """Ensure tensors in fp16/int32 format."""
    result = []
    for tensor in tensors:
        if tensor is not None and tensor.dtype == torch.float32:
            result.append(tensor.half())
        else:
            result.append(tensor)
    return (*result, )


class QwenAwqWeightFileMgr(QwenWeightFileMgr):
    """QwenAwqWeightFileMgr."""

    def __init__(self, new_params: dict, unused_params: dict):
        super().__init__(new_params, unused_params)

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        qkv_qw = self.params[f'transformer.h.{i}.attn.c_attn.qweight']
        q_qw, k_qw, v_qw = torch.split(qkv_qw, qkv_qw.size(-1) // 3, dim=-1)
        o_qw = self.params[f'transformer.h.{i}.attn.c_proj.qweight']
        return fp32tofp16_tensors((q_qw, k_qw, v_qw, o_qw))

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        qkv_b = self.params[f'transformer.h.{i}.attn.c_attn.bias']
        q_b, k_b, v_b = torch.split(qkv_b, qkv_b.size(-1) // 3)
        o_b = torch.zeros_like(q_b)
        return fp32tofp16_tensors((q_b, k_b, v_b, o_b))

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        qkv_qz = self.params[f'transformer.h.{i}.attn.c_attn.qzeros']
        q_qz, k_qz, v_qz = torch.split(qkv_qz, qkv_qz.size(-1) // 3, dim=-1)
        o_qz = self.params[f'transformer.h.{i}.attn.c_proj.qzeros']
        return fp32tofp16_tensors((q_qz, k_qz, v_qz, o_qz))

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        qkv_s = self.params[f'transformer.h.{i}.attn.c_attn.scales']
        q_s, k_s, v_s = torch.split(qkv_s, qkv_s.size(-1) // 3, dim=-1)
        o_s = self.params[f'transformer.h.{i}.attn.c_proj.scales']
        return fp32tofp16_tensors((q_s, k_s, v_s, o_s))

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        # ours: w2(silu(w1(x)) * w3(x))
        # qwen: c_proj(w1(x) * silu(w2(x)))
        w1_qw = self.params[f'transformer.h.{i}.mlp.w2.qweight']
        w3_qw = self.params[f'transformer.h.{i}.mlp.w1.qweight']
        w2_qw = self.params[f'transformer.h.{i}.mlp.c_proj.qweight']
        return fp32tofp16_tensors((w1_qw, w2_qw, w3_qw))

    def ffn_zero(self, i: int):
        """Get ffn qzeros for layer i."""
        w1_qz = self.params[f'transformer.h.{i}.mlp.w2.qzeros']
        w3_qz = self.params[f'transformer.h.{i}.mlp.w1.qzeros']
        w2_qz = self.params[f'transformer.h.{i}.mlp.c_proj.qzeros']
        return fp32tofp16_tensors((w1_qz, w2_qz, w3_qz))

    def ffn_scale(self, i: int):
        """Get ffn scales for layer i."""
        w1_s = self.params[f'transformer.h.{i}.mlp.w2.scales']
        w3_s = self.params[f'transformer.h.{i}.mlp.w1.scales']
        w2_s = self.params[f'transformer.h.{i}.mlp.c_proj.scales']
        return fp32tofp16_tensors((w1_s, w2_s, w3_s))


@INPUT_MODELS.register_module(name='qwen-awq')
class QwenAwqModel(QwenModel):
    """Qwen awq model in hf format."""

    WeightFileMgr = QwenAwqWeightFileMgr

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 quant_path: str = None,
                 **kwargs):
        super().__init__(model_path, tokenizer_path)
