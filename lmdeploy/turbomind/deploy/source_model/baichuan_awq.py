# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.turbomind.deploy.source_model.awq import (AwqModel,
                                                        AwqWeightFileMgr,
                                                        ensure_fp16orint32)
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS


class BaichuanAwqWeightFileMgr(AwqWeightFileMgr):
    """BaichuanAwqWeightFileMgr."""

    def __init__(self, new_params: dict, unused_params: dict):
        super().__init__(new_params, unused_params)

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        result = []
        qkv = self.params[f'model.layers.{i}.self_attn.W_pack.qweight']
        o = self.params[f'model.layers.{i}.self_attn.o_proj.qweight']
        result.extend(torch.split(qkv, qkv.shape[-1] // 3, dim=-1))
        result.append(o)
        return ensure_fp16orint32(result)

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        result = []
        qkv = self.params[f'model.layers.{i}.self_attn.W_pack.qzeros']
        o = self.params[f'model.layers.{i}.self_attn.o_proj.qzeros']
        result.extend(torch.split(qkv, qkv.shape[-1] // 3, dim=-1))
        result.append(o)
        return ensure_fp16orint32(result)

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        result = []
        qkv = self.params[f'model.layers.{i}.self_attn.W_pack.scales']
        o = self.params[f'model.layers.{i}.self_attn.o_proj.scales']
        result.extend(torch.split(qkv, qkv.shape[-1] // 3, dim=-1))
        result.append(o)
        return ensure_fp16orint32(result)


class Baichuan2AwqWeightFileMgr(BaichuanAwqWeightFileMgr):
    """Baichuan2AwqWeightFileMgr."""

    def __init__(self, new_params: dict, unused_params: dict):
        super().__init__(new_params, unused_params)

    def output_weight(self):
        """Get output."""
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L507
        tensor = self.params.get('lm_head.weight', None)
        if tensor is not None:
            tensor = tensor.cuda()
            tensor = torch.nn.functional.normalize(tensor)
        return tensor


@INPUT_MODELS.register_module(name='baichuan-awq')
class BaichuanAwqModel(AwqModel):
    """Baichuan awq model in hf format."""

    WeightFileMgr = BaichuanAwqWeightFileMgr

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 quant_path: str = None,
                 **kwargs):
        super().__init__(model_path, tokenizer_path)


@INPUT_MODELS.register_module(name='baichuan2-awq')
class Baichuan2AwqModel(AwqModel):
    """Baichuan2 awq model in hf format."""

    WeightFileMgr = Baichuan2AwqWeightFileMgr

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 quant_path: str = None,
                 **kwargs):
        super().__init__(model_path, tokenizer_path)
