# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .hf import HfModel, HfWeightFileMgr


def ensure_fp16orint32(tensors: torch.Tensor):
    """Ensure tensors in fp16/int32 format."""
    result = []
    for tensor in tensors:
        if tensor is not None:
            if tensor.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                result.append(tensor.half())
            else:
                assert tensor.dtype == torch.int32
                result.append(tensor)
        else:
            result.append(None)
    return (*result, )


class HfAwqWeightFileMgr(HfWeightFileMgr):
    """HfAwqWeightFileMgr."""

    def __init__(self, new_params: dict, unused_params: dict):
        super().__init__(new_params, unused_params)

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params[
                f'model.layers.{i}.self_attn.{key}_proj.qweight']
            result.append(tensor)
        return ensure_fp16orint32(result)

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'model.layers.{i}.self_attn.{key}_proj.qzeros', None)
            result.append(tensor)
        return ensure_fp16orint32(result)

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'model.layers.{i}.self_attn.{key}_proj.scales', None)
            result.append(tensor)
        return ensure_fp16orint32(result)

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        result = []
        for key in ['gate_proj', 'down_proj', 'up_proj']:
            tensor = self.params[f'model.layers.{i}.mlp.{key}.qweight']
            result.append(tensor)
        return ensure_fp16orint32(result)

    def ffn_zero(self, i: int):
        """Get ffn qzeros for layer i."""
        result = []
        for key in ['gate_proj', 'down_proj', 'up_proj']:
            tensor = self.params[f'model.layers.{i}.mlp.{key}.qzeros']
            result.append(tensor)
        return ensure_fp16orint32(result)

    def ffn_scale(self, i: int):
        """Get ffn scales for layer i."""
        result = []
        for key in ['gate_proj', 'down_proj', 'up_proj']:
            tensor = self.params[f'model.layers.{i}.mlp.{key}.scales']
            result.append(tensor)
        return ensure_fp16orint32(result)


@INPUT_MODELS.register_module(name='hf-awq')
class HfAwqModel(HfModel):
    """Awq model in hf format."""

    WeightFileMgr = HfAwqWeightFileMgr

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path, tokenizer_path, ckpt_path)
