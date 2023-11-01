# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.turbomind.deploy.source_model.hf import (INPUT_MODELS, HfModel,
                                                       HfWeightFileMgr)


class BaichuanWeightFileMgr(HfWeightFileMgr):
    """BaichuanWeightFileMgr."""

    def __init__(self,
                 new_params: dict,
                 unused_params: dict,
                 is_baichuan2: bool = True):
        super().__init__(new_params, unused_params)
        self.is_baichuan2 = is_baichuan2

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        result = []
        qkv = self.params[f'model.layers.{i}.self_attn.W_pack.weight']
        o = self.params[f'model.layers.{i}.self_attn.o_proj.weight']
        result.extend(torch.split(qkv, qkv.shape[0] // 3, dim=0))
        result.append(o)
        return (*result, )

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        result = []
        qkv = self.params.get(f'model.layers.{i}.self_attn.W_pack.bias', None)
        if qkv is not None:
            result.extend(torch.split(qkv, qkv.shape[0] // 3, dim=0))
            result.extend(
                self.params[f'model.layers.{i}.self_attn.o_proj.bias'])
        else:
            result.extend([None] * 4)
        return (*result, )


class Baichuan2WeightFileMgr(BaichuanWeightFileMgr):
    """Baichuan2WeightFileMgr."""

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


@INPUT_MODELS.register_module(name='baichuan')
class BaichuanModel(HfModel):
    """Llama model in baichuan format."""

    WeightFileMgr = BaichuanWeightFileMgr

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path, **kwargs)


@INPUT_MODELS.register_module(name='baichuan2')
class Baichuan2Model(HfModel):
    """Llama model in baichuan format."""

    WeightFileMgr = Baichuan2WeightFileMgr

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path, **kwargs)
