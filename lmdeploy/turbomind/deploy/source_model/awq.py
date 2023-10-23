# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
from lmdeploy.turbomind.deploy.source_model.hf import HfModel, HfWeightFileMgr


class AwqWeightFileMgr(HfWeightFileMgr):
    """AwqWeightFileMgr."""

    def __init__(self, new_params: dict, unused_params: dict):
        super().__init__(new_params, unused_params)

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params[
                f'model.layers.{i}.self_attn.{key}_proj.qweight']
            result.append(tensor)
        return (*result, )

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'model.layers.{i}.self_attn.{key}_proj.qzeros', None)
            result.append(tensor)
        return (*result, )

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'model.layers.{i}.self_attn.{key}_proj.scales', None)
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        result = []
        for key in ['gate_proj', 'down_proj', 'up_proj']:
            tensor = self.params[f'model.layers.{i}.mlp.{key}.qweight']
            result.append(tensor)
        return (*result, )

    def ffn_zero(self, i: int):
        """Get ffn qzeros for layer i."""
        result = []
        for key in ['gate_proj', 'down_proj', 'up_proj']:
            tensor = self.params[f'model.layers.{i}.mlp.{key}.qzeros']
            result.append(tensor)
        return (*result, )

    def ffn_scale(self, i: int):
        """Get ffn scales for layer i."""
        result = []
        for key in ['gate_proj', 'down_proj', 'up_proj']:
            tensor = self.params[f'model.layers.{i}.mlp.{key}.scales']
            result.append(tensor)
        return (*result, )


@INPUT_MODELS.register_module(name='awq')
class AwqModel(HfModel):
    """Qwen model in hf format."""

    WeightFileMgr = AwqWeightFileMgr

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 quant_path: str = None,
                 **kwargs):
        super().__init__(model_path, tokenizer_path)
