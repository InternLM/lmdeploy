# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .internlm2 import InternLM2Model, InternLM2Reader


class Xcomposer2Reader(InternLM2Reader):
    """Xcomposer2 model reader."""

    def attn_lora_a(self, i):
        """Get attn lora_a."""
        qkv = self.params[f'model.layers.{i}.attention.wqkv.Plora_A.weight']
        o = self.params[f'model.layers.{i}.attention.wo.Plora_A.weight']
        return qkv, o

    def attn_lora_b(self, i):
        """Get attn lora_b."""
        return self._attn(i, 'Plora_B.weight')

    def ffn_lora_a(self, i: int):
        """Get ffn lora_a weight for layer i."""
        return self._ffn(i, 'Plora_A.weight')

    def ffn_lora_b(self, i: int):
        """Get fnn lora_b weight for layer i."""
        return self._ffn(i, 'Plora_B.weight')


@INPUT_MODELS.register_module(name='xcomposer2')
class Xcomposer2Model(InternLM2Model):
    """Xcomposer2 model in hf format."""

    Reader = Xcomposer2Reader

    def _lora_cfg_7b(self):
        """lora config for internlm-xcomposer2-7b."""
        return dict(lora_r=256,
                    lora_scale=1.0,
                    lora_policy='plora',
                    lora_max_wo_r=256)

    def _lora_cfg_4khd_7b(self, model_info: dict):
        """lora config for internlm-xcomposer2-4khd-7b."""
        rank_pattern = ['attention.w_qkv:8', 'attention.wo:256']
        scale_pattern = ['attention.w_qkv:2.0', 'attention.wo:1.0']
        rank_pattern = ','.join(rank_pattern)
        scale_pattern = ','.join(scale_pattern)
        return dict(lora_r=256,
                    lora_scale=1.0,
                    lora_max_wo_r=256,
                    lora_policy='plora',
                    lora_rank_pattern=rank_pattern,
                    lora_scale_pattern=scale_pattern)

    def model_info(self):
        out = super().model_info()
        from lmdeploy.vl.model.xcomposer2 import ModelType, get_xcomposer_type
        model_type, _ = get_xcomposer_type(self.model_path)
        if model_type == ModelType.XCOMPOSER2_4KHD:
            out.update(self._lora_cfg_4khd_7b(out))
        else:
            out.update(self._lora_cfg_7b())
        return out
