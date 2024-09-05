# Copyright (c) OpenMMLab. All rights reserved.

from ..config import ModelConfig, config_from_dict, config_to_dict
from .base import OUTPUT_MODELS, BaseOutputModel


@OUTPUT_MODELS.register_module(name='tm')
class TurbomindModel(BaseOutputModel):
    """Export to turbomind fp16 format."""

    def update_model_config(self):
        """Update `self.model_config`.

        Firstly, call `update_model_config` of the superclass. Then update
        `inter_size` and `attn_bias` that are indicates from the input_model's
        weight files
        """
        super().update_model_config()
        final_cfg = config_to_dict(self.model_config)
        # get attn_bias, inter_size
        visit = False
        attn_bias = 0
        for bin in self.input_model.bins():
            for i in range(bin.start_layer_id, bin.end_layer_id):
                visit = True
                w1, w2, w3 = bin.ffn(i)
                inter_size = w2.size(-1)
                qb, _, _, _ = bin.attn_bias(i)
                if qb is not None:
                    attn_bias = 1
                break
            if visit:
                break
        inter_size = self._pad_inter_size(inter_size)
        final_cfg.update(dict(attn_bias=attn_bias, inter_size=inter_size))
        self.model_config = config_from_dict(ModelConfig, final_cfg)

    def _pad_inter_size(self, inter_size: int):
        group_size = max(1, self.model_config.group_size)
        tp = self.tensor_para_size
        groups_per_rank = (inter_size // group_size + tp - 1) // tp
        inter_size_padded = groups_per_rank * group_size * tp
        return inter_size_padded
