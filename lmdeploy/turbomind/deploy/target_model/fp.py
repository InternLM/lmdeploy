# Copyright (c) OpenMMLab. All rights reserved.

from ..source_model.base import BaseInputModel
from .base import (OUTPUT_MODELS, BaseOutputModel, LayerNormExporter,
                   TurbomindModelConfig, WeightExporter)


@OUTPUT_MODELS.register_module(name=['fp16', 'bf16'])
class TurbomindModel(BaseOutputModel):
    """Export to turbomind fp16 format."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        super().__init__(input_model, cfg, to_file, out_dir)
        self.exporters = [WeightExporter(self), LayerNormExporter(self)]

    def get_config(self, cfg: TurbomindModelConfig):
        """Get turbomind config."""
        final_cfg = super().get_config(cfg).__dict__

        # attn_bias, inter_size
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
        final_cfg.update(dict(attn_bias=attn_bias, inter_size=inter_size))
        return TurbomindModelConfig.from_dict(final_cfg)
