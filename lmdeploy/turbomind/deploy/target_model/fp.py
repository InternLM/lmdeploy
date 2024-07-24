# Copyright (c) OpenMMLab. All rights reserved.

from .base import OUTPUT_MODELS, BaseOutputModel, TurbomindModelConfig


@OUTPUT_MODELS.register_module(name='tm')
class TurbomindModel(BaseOutputModel):
    """Export to turbomind fp16 format."""

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
        inter_size = self._pad_inter_size(inter_size, final_cfg)
        final_cfg.update(dict(attn_bias=attn_bias, inter_size=inter_size))
        return TurbomindModelConfig.from_dict(final_cfg)

    def _pad_inter_size(self, inter_size: int, cfg: dict):
        group_size = max(1, cfg['group_size'])
        tp = cfg['tensor_para_size']
        groups_per_rank = (inter_size // group_size + tp - 1) // tp
        inter_size_padded = groups_per_rank * group_size * tp
        return inter_size_padded
