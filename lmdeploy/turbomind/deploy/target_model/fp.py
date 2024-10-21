# Copyright (c) OpenMMLab. All rights reserved.

from .base import OUTPUT_MODELS, BaseOutputModel


@OUTPUT_MODELS.register_module(name='tm')
class TurbomindModel(BaseOutputModel):
    """Export to turbomind fp16 format."""

    # TODO: move this to proper location
    def _pad_inter_size(self, inter_size: int, cfg: dict):
        group_size = max(1, cfg['group_size'])
        tp = cfg['tensor_para_size']
        groups_per_rank = (inter_size // group_size + tp - 1) // tp
        inter_size_padded = groups_per_rank * group_size * tp
        return inter_size_padded
