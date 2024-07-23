# Copyright (c) OpenMMLab. All rights reserved.

from ..source_model.base import BaseInputModel
from .base import (OUTPUT_MODELS, LayerNormExporter, QuantWeightExporter,
                   TurbomindModelConfig, pack_u4_row)
from .fp import TurbomindModel


@OUTPUT_MODELS.register_module(name='w4')
class TurbomindW4Model(TurbomindModel):
    """Export to turbomind w4a16 format."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        super().__init__(input_model, cfg, to_file, out_dir)
        self.exporters = [
            QuantWeightExporter(self, pack_u4_row),
            LayerNormExporter(self),
        ]
