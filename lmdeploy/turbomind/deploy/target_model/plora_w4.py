# Copyright (c) OpenMMLab. All rights reserved.
from ..source_model.base import BaseInputModel
from .base import (OUTPUT_MODELS, LayerNormExporter, QuantWeightExporter,
                   TurbomindModelConfig, pack_u4_row)
from .plora import TurbomindPloraModel


@OUTPUT_MODELS.register_module(name=['plora-w4'])
class TurbomindPloraW4Model(TurbomindPloraModel):
    """Export to turbomind plora w4 format."""

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
