# Copyright (c) OpenMMLab. All rights reserved.

from ..source_model.base import BaseInputModel
from .base import (OUTPUT_MODELS, BaseOutputModel, LayerNormExporter,
                   PLoraExporter, TurbomindModelConfig, WeightExporter)


@OUTPUT_MODELS.register_module(name=['plora'])
class TurbomindPloraModel(BaseOutputModel):
    """Export to turbomind fp16 format."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        super().__init__(input_model, cfg, to_file, out_dir)
        self.exporters = [
            WeightExporter(self),
            LayerNormExporter(self),
            PLoraExporter(self)
        ]
