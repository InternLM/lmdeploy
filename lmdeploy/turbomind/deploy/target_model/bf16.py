# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.turbomind.deploy.source_model.base import BaseInputModel

from .base import OUTPUT_MODELS, TurbomindModelConfig
from .fp import TurbomindModel


@OUTPUT_MODELS.register_module(name='bf16')
class TurbomindBF16Model(TurbomindModel):

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        super().__init__(input_model, cfg, to_file, out_dir)
