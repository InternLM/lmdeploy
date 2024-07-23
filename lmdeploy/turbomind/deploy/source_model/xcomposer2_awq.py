# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .llama_awq import process_awq_gemm
from .xcomposer2 import Xcomposer2Model, Xcomposer2Reader


class Xcomposer2AwqReader(Xcomposer2Reader):
    """LlamaAwqReader."""

    weight_suffix = 'qweight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _transform(self, x: torch.Tensor, kind: str):
        return process_awq_gemm(x)


@INPUT_MODELS.register_module(name='xcomposer2-awq')
class Xcomposer2AwqModel(Xcomposer2Model):
    """Llama Awq model in hf format."""

    Reader = Xcomposer2AwqReader
