# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .deepseek_vl import DeepSeekVLModel, DeepSeekVLReader
from .llama_awq import process_awq_gemm


class DeepSeekVLAwqReader(DeepSeekVLReader):
    """LlamaAwqReader."""

    weight_suffix = 'qweight'

    def _transform(self, x: torch.Tensor, kind: str):
        return process_awq_gemm(x)


@INPUT_MODELS.register_module(name='deepseekvl-awq')
class DeepSeekVLAwqModel(DeepSeekVLModel):
    """Llama Awq model in hf format."""

    Reader = DeepSeekVLAwqReader
