# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .baichuan import (BaichuanAwqModel, BaichuanAwqReader, BaichuanModel,
                       BaichuanReader)
from .base import INPUT_MODELS
from .llama import LlamaModel


class Baichuan2Reader(BaichuanReader):
    """Baichuan2Reader."""

    def output_weight(self):
        """Get output."""
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L507
        tensor = self.params.get('lm_head.weight', None)
        if tensor is not None:
            tensor = tensor.cuda()
            tensor = torch.nn.functional.normalize(tensor)
        return tensor


@INPUT_MODELS.register_module(name='baichuan2')
class Baichuan2Model(LlamaModel):
    """Llama model in baichuan format."""

    Reader = Baichuan2Reader


class Baichuan2AwqReader(BaichuanAwqReader):
    """Baichuan2AwqReader."""

    def output_weight(self):
        """Get output."""
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L507
        tensor = self.params.get('lm_head.weight', None)
        if tensor is not None:
            tensor = tensor.cuda()
            tensor = torch.nn.functional.normalize(tensor)
        return tensor


@INPUT_MODELS.register_module(name='baichuan2-awq')
class Baichuan2AwqModel(Baichuan2Model):
    """Baichuan2 awq model in hf format."""

    Reader = Baichuan2AwqReader
