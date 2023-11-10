# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .baichuan import Baichuan2Model, BaichuanModel, BaichuanReader
from .base import INPUT_MODELS
from .llama_awq import ensure_fp16orint32


class BaichuanAwqReader(BaichuanReader):
    """BaichuanAwqReader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool):
        super().__init__(new_params, unused_params, last_bin)

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        return ensure_fp16orint32(self._attn(i, 'qweight', -1, -1))

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        return ensure_fp16orint32(self._attn(i, 'qzeros', -1, -1))

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        return ensure_fp16orint32(self._attn(i, 'scales', -1, -1))

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qweight'))

    def ffn_zero(self, i: int):
        """Get ffn qzeros for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qzeros'))

    def ffn_scale(self, i: int):
        """Get ffn scales for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'scales'))


class Baichuan2AwqReader(BaichuanAwqReader):
    """Baichuan2AwqReader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool):
        super().__init__(new_params, unused_params, last_bin)

    def output_weight(self):
        """Get output."""
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L507
        tensor = self.params.get('lm_head.weight', None)
        if tensor is not None:
            tensor = tensor.cuda()
            tensor = torch.nn.functional.normalize(tensor)
        return tensor


@INPUT_MODELS.register_module(name='baichuan-awq')
class BaichuanAwqModel(BaichuanModel):
    """Baichuan awq model in hf format."""

    Reader = BaichuanAwqReader

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path,
                         tokenizer_path,
                         ckpt_path=ckpt_path,
                         **kwargs)


@INPUT_MODELS.register_module(name='baichuan2-awq')
class Baichuan2AwqModel(Baichuan2Model):
    """Baichuan2 awq model in hf format."""

    Reader = Baichuan2AwqReader

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path,
                         tokenizer_path,
                         ckpt_path=ckpt_path,
                         **kwargs)
