# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


def ensure_fp16orint32(tensors: torch.Tensor):
    """Ensure tensors in fp16/int32 format."""
    result = []
    for tensor in tensors:
        if tensor is not None:
            if tensor.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                result.append(tensor.half())
            else:
                assert tensor.dtype == torch.int32
                result.append(tensor)
        else:
            result.append(None)
    return (*result, )


class LlamaAwqReader(LlamaReader):
    """LlamaAwqReader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool):
        super().__init__(new_params, unused_params, last_bin)

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        return ensure_fp16orint32(self._attn(i, 'qweight'))

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        return ensure_fp16orint32(self._attn(i, 'qzeros'))

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        return ensure_fp16orint32(self._attn(i, 'scales'))

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qweight'))

    def ffn_zero(self, i: int):
        """Get ffn qzeros for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qzeros'))

    def ffn_scale(self, i: int):
        """Get ffn scales for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'scales'))


@INPUT_MODELS.register_module(name='hf-awq')
class LlamaAwqModel(LlamaModel):
    """Llama Awq model in hf format."""

    Reader = LlamaAwqReader

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path,
                         tokenizer_path,
                         ckpt_path=ckpt_path,
                         **kwargs)
