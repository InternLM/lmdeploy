# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


def ensure_dtype(tensors: torch.Tensor, dtype: torch.dtype):
    """Ensure tensors in the specified dytpe."""
    result = []
    for tensor in tensors:
        if tensor is not None and tensor.numel() > 0:
            if tensor.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                result.append(tensor.to(dtype))
            else:
                assert tensor.dtype == torch.int32
                result.append(tensor)
        else:
            result.append(None)
    return (*result, )


class LlamaQQQReader(LlamaReader):
    """LlamaQQQReader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        return ensure_dtype(self._attn(i, 'B'), torch.int32)

    def attn_scale_group(self, i: int):
        """Get q, k, v, o per-group scales for layer i."""
        return ensure_dtype(self._attn(i, 's_group'), torch.float16)

    def attn_scale_channel(self, i: int):
        """Get q, k, v, o per-channel scales for layer i."""
        return ensure_dtype(self._attn(i, 's_channel'), torch.float32)

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        return ensure_dtype(self._ffn(i, 'B'), torch.int32)

    def ffn_scale_group(self, i: int):
        """Get ffn per-group scales for layer i."""
        return ensure_dtype(self._ffn(i, 's_group'), torch.float16)

    def ffn_scale_channel(self, i: int):
        """Get ffn per-channel scales for layer i."""
        return ensure_dtype(self._ffn(i, 's_channel'), torch.float32)


@INPUT_MODELS.register_module(name='llama-qqq')
class LlamaQQQModel(LlamaModel):
    """Llama QQQ model in hf format."""

    Reader = LlamaQQQReader

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path,
                         tokenizer_path,
                         ckpt_path=ckpt_path,
                         **kwargs)
