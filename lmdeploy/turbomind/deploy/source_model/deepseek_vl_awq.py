# Copyright (c) OpenMMLab. All rights reserved.
from .base import INPUT_MODELS
from .deepseek_vl import DeepSeekVLModel, DeepSeekVLReader
from .llama_awq import ensure_fp16orint32


class DeepSeekVLAwqReader(DeepSeekVLReader):
    """LlamaAwqReader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

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


@INPUT_MODELS.register_module(name='deepseekvl-awq')
class DeepSeekVLAwqModel(DeepSeekVLModel):
    """Llama Awq model in hf format."""

    Reader = DeepSeekVLAwqReader

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path,
                         tokenizer_path,
                         ckpt_path=ckpt_path,
                         **kwargs)
