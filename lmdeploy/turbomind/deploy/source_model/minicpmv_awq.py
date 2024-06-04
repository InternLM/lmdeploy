# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .llama_awq import LlamaAwqModel, LlamaAwqReader


class MiniCPMVAwqReader(LlamaAwqReader):
    """MiniCPMVAwqReader model."""

    attn_layer_prefix = 'llm.model.layers'
    attn_layer_patten = r'llm.model.layers.([0-9]+).'
    tok_embeddings_key = 'llm.model.embed_tokens.weight'
    norm_weight_key = 'llm.model.norm.weight'
    output_weight_key = 'llm.lm_head.weight'


@INPUT_MODELS.register_module(name='minicpmv-awq')
class MiniCPMVAwqModel(LlamaAwqModel):
    """MiniCPMV awq model in hf format."""
    Reader = MiniCPMVAwqReader
