# Copyright (c) OpenMMLab. All rights reserved.

import json
import os.path as osp

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class MiniCPMVReader(LlamaReader):
    """MiniCPMVReader for llama model."""

    attn_layer_prefix = 'llm.model.layers'
    attn_layer_patten = r'llm\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'llm.model.embed_tokens.weight'
    norm_weight_key = 'llm.model.norm.weight'
    output_weight_key = 'llm.lm_head.weight'


@INPUT_MODELS.register_module(name='minicpmv')
class MiniCPMVModel(LlamaModel):
    """MiniCPMV model in hf format."""
    Reader = MiniCPMVReader

    def model_info(self):
        info = super().model_info()
        with open(osp.join(self.model_path, 'config.json')) as f:
            config = json.load(f)
            if str(config.get('version')) == '2.6':
                info['attn_bias'] = True
        return info
