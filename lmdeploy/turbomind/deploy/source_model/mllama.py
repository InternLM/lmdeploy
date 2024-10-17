# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class MLlamaReader(LlamaReader):
    """Llama3.2Reader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg,
                         **kwargs)


@INPUT_MODELS.register_module(name='mllama')
class MLlamaModel(LlamaModel):
    """Llama3.2 model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        self.Reader = MLlamaReader
