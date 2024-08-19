# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .llama_awq import LlamaAwqModel, LlamaAwqReader
from .qwen_awq import Qwen2AwqReader


class MiniCPMVAwqReader(LlamaAwqReader):
    """MiniCPMVAwqReader model."""

    attn_layer_prefix = 'llm.model.layers'
    attn_layer_patten = r'llm.model.layers.([0-9]+).'
    tok_embeddings_key = 'llm.model.embed_tokens.weight'
    norm_weight_key = 'llm.model.norm.weight'
    output_weight_key = 'llm.lm_head.weight'


class MiniCPMV26AwqReader(Qwen2AwqReader):
    """MiniCPMV26AwqReader."""
    attn_layer_prefix = 'llm.model.layers'
    attn_layer_patten = r'llm.model.layers.([0-9]+).'
    tok_embeddings_key = 'llm.model.embed_tokens.weight'
    norm_weight_key = 'llm.model.norm.weight'
    output_weight_key = 'llm.lm_head.weight'


@INPUT_MODELS.register_module(name='minicpmv-awq')
class MiniCPMVAwqModel(LlamaAwqModel):
    """MiniCPMV awq model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        from transformers.dynamic_module_utils import \
            get_class_from_dynamic_module

        def check_llm(name):
            try:
                cls = '.'.join(['modeling_minicpmv', name])
                _ = get_class_from_dynamic_module(cls, model_path)
                return True
            except Exception:
                return False

        if check_llm('LlamaForCausalLM'):
            reader = MiniCPMVAwqReader
        elif check_llm('Qwen2ForCausalLM'):
            reader = MiniCPMV26AwqReader
        else:
            raise NotImplementedError
        self.Reader = reader
