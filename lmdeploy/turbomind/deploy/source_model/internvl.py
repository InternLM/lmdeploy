# Copyright (c) OpenMMLab. All rights reserved.
from .base import INPUT_MODELS
from .gpt_oss import GptOssReader
from .internlm2 import InternLM2Reader
from .llama import LlamaModel, LlamaReader
from .qwen import Qwen3MoeReader, Qwen3Reader


class InternVLReader(LlamaReader):
    """InternVLReader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('llm_config') or model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


# Note the subtle difference in keys
class InternVL2Reader(InternLM2Reader):
    """InternVLReader for InternLM2 model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'language_model.model.tok_embeddings.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.output.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('llm_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


class InternVL3d5Reader(Qwen3Reader):
    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('llm_config') or model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


class InternVL3d5Qwen3MoEReader(Qwen3MoeReader):
    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('llm_config') or model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


class InternVL3d5GptOSSReader(GptOssReader):
    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('llm_config') or model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


class InternS1Reader(Qwen3MoeReader):
    """InternS1Reader for internlm/InternS1 model."""

    attn_layer_prefix = 'model.language_model.layers'
    attn_layer_patten = r'model\.language_model\.layers\.([0-9]+).'
    tok_embeddings_key = 'model.language_model.embed_tokens.weight'
    norm_weight_key = 'model.language_model.norm.weight'
    output_weight_key = 'lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('text_config')
        if model_cfg is None:
            raise ValueError(f'Miss "text_config" in model config: {model_cfg}')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


class InternS1MiniReader(Qwen3Reader):

    attn_layer_prefix = 'model.language_model.layers'
    attn_layer_patten = r'model\.language_model\.layers\.([0-9]+).'
    tok_embeddings_key = 'model.language_model.embed_tokens.weight'
    norm_weight_key = 'model.language_model.norm.weight'
    output_weight_key = 'lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('text_config')
        if model_cfg is None:
            raise ValueError(f'Miss "text_config" in model config: {model_cfg}')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


@INPUT_MODELS.register_module(name='internvl')
class InternVLModel(LlamaModel):
    """InternVL model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        arch = config.architectures[0]
        if arch == 'InternVLChatModel' or arch == 'InternVLForConditionalGeneration':
            relations = dict(InternLM2ForCausalLM=('internlm2', InternVL2Reader),
                             LlamaForCausalLM=('llama', InternVLReader),
                             Qwen2ForCausalLM=('qwen2', InternVLReader),
                             Qwen3MoeForCausalLM=('qwen3-moe', InternVL3d5Qwen3MoEReader),
                             Qwen3ForCausalLM=('qwen3', InternVL3d5Reader),
                             GptOssForCausalLM=('gpt-oss', InternVL3d5GptOSSReader))
        elif arch == 'InternS1ForConditionalGeneration':
            relations = dict(Qwen3MoeForCausalLM=('qwen3-moe', InternS1Reader),
                             Qwen3ForCausalLM=('qwen3', InternS1MiniReader))
        else:
            raise ValueError('unsupported model arch {arch}')
        self.llm_config = getattr(config, 'llm_config', None) or getattr(config, 'text_config', None)
        arch = self.llm_config.architectures[0]
        llm_model, self.Reader = relations[arch]
        self.llm_model = INPUT_MODELS.get(llm_model)(model_path=model_path, tokenizer_path=tokenizer_path, **kwargs)

    def model_info(self):
        """Read model info."""
        self.llm_model.model_config = self.llm_config.to_dict()
        return self.llm_model.model_info()
