# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-4 MoE Lite (e.g. GLM-4.7-Flash) source model for TurboMind.

Architecture: MLA (Multi-head Latent Attention) + MoE with dense first layer.
Weight layout follows HuggingFace checkpoint with model.layers.* (same family as DeepSeek2).
"""

from .base import INPUT_MODELS
from .deepseek2 import DeepSeek2Model, DeepSeek2Reader


class Glm4MoeLiteReader(DeepSeek2Reader):
    """Reader for Glm4MoeLiteForCausalLM (GLM-4.7-Flash).

    Uses same key layout as DeepSeek2: model.layers.{i}.self_attn.*, model.layers.{i}.mlp.*
    Supports noaux_tc via e_score_correction_bias.
    """

    attn_layer_prefix = 'model.layers'
    attn_layer_patten = r'model\.layers\.([0-9]+).'
    tok_embeddings_key = 'model.embed_tokens.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'lm_head.weight'

    def moe_ffn_gate_correction_bias(self, i: int):
        """Per-expert score correction bias for noaux_tc routing."""
        return self.params.get(f'{self.attn_layer_prefix}.{i}.mlp.gate.e_score_correction_bias')


@INPUT_MODELS.register_module(name='glm4-moe-lite')
class Glm4MoeLiteModel(DeepSeek2Model):
    """GLM-4 MoE Lite (e.g. GLM-4.7-Flash) in HF format.

    MLA + MoE with first_k_dense_replace; config mapping aligned to DeepSeek2.
    """

    Reader = Glm4MoeLiteReader

    def model_info(self):
        cfg = self.model_config
        # Set default MoE routing config for GLM-4 MoE Lite if not in HF config
        if 'topk_method' not in cfg:
            cfg['topk_method'] = 'noaux_tc'
        if 'topk_group' not in cfg:
            cfg['topk_group'] = 1
        if 'n_group' not in cfg:
            cfg['n_group'] = 1
        if 'scoring_func' not in cfg:
            cfg['scoring_func'] = 'sigmoid'

        info = super().model_info()
        # GLM4 MoE Lite uses noaux_tc routing with sigmoid scoring
        info['topk_method'] = 'noaux_tc'
        info['scoring_func'] = 'sigmoid'
        if 'router_n_groups' in cfg and cfg['router_n_groups'] > 0:
            info['router_n_groups'] = cfg['router_n_groups']

        return info
