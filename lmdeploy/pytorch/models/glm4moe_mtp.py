# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Iterable

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, build_rotary_embedding_from_config
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .deepseek_mtp import DeepseekMTPModel
from .glm4_moe import Glm4MoE, Glm4MoeAttention, Glm4MoeDecoderLayer, Glm4MoeMLP


class Glm4MoeMTPDecoderLayer(Glm4MoeDecoderLayer):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = Glm4MoeAttention(config, dtype=dtype, device=device, is_tp=False)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm4MoE(config, layer_idx=layer_idx, dtype=dtype, device=device, is_tp=False)
            self.mlp._all_reduce = False
        else:
            self.mlp = Glm4MoeMLP(config, dtype=dtype, device=device, is_tp=False, all_reduce=False)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)


class Glm4MoeMTPModel(DeepseekMTPModel):
    """ModelForCausalLM."""

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(
            config,
            ctx_mgr,
            dtype=dtype,
            device=device,
            decoder_layer_cls=Glm4MoeMTPDecoderLayer,
            build_rotary_embedding_func=build_rotary_embedding_from_config,
        )

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                             expert_params_mapping: list[list[str]]):
        """Load weight experts."""
        for (param_name, weight_name, expert_id, shard_id) in expert_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            load_weight(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
            break
        else:
            param = params_dict[name]
            load_weight(param, loaded_weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights."""

        def __skip_nextn(name, nextn_keys):
            for nextn_key in nextn_keys:
                if nextn_key in name:
                    return True
            return False

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        num_hidden_layers = self.config.num_hidden_layers

        num_nextn_predict_layers = getattr(self.config, 'num_nextn_predict_layers', 1)
        nextn_keys = [f'.layers.{num_hidden_layers+i}' for i in range(num_nextn_predict_layers)]

        # expert map
        num_experts = self.config.n_routed_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # keep nextn
            if not __skip_nextn(name, nextn_keys):
                continue
            if '.layers' in name:
                layer_idx = int(name.split('layers.')[1].split('.')[0])
                name = self._rewrite_spec_layer_name(layer_idx, name)
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            name = name.replace('.block_sparse_moe.', '.mlp.')
            if '.experts' in name:
                self._load_weight_experts(name, loaded_weight, params_dict, expert_params_mapping=expert_params_mapping)
            else:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    load_weight(param, loaded_weight, shard_id=shard_id)
                    break
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
