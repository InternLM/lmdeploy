# Copyright (c) OpenMMLab. All rights reserved.

from typing import Iterable

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, build_rotary_embedding_from_config
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .deepseek_mtp import DeepseekMTPModel
from .qwen3_5 import Qwen3_5Attention, Qwen3_5MLP
from .qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeSparseMoeBlock


class Qwen3_5MoeMtpDecoderLayer(Qwen3_5MoeDecoderLayer):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.layer_type = 'full_attention'
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = Qwen3_5Attention(config, layer_idx, dtype=dtype, device=device, is_tp=False)

        # build MLP
        if 'moe' in config.model_type.lower():
            self.mlp = Qwen3_5MoeSparseMoeBlock(config, layer_idx, dtype=dtype, device=device, is_tp=False)
            self.mlp._all_reduce = False
        else:
            self.mlp = Qwen3_5MLP(config, dtype=dtype, device=device, is_tp=False, all_reduce=False)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)


class Qwen3_5MTPModel(DeepseekMTPModel):
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

        text_config = config.text_config
        text_config.num_nextn_predict_layers = text_config.mtp_num_hidden_layers

        super().__init__(
            text_config,
            ctx_mgr,
            dtype=dtype,
            device=device,
            decoder_layer_cls=Qwen3_5MoeMtpDecoderLayer,
            build_rotary_embedding_func=build_rotary_embedding_from_config,
        )

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                             expert_params_mapping: list[tuple[str]]):
        """Load weight experts."""
        # this func is not used, but it has same layout with tranformers implementation
        # so I will keep it for now.
        # load fused weights
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

    def _load_weight_fused_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter]):
        """Load weight of fused expert weights."""
        num_experts = self.config.num_experts
        fused_gateup_name = 'gate_up_proj'
        fused_down_name = 'down_proj'
        if fused_gateup_name in name:

            for expert_id in range(num_experts):
                param_name = name.replace(f'experts.{fused_gateup_name}', 'experts.gate_up.weight')
                param = params_dict[param_name]
                weight = loaded_weight[expert_id]
                w1, w3 = weight.chunk(2, 0)
                load_weight(param, w1, expert_id=expert_id, shard_id='gate')
                load_weight(param, w3, expert_id=expert_id, shard_id='up')

        elif fused_down_name in name:

            for expert_id in range(num_experts):
                param_name = name.replace(f'experts.{fused_down_name}', 'experts.down.weight')
                param = params_dict[param_name]
                w2 = loaded_weight[expert_id]
                load_weight(param, w2, expert_id=expert_id, shard_id='down')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
            ('.in_proj_zba', '.in_proj_z', 'z'),
            ('.in_proj_zba', '.in_proj_b', 'b'),
            ('.in_proj_zba', '.in_proj_a', 'a'),
        ]

        num_hidden_layers = self.config.num_hidden_layers
        num_nextn_predict_layers = self.config.mtp_num_hidden_layers
        assert num_nextn_predict_layers == 1, f'only support 1, but given: {num_nextn_predict_layers}'
        mtp_layer_id = num_hidden_layers
        expert_params_mapping = []

        # expert map
        if hasattr(self.config, 'num_experts'):
            num_experts = self.config.num_experts
            for exp_id in range(num_experts):
                gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
                up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
                down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
                expert_params_mapping += [gate_param, up_param, down_param]

        rms_norm_keys = [
            'model.norm', '.input_layernorm', '.post_attention_layernorm', '.q_norm', '.k_norm', '.norm', '.enorm',
            '.hnorm'
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if '.layers.' in name:
                if not name.startswith('mtp.'):
                    continue
                layer_id, suffix = name.split('.layers.')[1].split('.', 1)
                layer_id = int(layer_id) + num_hidden_layers
                name = f'model.layers.{layer_id}.mtp_block.{suffix}'
            elif name.startswith('mtp.') or name in ['lm_head.weight', 'model.language_model.embed_tokens.weight']:
                mtp_prefix = f'model.layers.{mtp_layer_id}.'
                name_mapping = {
                    'mtp.fc.weight': mtp_prefix + 'eh_proj.weight',
                    'mtp.norm.weight': mtp_prefix + 'shared_head.norm.weight',
                    'mtp.pre_fc_norm_embedding.weight': mtp_prefix + 'enorm.weight',
                    'mtp.pre_fc_norm_hidden.weight': mtp_prefix + 'hnorm.weight',
                    'lm_head.weight': mtp_prefix + 'shared_head.head.weight',
                    'model.language_model.embed_tokens.weight': mtp_prefix + 'embed_tokens.weight'
                }
                assert name in name_mapping, name
                name = name_mapping[name]
            else:
                continue

            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            if '.experts' in name and '.shared_expert' not in name:
                if name.split('.experts.', 1)[1].split('.', 1)[0].isdigit():
                    self._load_weight_experts(name, loaded_weight, params_dict, expert_params_mapping)
                else:
                    self._load_weight_fused_experts(name, loaded_weight, params_dict)
            else:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    load_weight(param, loaded_weight, shard_id=shard_id)
                    break
                else:
                    if '.qkv.' in name:
                        # vl attention
                        param = params_dict[name]
                        q, k, v = param.weight_spliter(loaded_weight)
                        load_weight(param, q, shard_id='q')
                        load_weight(param, k, shard_id='k')
                        load_weight(param, v, shard_id='v')
                    else:
                        for rms_norm_key in rms_norm_keys:
                            if rms_norm_key in name and 'weight' in name:
                                loaded_weight = loaded_weight + 1
                                break
                        param = params_dict[name]
                        load_weight(param, loaded_weight)
