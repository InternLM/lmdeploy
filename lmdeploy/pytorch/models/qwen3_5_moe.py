# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import RMSNorm
from lmdeploy.pytorch.nn.linear import build_rowwise_linear
from lmdeploy.pytorch.nn.moe import SoftmaxTopK, build_fused_moe
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .qwen2_5_vl import Qwen2_5_VLInputProcessor as Qwen3_5MoeInputProcessor
from .qwen3_5 import (Qwen3_5Attention, Qwen3_5DecoderLayer, Qwen3_5ForConditionalGeneration, Qwen3_5GatedDeltaNet,
                      Qwen3_5MLP, Qwen3_5Model, Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding)
from .qwen3_5 import Qwen3_5VisionModel as Qwen3_5MoeVisionModel


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        # TODO: zhouxinyu, determine modules_to_not_convert from config file
        quantization_config = getattr(config, 'quantization_config', None)
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.renormalize = self.norm_topk_prob

        self.gate = build_rowwise_linear(
            self.hidden_dim,
            self.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )

        self.softmax_topk = SoftmaxTopK(
            self.top_k,
            n_groups=getattr(config, 'router_n_groups', -1),
        )

        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            renormalize=self.renormalize,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            all_reduce=False,
            layer_idx=layer_idx,
        )

        self.shared_expert = Qwen3_5MLP(
            config=config,
            intermediate_size=config.shared_expert_intermediate_size,
            dtype=dtype,
            device=device,
            is_tp=True,
            all_reduce=False,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False, device=device, dtype=dtype)

        # get all reduce
        dist_ctx = get_dist_manager().current_context()
        dp = dist_ctx.dist_config.dp
        world_size = dist_ctx.dist_config.moe_tp
        if dp == 1 and world_size > 1:
            self._all_reduce = True
        else:
            self._all_reduce = False

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self.softmax_topk(router_logits)
        out_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        shared_states = self.shared_expert(hidden_states)
        shared_states = self.shared_expert_gate(hidden_states).sigmoid() * shared_states

        out_states += shared_states
        out_states = out_states.reshape(batch_size, sequence_length, -1)

        if self._all_reduce:
            dist.all_reduce(out_states)
        return out_states


class Qwen3_5MoeDecoderLayer(Qwen3_5DecoderLayer):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, dtype=dtype, device=device)
        elif self.layer_type == 'full_attention':
            self.self_attn = Qwen3_5Attention(config, layer_idx, dtype=dtype, device=device)

        # build MLP
        self.mlp = Qwen3_5MoeSparseMoeBlock(config, layer_idx, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)


class Qwen3_5MoeTextModel(Qwen3_5TextModel):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype | None = None, device: torch.device | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        # TODO: use full config.num_hidden_layers
        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(self.config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=device)


class Qwen3_5MoeModel(Qwen3_5Model):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype | None = None, device: torch.device | None = None):
        nn.Module.__init__(self)

        self.visual = Qwen3_5MoeVisionModel(config.vision_config, dtype=dtype, device=device)
        self.language_model = Qwen3_5TextModel(config.text_config, dtype=dtype, device=device)


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
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
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.ctx_mgr = ctx_mgr

        # build preprocessor
        self.input_processor = Qwen3_5MoeInputProcessor(self.config)

        # build model
        self.model = Qwen3_5MoeModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                             expert_params_mapping: List):
        """Load weight experts."""
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        def __skip_layers(name):
            """We might change the number of layers so we can debug the model
            with less gpus."""
            import re
            if '.layers.' not in name:
                return False
            matches = re.findall(r'\.layers\.(\d+)\.', name)
            layer_id = int(matches[0])
            return layer_id >= self.config.text_config.num_hidden_layers

        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        # expert map
        num_experts = self.config.num_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        rms_norm_keys = ['model.norm', '.input_layernorm', '.post_attention_layernorm', '.q_norm', '.k_norm']

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            if __skip_layers(name):
                continue

            if 'mtp.' in name:
                continue
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            name = name.replace('.block_sparse_moe.', '.mlp.')
            if '.experts' in name and '.shared_expert' not in name:
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
