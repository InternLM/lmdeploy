# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .glm4_moe import Glm4MoE, Glm4MoeAttention, Glm4MoeDecoderLayer, Glm4MoeMLP
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


class Glm4MoeMTPDecoderLayer(Glm4MoeDecoderLayer):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config, layer_idx, dtype=dtype, device=device)
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


class SharedHead(nn.Module):
    """Deepseekv2 shared head."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        # build lm_head
        self.head = build_rowwise_linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Glm4MoeMultiTokenPredictorLayer(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        quantization_config = getattr(config, 'quantization_config', None)

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.eh_proj = build_colwise_linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
            dp_disable_tp=True,
        )

        self.shared_head = SharedHead(config=config, dtype=dtype, device=device)

        self.mtp_block = Glm4MoeMTPDecoderLayer(config, layer_idx=layer_idx, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = build_rotary_embedding_from_config(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_value: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None

        # masking inputs at position 0, as not needed by MTP
        inputs_embeds[position_ids == 0] = 0
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        # rotary emb
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        hidden_states, residual = self.mtp_block(
            hidden_states,
            rotary_pos_emb,
            past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states
        return hidden_states


class Glm4MoeMultiTokenPredictor(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx):
            Glm4MoeMultiTokenPredictorLayer(
                config,
                idx,
                dtype=dtype,
                device=device,
            )
            for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        past_key_value = past_key_values[current_step_idx]
        return self.layers[str(layer_idx)](
            input_ids,
            position_ids,
            previous_hidden_states,
            past_key_value,
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
            spec_step_index=current_step_idx,
        )

    def get_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]

        hidden_states = mtp_layer.shared_head(hidden_states)
        logits = mtp_layer.shared_head.head(hidden_states)
        return logits


class Glm4MoeMTPModel(nn.Module, CudaGraphMixin):
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
        super().__init__()
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = Glm4MoeMultiTokenPredictor(config, dtype=dtype, device=device)

    def get_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
        """Compute logits of the model output."""
        return self.model.get_logits(hidden_states, spec_step_idx=spec_step_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids,
                                   position_ids,
                                   target_hidden_states,
                                   inputs_embeds=inputs_embeds,
                                   past_key_values=past_key_values,
                                   attn_metadata=attn_metadata,
                                   spec_step_idx=spec_step_idx)
        return hidden_states

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers['input_ids'].new_zeros(1,
                                                                                     max_tokens,
                                                                                     self.config.hidden_size,
                                                                                     dtype=self.dtype)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, input_ids: torch.Tensor, **kwargs):
        """Fill cudagraph buffers from forward inputs."""

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, input_ids=input_ids, **kwargs)

        num_tokens = input_ids.size(-1)
        input_buffers = graph_meta.input_buffers
        target_hidden_states = kwargs.get('target_hidden_states')
        assert target_hidden_states is not None
        input_buffers['target_hidden_states'][:, :num_tokens] = target_hidden_states
        new_inputs['target_hidden_states'] = input_buffers['target_hidden_states']
        return new_inputs

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        target_hidden_states = context.target_hidden_states
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            target_hidden_states=target_hidden_states,
        )

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                             expert_params_mapping: List):
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
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

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """Rewrite the weight name to match the format of the original model.

        Add .mtp_block for modules in transformer layer block for spec layer
        """
        spec_layer_weight_names = ['embed_tokens', 'enorm', 'hnorm', 'eh_proj', 'shared_head']
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(f'model.layers.{spec_layer}.', f'model.layers.{spec_layer}.mtp_block.')
        return name
