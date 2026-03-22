# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Iterable

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import RMSNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import add_prefix, get_build_model_context
from .qwen3_5 import Qwen3_5Attention, Qwen3_5DecoderLayer, Qwen3_5MLP, Qwen3_5TextRotaryEmbedding
from .qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


class Qwen3_5MtpDecoderLayer(Qwen3_5DecoderLayer):
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


class Qwen3_5MultiTokenPredictor(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()
        self.config = config
        self.mtp_start_layer_idx = 0
        self.num_mtp_layers = config.mtp_num_hidden_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx):
            Qwen3_5MtpDecoderLayer(
                config,
                self.mtp_start_layer_idx + idx,
                dtype=dtype,
                device=device,
            )
            for idx in range(self.num_mtp_layers)
        })

        quantization_config = getattr(config, 'quantization_config', None)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)

        # shared with target model
        self.embed_tokens = None

        self.fc = build_colwise_linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
            dp_disable_tp=True,
        )

        # build rotary embedding
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=device)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, embed_tokens: nn.Embedding):
        """Set embed tokens."""
        self.embed_tokens = embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        all_routed_experts: torch.Tensor | None = None,
        mrope_position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        past_key_value = past_key_values[current_step_idx]

        # TODO: fix input mrope position ids
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        previous_hidden_states = self.pre_fc_norm_hidden(previous_hidden_states)
        hidden_states = torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)

        # rotary embedding
        if mrope_position_ids is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            mrope_position_ids = mrope_position_ids.unsqueeze(1)
            cos, sin = self.rotary_emb(hidden_states, mrope_position_ids)

        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        hidden_states, residual = self.layers[str(layer_idx)](
            hidden_states,
            rotary_pos_emb,
            past_key_value,
            attn_metadata=attn_metadata,
            all_routed_experts=all_routed_experts,
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5MTPModel(nn.Module, CudaGraphMixin):
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
                 device: torch.device | None = None,
                 prefix: str = ''):

        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype

        self.model = Qwen3_5MultiTokenPredictor(config.text_config,
                                                dtype=dtype,
                                                device=device,
                                                prefix=add_prefix('model', prefix=prefix))

        self.num_experts = getattr(config.text_config, 'num_experts', None)
        # for router replay

        self.enable_return_routed_experts = False
        if self.num_experts is not None:
            bm_ctx = get_build_model_context()
            self.enable_return_routed_experts = bm_ctx.enable_return_routed_experts
            # TODO support later
            self.enable_return_routed_experts = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any,
        inputs_embeds: torch.Tensor | None = None,
        mrope_position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        all_routed_experts = None
        if self.enable_return_routed_experts:
            config = self.config.text_config
            num_tokens = input_ids.size(1)
            all_routed_experts = position_ids.new_empty(
                (num_tokens, config.mtp_num_hidden_layers, config.num_experts_per_tok), dtype=torch.uint16)

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
            previous_hidden_states=target_hidden_states,
            all_routed_experts=all_routed_experts,
        )
        if all_routed_experts is None:
            return hidden_states
        return dict(hidden_states=hidden_states, all_routed_experts=all_routed_experts)

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers['input_ids'].new_zeros(
            1, max_tokens, self.config.text_config.hidden_size, dtype=self.dtype)

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

    def set_input_embeddings(self, embed_tokens: nn.Embedding):
        """Set embed tokens."""
        self.model.set_input_embeddings(embed_tokens)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        target_hidden_states = context.target_hidden_states
        mrope_position_ids = getattr(context, 'mrope_position_ids', None)
        if context.target_inputs_embeds is not None:
            inputs_embeds = context.target_inputs_embeds

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            target_hidden_states=target_hidden_states,
            mrope_position_ids=mrope_position_ids,
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
        num_experts = self.config.text_config.num_experts
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

        expert_params_mapping = []
        # expert map
        if self.num_experts is not None:
            for exp_id in range(self.num_experts):
                gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
                up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
                down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
                expert_params_mapping += [gate_param, up_param, down_param]

        rms_norm_keys = [
            'model.norm', '.input_layernorm', '.post_attention_layernorm', '.q_norm', '.k_norm', 'mtp.norm',
            '.pre_fc_norm_embedding', '.pre_fc_norm_hidden'
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith('mtp.'):
                continue
            name = name.replace('mtp.', 'model.')
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
