# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .qwen3_moe import Qwen3MoeModel
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .qwen3_vl import Qwen3VLTextRotaryEmbedding as Qwen3VLMoeTextRotaryEmbedding


class Qwen3VLMoeTextModel(Qwen3MoeModel):
    """Text part of Qwen3VL.

    not a pure text-only model, as DeepStack integrates visual features into the early hidden states.
    """

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__(config=config, dtype=dtype, device=device)

        # build rotary embedding
        # TODO: zhouxinyu, add triton kernel for interleaved mrope
        self.rotary_emb = Qwen3VLMoeTextRotaryEmbedding(config, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mrope_position_ids: torch.LongTensor = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ):
        """visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`,
        *optional*):

        The mask of the visual positions. deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):     The deepstack
        visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).     The feature is extracted from the
        different visual encoder layers, and fed to the decoder     hidden states. It's from the paper DeepStack (
        https://arxiv.org/abs/2406.04)
        """

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        if mrope_position_ids is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            mrope_position_ids = mrope_position_ids.unsqueeze(1)
            cos, sin = self.rotary_emb(hidden_states, mrope_position_ids)

        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and idx in range(len(deepstack_visual_embeds)):
                hidden_states = hidden_states + residual
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[idx],
                )
                residual = None

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local = torch.zeros_like(hidden_states)
        local.masked_scatter_(visual_pos_masks, visual_embeds)
        hidden_states += local
        return hidden_states


class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
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
        super().__init__(config=config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)

        self.language_model = Qwen3VLMoeTextModel(config.text_config, dtype=dtype, device=device)

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                             expert_params_mapping: List):
        """Load weight experts."""

        for (param_name, weight_name, expert_id, shard_id) in expert_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            load_weight(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
        else:
            param = params_dict[name]
            load_weight(param, loaded_weight)

    # modify from vllm qwen3vlmoe fused expert loading
    def _load_weight_fused_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                                   fused_expert_params_mapping: List):
        """Load weight of fused expert weights."""
        num_experts = self.config.text_config.num_experts

        for (param_name, weight_name) in fused_expert_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]

            loaded_weight = loaded_weight.transpose(-1, -2)  # no bias
            if 'gate_up' in name:
                loaded_weight = loaded_weight.chunk(2, dim=-2)
                w1 = loaded_weight[0]
                w3 = loaded_weight[1]
                for expert_id in range(num_experts):
                    load_weight(param, w1[expert_id], expert_id=expert_id, shard_id='gate')
                    load_weight(param, w3[expert_id], expert_id=expert_id, shard_id='up')
            elif 'down' in name:
                w2 = loaded_weight
                for expert_id in range(num_experts):
                    load_weight(param, w2[expert_id], expert_id=expert_id, shard_id='down')

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        # expert mapping
        num_experts = self.config.text_config.num_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            # (param_name, weight_name, expert_id, shard_id)
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        # fused expert mapping
        fused_expert_params_mapping = [
            # (param_name, weight_name)
            ('.experts.gate_up.weight', '.experts.gate_up_proj'),
            ('.experts.down.weight', '.experts.down_proj'),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            name = name.replace('.block_sparse_moe.', '.mlp.')
            if '.experts' in name:
                is_fused_expert = ('experts.gate_up_proj' in name or 'experts.down_proj' in name)
                if is_fused_expert:
                    self._load_weight_fused_experts(name,
                                                    loaded_weight,
                                                    params_dict,
                                                    fused_expert_params_mapping=fused_expert_params_mapping)
                else:
                    self._load_weight_experts(name,
                                              loaded_weight,
                                              params_dict,
                                              expert_params_mapping=expert_params_mapping)
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
                        param = params_dict[name]
                        q, k, v = param.weight_spliter(loaded_weight)
                        load_weight(param, q, shard_id='q')
                        load_weight(param, k, shard_id='k')
                        load_weight(param, v, shard_id='v')
                    else:
                        param = params_dict[name]
                        load_weight(param, loaded_weight)
