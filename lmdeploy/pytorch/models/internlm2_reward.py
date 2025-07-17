# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn.linear import build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .internlm2 import InternLM2Model
from .utils.cudagraph import CudaGraphMixin


class InternLM2ForRewardModel(nn.Module, CudaGraphMixin):
    """Rewrote model of InternLM2ForRewardModel."""

    packed_modules_mapping = {
        'gate_up_proj': [
            'w1',
            'w3',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build Model
        self.model = InternLM2Model(config, dtype=dtype, device=device)
        # build v_head
        self.v_head = build_rowwise_linear(config.hidden_size, 1, bias=False, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.v_head(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        vision_embeddings = context.input_embeddings
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            raise ValueError('InternLM2RewardModel does not support vision embedding')

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def load_lora_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], adapter_id: int):
        """Load lora weights."""

        from lmdeploy.pytorch.adapter.adapter import load_lora_weights

        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        hidden_size = self.config.hidden_size
        head_dim = hidden_size // num_heads
        group_size = num_heads // num_key_value_heads

        def _rearange_wqkv(weights):
            for name, loaded_weight in weights:
                if 'wqkv.lora_B' in name:
                    loaded_weight = loaded_weight.unflatten(0, (-1, 2 + group_size, head_dim))
                    q = loaded_weight[:, :-2].flatten(0, 2)
                    k = loaded_weight[:, -2].flatten(0, 1)
                    v = loaded_weight[:, -1].flatten(0, 1)
                    loaded_weight = torch.cat([q, k, v], dim=0)
                yield name, loaded_weight

        weights_iter = _rearange_wqkv(weights)
        load_lora_weights(self, weights_iter, adapter_id)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.w1', 0),
            ('.gate_up_proj', '.w3', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if '.wqkv' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight, layout='hgd')
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
