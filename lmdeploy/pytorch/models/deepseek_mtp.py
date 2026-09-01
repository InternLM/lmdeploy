# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (
    ParallelLMHead,
    RMSNorm,
    RopeType,
    build_rotary_embedding,
    build_rotary_params,
)
from lmdeploy.pytorch.nn.linear import build_colwise_linear
from lmdeploy.pytorch.nn.rotary_embedding import get_rope_theta
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.utils import get_logger

from .deepseek_v2 import DeepseekV2DecoderLayer
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin

logger = get_logger('lmdeploy')


class SharedHead(nn.Module):
    """DeepSeek MTP normalization and vocab-parallel head."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        # build lm_head
        self.head = ParallelLMHead(config.vocab_size,
                                   config.hidden_size,
                                   bias=False,
                                   dtype=dtype,
                                   device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


def build_deepseek_rotary_embedding(config: PretrainedConfig):
    """Build deepseek rotary embedding."""
    emb_type = RopeType.LinearScaling
    rope_dim = config.qk_rope_head_dim
    rope_max_pos_emb = config.max_position_embeddings
    rope_base = get_rope_theta(config)

    rope_params = dict(emb_type=emb_type, dim=rope_dim, max_position_embeddings=rope_max_pos_emb, base=rope_base)
    update_params = build_rotary_params(config)
    rope_params.update(update_params)
    return build_rotary_embedding(**rope_params)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    """DeepSeek MTP layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
        decoder_layer_cls=DeepseekV2DecoderLayer,
        build_rotary_embedding_func=build_deepseek_rotary_embedding,
    ) -> None:
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
        # Keep the recurrent MTP projection replicated across TP ranks.
        self.eh_proj = build_colwise_linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
        )

        self.shared_head = SharedHead(config=config, dtype=dtype, device=device)

        self.mtp_block = decoder_layer_cls(config, layer_idx=layer_idx, dtype=dtype, device=device)

        self.rotary_emb = build_rotary_embedding_func(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_value: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: Any = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None

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


class DeepSeekMultiTokenPredictor(nn.Module):
    """DeepSeek multi-token predictor."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: torch.dtype = None,
        device: torch.device = None,
        decoder_layer_cls=DeepseekV2DecoderLayer,
        build_rotary_embedding_func=build_deepseek_rotary_embedding,
    ):
        super().__init__()
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx):
            DeepSeekMultiTokenPredictorLayer(
                config,
                idx,
                dtype=dtype,
                device=device,
                decoder_layer_cls=decoder_layer_cls,
                build_rotary_embedding_func=build_rotary_embedding_func,
            )
            for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
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


class DeepseekMTPModel(nn.Module, CudaGraphMixin):
    """DeepSeek MTP model."""

    def __init__(
        self,
        config: PretrainedConfig,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype = None,
        device: torch.device = None,
        decoder_layer_cls=DeepseekV2DecoderLayer,
        build_rotary_embedding_func=build_deepseek_rotary_embedding,
    ):
        super().__init__()
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = DeepSeekMultiTokenPredictor(config,
                                                 dtype=dtype,
                                                 device=device,
                                                 decoder_layer_cls=decoder_layer_cls,
                                                 build_rotary_embedding_func=build_rotary_embedding_func)

        self._load_buffers = dict()

    def get_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
        """Compute logits of the model output."""
        return self.model.get_logits(hidden_states, spec_step_idx=spec_step_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
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
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
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

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                             expert_params_mapping: list):
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

    def _load_weight_attention(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                               update_pe_mapping: list):
        """Load weight attention."""
        device = next(iter(params_dict.values())).device

        def __update_pe(weight, head_dim: int, pe_dim_offset: int):
            # (num_heads, q_head_dim, input_dim)
            weight = weight.unflatten(0, (-1, head_dim))
            # (num_heads, nope_head_dim, input_dim)
            w_pe = weight[:, pe_dim_offset:]
            # (num_heads, nope_head_dim//2, 2, input_dim)
            new_w_pe = w_pe.unflatten(1, (-1, 2))
            # (num_heads, nope_head_dim, input_dim)
            new_w_pe = new_w_pe.transpose(1, 2).flatten(1, 2)
            weight[:, pe_dim_offset:] = new_w_pe
            weight = weight.flatten(0, 1)
            return weight

        def __load_kcvc(name: str, weight: torch.Tensor):
            """Load kc and vc from weight."""
            config = self.config
            v_head_dim = config.v_head_dim
            qk_nope_head_dim = config.qk_nope_head_dim
            w_kc, w_vc = weight.unflatten(0, (-1, qk_nope_head_dim + v_head_dim)).split([qk_nope_head_dim, v_head_dim],
                                                                                        dim=1)
            w_vc = w_vc.transpose(1, 2).contiguous()
            kc_param_name = name.replace('.kv_b_proj', '.kc')
            param_kc = params_dict[kc_param_name]
            load_weight(param_kc, w_kc)
            vc_param_name = name.replace('.kv_b_proj', '.vc')
            param_vc = params_dict[vc_param_name]
            load_weight(param_vc, w_vc)

        def __dequant_weight(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
            """Dequant weight."""
            dim_w0, dim_w1 = weight.shape
            dim_s0, dim_s1 = scale.shape
            assert dim_w0 % dim_s0 == 0
            assert dim_w1 % dim_s1 == 0
            group0 = dim_w0 // dim_s0
            group1 = dim_w1 // dim_s1
            weight = weight.reshape(dim_s0, group0, dim_s1, group1)
            scale = scale.reshape(dim_s0, 1, dim_s1, 1)
            weight = weight.to(scale.dtype) * scale
            weight = weight.to(dtype)
            weight = weight.reshape(dim_w0, dim_w1)
            return weight

        def __load_kcvc_blocked_fp8(name: str, loaded_weight: torch.Tensor):
            """Dequant weight."""
            if name.endswith('.weight'):
                weight_name = name
                scale_name = name.replace('.weight', '.scale')
            elif name.endswith('.weight_scale_inv'):
                weight_name = name.replace('.weight_scale_inv', '.weight')
                scale_name = name
            self._load_buffers[name] = loaded_weight
            if (weight_name in self._load_buffers and scale_name in self._load_buffers):
                weight = self._load_buffers.pop(weight_name)
                scale = self._load_buffers.pop(scale_name)
                kc_param_name = weight_name.replace('.kv_b_proj', '.kc')
                dtype = params_dict[kc_param_name].dtype
                weight = __dequant_weight(weight, scale, dtype)
                __load_kcvc(weight_name, weight)

        for (mod_name, head_dim, pe_dim_offset) in update_pe_mapping:
            if mod_name not in name:
                continue
            if name.endswith('.weight_scale_inv'):
                weight = loaded_weight
            else:
                loaded_weight = loaded_weight.to(device)
                weight = __update_pe(loaded_weight, head_dim, pe_dim_offset)
            param = params_dict[name]
            load_weight(param, weight)
            break
        else:
            if '.kv_b_proj' in name:
                quantization_config = self.quantization_config
                quant_method = None
                if quantization_config is not None:
                    quant_method = quantization_config.get('quant_method')

                loaded_weight = loaded_weight.to(device)
                if quant_method == 'fp8':
                    # update blocked fp8 weight
                    __load_kcvc_blocked_fp8(name, loaded_weight)
                else:
                    __load_kcvc(name, loaded_weight)
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
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        config = self.config

        qk_rope_head_dim = config.qk_rope_head_dim
        kv_lora_rank = config.kv_lora_rank
        qk_nope_head_dim = config.qk_nope_head_dim
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        kv_dim = kv_lora_rank + qk_rope_head_dim
        update_pe_mapping = [('q_proj', q_head_dim, qk_nope_head_dim), ('q_b_proj', q_head_dim, qk_nope_head_dim),
                             ('kv_a_proj_with_mqa', kv_dim, kv_lora_rank)]

        num_experts = self.config.n_routed_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        num_hidden_layers = self.config.num_hidden_layers

        num_nextn_predict_layers = getattr(self.config, 'num_nextn_predict_layers', 1)
        nextn_keys = [f'.layers.{num_hidden_layers+i}' for i in range(num_nextn_predict_layers)]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # keep nextn
            if not __skip_nextn(name, nextn_keys):
                continue
            if '.layers' in name:
                layer_idx = int(name.split('layers.')[1].split('.')[0])
                name = self._rewrite_spec_layer_name(layer_idx, name)
            if '.experts' in name:
                self._load_weight_experts(name, loaded_weight, params_dict, expert_params_mapping=expert_params_mapping)
            elif '.self_attn' in name:
                # attention
                self._load_weight_attention(name, loaded_weight, params_dict, update_pe_mapping)
            else:
                # other
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
