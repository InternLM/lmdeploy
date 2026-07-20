# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.model_inputs import (
    StepContext,
    StepContextManager,
)
from lmdeploy.pytorch.nn import (
    RMSNorm,
    build_rotary_embedding_from_config,
)
from lmdeploy.pytorch.nn.moe import build_fused_moe
from lmdeploy.pytorch.nn.moe.route import NoauxTCRouter
from lmdeploy.pytorch.weight_loader.model_weight_loader import (
    load_weight,
)

from .patch import add_prefix
from .qwen3_moe import Qwen3MoeAttention, Qwen3MoeMLP
from .utils.cudagraph import CudaGraphMixin
from .utils.model import (
    DeployModelMixinV1,
    build_embedding,
)


class Hy3Attention(Qwen3MoeAttention):
    """Attention implementation for Hy3."""
    pass

class Hy3MLP(Qwen3MoeMLP):
    """MLP implementation for Hy3."""
    pass

class Hy3Router(nn.Module):
    """Sigmoid top-k router for Hy3."""

    def __init__(
        self,
        config,
        device: torch.device = None,
    ):
        super().__init__()

        self.gate = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            dtype=torch.float32,
            device=device,
        )

        self.topk_router = NoauxTCRouter(
            scoring_func='sigmoid',
            top_k=config.num_experts_per_tok,
            n_group=1,
            topk_group=1,
            n_routed_experts=config.num_experts,
            routed_scaling_factor=(
                config.router_scaling_factor
            ),
            renormalize=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_bias: torch.Tensor,
    ):
        hidden_states = hidden_states.reshape(
            -1,
            hidden_states.shape[-1],
        )

        router_logits = self.gate(
            hidden_states.float()
        )

        topk_weights, topk_ids = self.topk_router(
            router_logits,
            expert_bias,
        )

        return router_logits, topk_weights, topk_ids

class Hy3MoE(nn.Module):
    """Mixture-of-experts block for Hy3."""

    def __init__(
        self,
        config,
        layer_idx: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()

        quantization_config = getattr(
            config,
            'quantization_config',
            None,
        )

        self.hidden_size = config.hidden_size
        self.enable_moe_fp32_combine = (
            config.enable_moe_fp32_combine
        )

        self.router = Hy3Router(
            config,
            device=device,
        )

        # The checkpoint parameter name is mlp.expert_bias.
        self.expert_bias = nn.Parameter(
            torch.empty(
                config.num_experts,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )

        self.experts = build_fused_moe(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            top_k=config.num_experts_per_tok,
            renormalize=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            all_reduce=False,
            layer_idx=layer_idx,
            prefix=add_prefix('experts', prefix),
        )

        shared_intermediate_size = (
            config.moe_intermediate_size
            * config.num_shared_experts
        )

        self.shared_mlp = Hy3MLP(
            config,
            intermediate_size=shared_intermediate_size,
            dtype=dtype,
            device=device,
            is_tp=True,
            all_reduce=False,
            prefix=add_prefix('shared_mlp', prefix),
        )

        world_size, _ = get_tp_world_rank()
        self._all_reduce = world_size > 1

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(
            -1,
            self.hidden_size,
        )

        _, topk_weights, topk_ids = self.router(
            hidden_states,
            self.expert_bias,
        )

        routed_output = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        shared_output = self.shared_mlp(hidden_states)

        if self.enable_moe_fp32_combine:
            output = (
                routed_output.float()
                + shared_output.float()
            ).to(hidden_states.dtype)
        else:
            output = routed_output + shared_output

        output = output.reshape(original_shape)

        if self._all_reduce:
            dist.all_reduce(output)

        return output

class Hy3DecoderLayer(nn.Module):
    """Decoder layer for Hy3."""

    def __init__(
        self,
        config,
        layer_idx: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()

        self.layer_idx = layer_idx
        quantization_config = getattr(
            config,
            'quantization_config',
            None,
        )

        self.self_attn = Hy3Attention(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('self_attn', prefix),
        )

        # Hy3 的第 0 层是 Dense，后续层是 MoE。
        if layer_idx < config.first_k_dense_replace:
            self.mlp = Hy3MLP(
                config,
                intermediate_size=config.intermediate_size,
                dtype=dtype,
                device=device,
                prefix=add_prefix('mlp', prefix),
            )
        else:
            self.mlp = Hy3MoE(
                config,
                layer_idx=layer_idx,
                dtype=dtype,
                device=device,
                prefix=add_prefix('mlp', prefix),
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix(
                'input_layernorm',
                prefix,
            ),
        )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix(
                'post_attention_layernorm',
                prefix,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[
            torch.FloatTensor,
            torch.FloatTensor,
        ],
        past_key_value: list[torch.FloatTensor] | None,
        residual: torch.Tensor | None = None,
        attn_metadata: Any = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(
                hidden_states
            )
        else:
            hidden_states, residual = (
                self.input_layernorm(
                    hidden_states,
                    residual,
                )
            )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = (
            self.post_attention_layernorm(
                hidden_states,
                residual,
            )
        )

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

class Hy3Model(nn.Module):
    """Transformer model for Hy3."""

    def __init__(
        self,
        config,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()

        self.padding_idx = getattr(
            config,
            'pad_token_id',
            None,
        )
        self.vocab_size = config.vocab_size

        self.embed_tokens = build_embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=dtype,
            device=device,
        )

        self.layers = nn.ModuleList([
            Hy3DecoderLayer(
                config,
                layer_idx=layer_idx,
                dtype=dtype,
                device=device,
                prefix=add_prefix(
                    f'layers.{layer_idx}',
                    prefix,
                ),
            )
            for layer_idx in range(
                config.num_hidden_layers
            )
        ])

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('norm', prefix),
        )

        self.rotary_emb = (
            build_rotary_embedding_from_config(
                config,
                device=device,
            )
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[
            torch.FloatTensor
        ] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(
                input_ids
            )

        hidden_states = inputs_embeds

        cos, sin = self.rotary_emb(
            hidden_states,
            position_ids,
        )

        # LMDeploy RoPE 的第 0 维是 batch 维。
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        residual = None

        for layer_idx, decoder_layer in enumerate(
            self.layers
        ):
            past_key_value = past_key_values[
                layer_idx
            ]

            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        hidden_states, _ = self.norm(
            hidden_states,
            residual,
        )

        return hidden_states

    def get_input_embeddings(self):
        """Return token embeddings."""
        return self.embed_tokens

class HYV3ForCausalLM(
    nn.Module,
    DeployModelMixinV1,
    CudaGraphMixin,
):
    """Causal language model for Hy3."""

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

    def __init__(
        self,
        config,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()

        self.config = config
        self.ctx_mgr = ctx_mgr

        self.model = Hy3Model(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('model', prefix),
        )

        self.lm_head = self.build_lm_head(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Run the transformer and return hidden states."""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def get_input_embeddings(self):
        """Return token embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare inputs for engine generation."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'attn_metadata': attn_metadata,
            'inputs_embeds': inputs_embeds,
        }

    def _load_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        expert_params_mapping: list,
    ):
        """Load one expert weight into FusedMoE."""

        for (
            target_pattern,
            source_pattern,
            expert_id,
            shard_id,
        ) in expert_params_mapping:
            if source_pattern not in name:
                continue

            target_name = name.replace(
                source_pattern,
                target_pattern,
            )
            parameter = params_dict[target_name]

            load_weight(
                parameter,
                loaded_weight,
                expert_id=expert_id,
                shard_id=shard_id,
            )
            return

        raise RuntimeError(
            f'Unsupported Hy3 expert weight: {name}'
        )

    def load_weights(
        self,
        weights: Iterable[
            tuple[str, torch.Tensor]
        ],
    ):
        """Load Hy3 weights."""

        stacked_params_mapping = [
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        num_hidden_layers = (
            self.config.num_hidden_layers
        )
        num_mtp_layers = getattr(
            self.config,
            'num_nextn_predict_layers',
            0,
        )

        mtp_layer_names = [
            f'.layers.{num_hidden_layers + index}.'
            for index in range(num_mtp_layers)
        ]

        expert_params_mapping = []

        for expert_id in range(
            self.config.num_experts
        ):
            expert_params_mapping.extend([
                (
                    '.experts.gate_up',
                    (
                        f'.experts.{expert_id}'
                        '.gate_proj'
                    ),
                    expert_id,
                    'gate',
                ),
                (
                    '.experts.gate_up',
                    (
                        f'.experts.{expert_id}'
                        '.up_proj'
                    ),
                    expert_id,
                    'up',
                ),
                (
                    '.experts.down',
                    (
                        f'.experts.{expert_id}'
                        '.down_proj'
                    ),
                    expert_id,
                    'down',
                ),
            ])

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # MTP weights are not supported yet.
            if any(
                layer_name in name
                for layer_name in mtp_layer_names
            ):
                continue

            if 'rotary_emb.inv_freq' in name:
                continue

            if (
                'rotary_emb.cos_cached' in name
                or 'rotary_emb.sin_cached' in name
            ):
                continue

            if (
                self.config.tie_word_embeddings
                and name == 'lm_head.weight'
            ):
                continue

            # Load individual expert weights into fused tensors.
            if '.experts.' in name:
                self._load_expert_weight(
                    name,
                    loaded_weight,
                    params_dict,
                    expert_params_mapping,
                )
                continue

            for (
                param_name,
                weight_name,
                shard_id,
            ) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                target_name = name.replace(
                    weight_name,
                    param_name,
                )
                parameter = params_dict[target_name]

                load_weight(
                    parameter,
                    loaded_weight,
                    shard_id=shard_id,
                )
                break
            else:
                parameter = params_dict[name]
                load_weight(
                    parameter,
                    loaded_weight,
                )
