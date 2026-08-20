# Copyright (c) OpenMMLab. All rights reserved.
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.backends.mimo_swa import (
    MiMoSWAAttentionMetadata,
    mimo_swa_state_attention,
)
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul, build_rotary_embedding
from lmdeploy.pytorch.nn.eplb import EPLBManager
from lmdeploy.pytorch.nn.linear import build_down_linear, build_gateup_linear, build_o_proj, build_qkv_proj
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader, load_weight

from .deepseek_v2 import DeepseekV2MoE
from .patch import add_prefix, get_build_model_context
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixinV1, build_embedding


def _get_norm_eps(config: Any) -> float:
    """Read the native or remote-code MiMo RMSNorm epsilon field."""
    norm_eps = getattr(config, "rms_norm_eps", None)
    if norm_eps is None:
        norm_eps = config.layernorm_epsilon
    return norm_eps


def _all_reduce_mimo(output: torch.Tensor, group=None) -> torch.Tensor:
    """Reduce a MiMo tensor over its tensor-parallel group."""
    dist.all_reduce(output, group=group)
    return output


def _reduce_tp_output(linear: nn.Module, output: torch.Tensor) -> torch.Tensor:
    """Reduce one MiMo row-parallel output."""
    if linear.tp > 1:
        output = _all_reduce_mimo(output, group=linear.tp_group)
    return output


def _dequantize_blocked_fp8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize 2D FP8 using the checkpoint's serialized scale grid.

    MiMo Full-K is (768, 4096) with an (8, 32) scale grid, so its effective
    output tile is 96 rather than the config-level 128.  Deriving the tile from
    both tensors matches Transformers' official FP8 dequantization reference.
    """
    rows, columns = weight.shape
    scale_rows, scale_columns = scale.shape
    if rows % scale_rows != 0 or columns % scale_columns != 0:
        raise ValueError(
            f"Invalid blocked-FP8 weight/scale shapes: weight={tuple(weight.shape)}, scale={tuple(scale.shape)}."
        )
    weight = weight.reshape(scale_rows, rows // scale_rows, scale_columns, columns // scale_columns)
    weight = weight.float() * scale.reshape(scale_rows, 1, scale_columns, 1).float()
    return weight.to(dtype).reshape(rows, columns)


def _load_attention_sink(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Load this rank's Q-head slice of an attention sink vector."""
    world_size, rank = get_tp_world_rank("attn")
    if loaded_weight.size(0) % world_size != 0:
        raise ValueError(f"Cannot shard {loaded_weight.size(0)} MiMo sink heads over TP={world_size}.")
    loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
    default_weight_loader(param, loaded_weight)


@dataclass
class MiMoV2Caches:
    """Named P1 Full block caches and SWA state rings."""

    block_caches: Any
    state_caches: Any

    def block_cache(self, cache_name: str, layer_idx: int) -> torch.Tensor:
        """Resolve a compact named-cache row by global layer index."""
        if hasattr(self.block_caches, "layer"):
            return self.block_caches.layer(cache_name, layer_idx)
        return self.block_caches[cache_name][layer_idx]

    def state_cache(self, cache_name: str, layer_idx: int) -> torch.Tensor:
        """Resolve a compact named-state row by global layer index."""
        if hasattr(self.state_caches, "layer"):
            return self.state_caches.layer(cache_name, layer_idx)
        return self.state_caches[cache_name][layer_idx]


class MiMoV2Attention(nn.Module):
    """Common MiMo-V2 QKV projection, partial RoPE and output projection."""

    def __init__(
        self,
        config: Any,
        is_swa: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = "",
    ):
        super().__init__()
        quantization_config = getattr(config, "quantization_config", None)
        if is_swa:
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            num_heads = config.swa_num_attention_heads
            num_kv_heads = config.swa_num_key_value_heads
            num_replicate_kv_heads = getattr(config, "swa_num_replicate_key_value_heads", 1)
            sliding_window = getattr(config, "sliding_window_size", getattr(config, "sliding_window", -1))
            has_sink = getattr(config, "add_swa_attention_sink_bias", False)
        else:
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            num_replicate_kv_heads = getattr(config, "num_replicate_key_value_heads", 1)
            sliding_window = None
            has_sink = getattr(config, "add_full_attention_sink_bias", False)

        self.v_scale = getattr(config, "attention_value_scale", None)
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim or self.rotary_dim % 2 != 0:
            raise ValueError(f"Invalid MiMo-V2 rotary dimension: {self.rotary_dim}.")

        self.qkv_proj = build_qkv_proj(
            config.hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.v_head_dim,
            bias=config.attention_bias,
            # P0 keeps QKV in BF16. A 192-wide local K head is not aligned to
            # LMDeploy's 128-row packed-FP8 scale partitions at TP=4/8; the
            # loader dequantizes the official FP8 Q/K/V tensors before the
            # head-aware BF16 QKV loader shards or replicates them.
            quant_config=None,
            dtype=dtype,
            device=device,
            num_replicate_kv_heads=num_replicate_kv_heads,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.attn_fwd = Attention(
            num_heads,
            self.head_dim,
            num_kv_heads=num_kv_heads,
            v_head_size=self.v_head_dim,
            sliding_window=sliding_window,
            learnable_sink=has_sink,
            # Full Attention can use FA3 when its wheel contains the asymmetric
            # Q/K=192, V=128 instantiation. SWA remains on the MiMo-specific
            # Triton/ring path because it also implements attention sinks.
            enable_fa3=not is_swa,
        )
        self.attention_sink_bias = None
        if has_sink:
            self.attention_sink_bias = nn.Parameter(
                torch.empty(self.qkv_proj.num_q_heads, dtype=dtype, device=device),
                requires_grad=False,
            )
            self.attention_sink_bias.weight_loader = _load_attention_sink
        self.o_proj = build_o_proj(
            num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            # Keep reduction outside the linear so MiMo uses the same explicit
            # tensor-parallel group for attention, dense FFN and MoE outputs.
            all_reduce=False,
            prefix=add_prefix("o_proj", prefix),
        )

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Apply o_proj and the standard BF16 tensor-parallel reduction."""
        output = self.o_proj(attn_output)
        return _reduce_tp_output(self.o_proj, output)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project QKV, apply partial RoPE and scale V before caching."""
        qkv_states = self.qkv_proj(hidden_states).flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        # MiMo rotates only the leading rotary_dim features. Passing the views
        # explicitly also keeps the default backend valid; broadcasting a
        # rotary_dim-wide cos/sin table over the complete 192 features would
        # be incorrect.
        query_rope = query_states[..., : self.rotary_dim]
        key_rope = key_states[..., : self.rotary_dim]
        cos, sin = rotary_pos_emb
        self.apply_rotary_pos_emb(query_rope, key_rope, cos, sin, inplace=True)
        if self.v_scale is not None:
            # Scale before either block/ring cache writes so current attention,
            # decode and prefix restore share one V representation.
            value_states = value_states * self.v_scale
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Run the Full-Attention paged KV path."""
        query_states, key_states, value_states = self._project_qkv(hidden_states, rotary_pos_emb)

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            s_aux=self.attention_sink_bias,
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        return self._project_output(attn_output)


class MiMoV2FullAttention(MiMoV2Attention):
    """MiMo-V2 full attention backed by the named paged KV cache."""

    def __init__(self, config: Any, **kwargs):
        super().__init__(config, is_swa=False, **kwargs)


class MiMoV2SWAAttention(MiMoV2Attention):
    """P1 SWA attention backed by a sequence-scoped BF16 state ring."""

    def __init__(self, config: Any, **kwargs):
        super().__init__(config, is_swa=True, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
        state_cache: tuple[torch.Tensor, torch.Tensor],
        swa_metadata: MiMoSWAAttentionMetadata,
    ) -> torch.Tensor:
        """Run varlen attention over chronological ring history + current KV."""
        query_states, key_states, value_states = self._project_qkv(hidden_states, rotary_pos_emb)
        attn_output = mimo_swa_state_attention(
            self.attn_fwd,
            query_states,
            key_states,
            value_states,
            state_cache[0],
            state_cache[1],
            swa_metadata,
            sink=self.attention_sink_bias,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        return self._project_output(attn_output)


class MiMoV2MLP(nn.Module):
    """Dense SwiGLU block used by MiMo-V2-Flash."""

    def __init__(
        self,
        config: Any,
        intermediate_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        quantization_config = getattr(config, "quantization_config", None)

        # The official checkpoint stores gate_proj and up_proj separately.
        # LMDeploy packs them into one column-parallel projection; SiluAndMul
        # consumes the packed [gate, up] output without changing the math.
        self.gate_up_proj = build_gateup_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.act_fn = SiluAndMul(inplace=True)
        self.down_proj = build_down_linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            # Keep the framework reduction enabled.  In DP+TP mode this is a
            # reduce-scatter that restores each DP rank's local token layout;
            # replacing it with a plain all-reduce leaves gathered tokens in
            # the output and breaks the residual shape in the next layer.
            all_reduce=True,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply down_proj(silu(gate_proj(x)) * up_proj(x))."""
        gate_up = self.gate_up_proj(hidden_states)
        return self.down_proj(self.act_fn(gate_up))


class MiMoV2MoE(DeepseekV2MoE):
    """MiMo-V2 MoE using LMDeploy's fused DeepSeek-style execution path."""

    def forward(self, hidden_states: torch.Tensor, all_routed_experts: torch.Tensor = None):
        """Run fused experts and reduce their BF16 output over MiMo TP."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        routed_experts = None
        if all_routed_experts is not None:
            routed_experts = all_routed_experts[:, self.layer_idx, :]
        topk_weights, topk_ids = self.gate(hidden_states, routed_experts=routed_experts)

        out_states = self.experts(hidden_states, topk_weights, topk_ids)
        if self.shared_experts is not None:
            out_states += self.shared_experts(hidden_states)
        out_states = out_states.reshape(batch_size, sequence_length, -1)

        if self._all_reduce:
            out_states = _all_reduce_mimo(out_states, group=self.experts.tp_group)
        return out_states


class MiMoV2DecoderLayer(nn.Module):
    """MiMo-V2 decoder layer with per-layer attention and FFN selection."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = config.hybrid_layer_pattern[layer_idx] == 1
        self.is_moe = getattr(config, "n_routed_experts", None) is not None and bool(config.moe_layer_freq[layer_idx])
        quantization_config = getattr(config, "quantization_config", None)

        attention_cls = MiMoV2SWAAttention if self.is_swa else MiMoV2FullAttention
        self.self_attn = attention_cls(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix("self_attn", prefix),
        )
        if self.is_moe:
            self.mlp = MiMoV2MoE(config, layer_idx, dtype=dtype, device=device)
        else:
            self.mlp = MiMoV2MLP(
                config,
                dtype=dtype,
                device=device,
                prefix=add_prefix("mlp", prefix),
            )

        norm_eps = _get_norm_eps(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix("input_layernorm", prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix("post_attention_layernorm", prefix),
        )

        cache_prefix = "mimo_swa_ring" if self.is_swa else "mimo_full"
        self.k_cache_name = f"{cache_prefix}_k"
        self.v_cache_name = f"{cache_prefix}_v"

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
        caches: MiMoV2Caches,
        residual: torch.Tensor | None = None,
        attn_metadata: Any = None,
        swa_metadata: MiMoSWAAttentionMetadata | None = None,
        all_routed_experts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run pre-norm attention and FFN while carrying fused residual."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.is_swa:
            if swa_metadata is None:
                raise RuntimeError("MiMo SWA layer requires state-ring attention metadata.")
            state_cache = (
                caches.state_cache(self.k_cache_name, self.layer_idx),
                caches.state_cache(self.v_cache_name, self.layer_idx),
            )
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                state_cache=state_cache,
                swa_metadata=swa_metadata,
            )
        else:
            past_key_value = (
                caches.block_cache(self.k_cache_name, self.layer_idx),
                caches.block_cache(self.v_cache_name, self.layer_idx),
            )
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
            )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.is_moe:
            hidden_states = self.mlp(hidden_states, all_routed_experts=all_routed_experts)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiMoV2Model(nn.Module):
    """MiMo-V2 transformer body using heterogeneous named block caches."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        if len(config.hybrid_layer_pattern) != config.num_hidden_layers:
            raise ValueError("MiMo-V2 hybrid_layer_pattern must contain one entry per decoder layer.")
        if len(config.moe_layer_freq) != config.num_hidden_layers:
            raise ValueError("MiMo-V2 moe_layer_freq must contain one entry per decoder layer.")

        self.embed_tokens = build_embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=dtype,
            device=device,
            is_tp=True,
        )

        if get_dist_manager().current_context().dist_config.enable_eplb:
            ep_size, _ = get_ep_world_rank()
            EPLBManager.init_global_eplb_metadata(
                ep_size=ep_size,
                num_routed_experts=config.n_routed_experts,
                num_hidden_layers=config.num_hidden_layers,
            )

        self.layers = nn.ModuleList(
            [
                MiMoV2DecoderLayer(
                    config,
                    layer_idx,
                    dtype=dtype,
                    device=device,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        norm_eps = _get_norm_eps(config)
        self.norm = RMSNorm(
            config.hidden_size,
            norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix("norm", prefix),
        )

        rotary_kwargs = dict(
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=config.partial_rotary_factor,
            device=device,
        )
        self.rotary_emb = build_rotary_embedding(
            dim=config.head_dim,
            base=config.rope_theta,
            **rotary_kwargs,
        )
        self.swa_rotary_emb = build_rotary_embedding(
            dim=config.swa_head_dim,
            base=config.swa_rope_theta,
            **rotary_kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None,
        position_ids: torch.LongTensor,
        caches: MiMoV2Caches,
        attn_metadata: Any = None,
        swa_metadata: MiMoSWAAttentionMetadata | None = None,
        inputs_embeds: torch.Tensor | None = None,
        all_routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all decoder layers with layer-appropriate RoPE and caches."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        full_cos, full_sin = self.rotary_emb(hidden_states, position_ids)
        swa_cos, swa_sin = self.swa_rotary_emb(hidden_states, position_ids)
        full_rotary_pos_emb = (full_cos[0], full_sin[0])
        swa_rotary_pos_emb = (swa_cos[0], swa_sin[0])

        residual = None
        for decoder_layer in self.layers:
            rotary_pos_emb = swa_rotary_pos_emb if decoder_layer.is_swa else full_rotary_pos_emb
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                caches=caches,
                residual=residual,
                attn_metadata=attn_metadata,
                swa_metadata=swa_metadata,
                all_routed_experts=all_routed_experts,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_input_embeddings(self):
        """Return the token embedding module."""
        return self.embed_tokens


class MiMoV2FlashForCausalLM(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    """MiMo-V2-Flash causal LM with named caches and FP8 checkpoint loading."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        """Return whether the target-only invocation supports CUDA Graph."""
        return super().support_cuda_graph(
            input_ids,
            position_ids,
            past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def __init__(
        self,
        config: Any,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.model = MiMoV2Model(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = self.build_lm_head(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.enable_return_routed_experts = get_build_model_context().enable_return_routed_experts
        self._load_buffers: dict[str, dict[str, torch.Tensor]] = {}
        self._first_swa_layer = next(
            (layer_id for layer_id, layer_type in enumerate(config.hybrid_layer_pattern) if layer_type == 1),
            None,
        )
        if self._first_swa_layer is None:
            raise ValueError("MiMo-V2 requires at least one SWA layer.")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        state_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Resolve named caches once, then execute the transformer body."""
        context = self.ctx_mgr.current_context()
        if context.block_caches is None:
            raise RuntimeError("MiMo-V2 requires named block caches in the current StepContext.")
        if context.named_state_caches is None:
            raise RuntimeError("MiMo-V2 requires named SWA state caches in the current StepContext.")
        if state_ids is None:
            raise RuntimeError("MiMo-V2 requires state_ids to provide stable SWA ring slots.")
        caches = MiMoV2Caches(
            block_caches=context.block_caches,
            state_caches=context.named_state_caches,
        )
        swa_metadata = None
        first_swa_ring = context.named_state_caches.layer("mimo_swa_ring_k", self._first_swa_layer)
        swa_metadata = MiMoSWAAttentionMetadata.from_step_context(
            attn_metadata,
            context,
            state_ids,
            num_state_slots=first_swa_ring.size(0),
            window_size=first_swa_ring.size(1),
        )

        all_routed_experts = None
        if self.enable_return_routed_experts:
            num_tokens = inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1)
            all_routed_experts = position_ids.new_full(
                (num_tokens, self.config.num_hidden_layers, self.config.num_experts_per_tok),
                torch.iinfo(torch.uint16).max,
                dtype=torch.uint16,
            )

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            caches=caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            swa_metadata=swa_metadata,
            all_routed_experts=all_routed_experts,
        )
        if all_routed_experts is None:
            return hidden_states
        return dict(hidden_states=hidden_states, all_routed_experts=all_routed_experts)

    def get_input_embeddings(self):
        """Return the model token embedding module."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Prepare the standard engine inputs; named caches live on context."""
        return dict(
            input_ids=context.input_ids,
            position_ids=context.position_ids,
            past_key_values=past_key_values,
            attn_metadata=context.attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
        )

    def _load_qkv_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        shard_id: str,
    ):
        """Pair FP8 Q/K/V weights with scales, dequantize, then TP-shard."""
        if name.endswith(".weight_scale_inv"):
            source_prefix = name.removesuffix(".weight_scale_inv")
            tensor_kind = "scale"
        elif name.endswith(".weight"):
            source_prefix = name.removesuffix(".weight")
            tensor_kind = "weight"
        else:
            raise KeyError(f"Unexpected MiMo QKV tensor name: {name}")

        target_prefix = re.sub(r"\.(q|k|v)_proj$", ".qkv_proj", source_prefix)
        target_param = params_dict[f"{target_prefix}.weight"]
        if tensor_kind == "weight" and loaded_weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            load_weight(target_param, loaded_weight, shard_id=shard_id)
            return

        buffer = self._load_buffers.setdefault(source_prefix, {})
        buffer[tensor_kind] = loaded_weight
        if "weight" not in buffer or "scale" not in buffer:
            return

        weight = _dequantize_blocked_fp8(buffer["weight"], buffer["scale"], target_param.dtype)
        load_weight(target_param, weight, shard_id=shard_id)
        self._load_buffers.pop(source_prefix)

    @staticmethod
    def _load_expert_weight(name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter]):
        """Map one checkpoint expert tensor to the fused MoE parameter."""
        match = re.match(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$",
            name,
        )
        if match is None:
            raise KeyError(f"Unexpected MiMo expert tensor name: {name}")
        experts_prefix, expert_id, projection, suffix = match.groups()
        if projection == "gate_proj":
            target_projection, shard_id = "gate_up", "gate"
        elif projection == "up_proj":
            target_projection, shard_id = "gate_up", "up"
        else:
            target_projection, shard_id = "down", "down"
        param = params_dict[f"{experts_prefix}.{target_projection}.{suffix}"]
        load_weight(param, loaded_weight, expert_id=int(expert_id), shard_id=shard_id)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load MiMo-V2-Flash target-model weights."""
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith("model.mtp."):
                # The target model does not consume the optional draft block.
                continue
            if "rotary_emb.inv_freq" in name or "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            if ".experts." in name:
                self._load_expert_weight(name, loaded_weight, params_dict)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if weight_name in (".q_proj", ".k_proj", ".v_proj"):
                    self._load_qkv_weight(name, loaded_weight, params_dict, shard_id)
                else:
                    target_name = name.replace(weight_name, param_name)
                    load_weight(params_dict[target_name], loaded_weight, shard_id=shard_id)
                break
            else:
                try:
                    param = params_dict[name]
                except KeyError as error:
                    raise KeyError(f"Unexpected MiMo-V2-Flash weight: {name}") from error
                load_weight(param, loaded_weight)

    def update_weights(self):
        """Reject incomplete FP8 pairs before finalizing tied weights."""
        if self._load_buffers:
            pending = ", ".join(sorted(self._load_buffers))
            raise RuntimeError(f"Incomplete MiMo FP8 QKV weight/scale pairs: {pending}")
        super().update_weights()
