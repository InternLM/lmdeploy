# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi-K2 language runtime built on the upstream DeepSeek MLA primitives."""

import re
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank
from lmdeploy.pytorch.model_inputs import (
    StepContext,
    StepContextManager,
    get_step_ctx_manager,
)
from lmdeploy.pytorch.nn import (
    ApplyRotaryEmb,
    Attention,
    ParallelEmbedding,
    RMSNorm,
    RopeType,
    SiluAndMul,
    build_rotary_embedding,
    build_rotary_params,
)
from lmdeploy.pytorch.nn.eplb import EPLBManager
from lmdeploy.pytorch.nn.linear import (
    build_colwise_linear,
    build_down_linear,
    build_gateup_linear,
    build_merged_colwise_linear,
    build_o_proj,
    build_rowwise_linear,
)
from lmdeploy.pytorch.nn.moe import MoeType, build_fused_moe
from lmdeploy.pytorch.nn.rotary_embedding import (
    get_rope_parameters,
    get_rope_theta,
)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .deepseek_v2 import (
    DeepseekV2BMM as KimiK2BMM,
)
from .deepseek_v2 import (
    MoEGate,
    execute_batch,
    merge_output,
    split_input,
    yarn_get_mscale,
)
from .patch import add_prefix, get_build_model_context
from .utils.cudagraph import CudaGraphMixin

_COMPRESSED_TENSORS_EXPERT_WEIGHT_RE = re.compile(
    r'^(?P<prefix>(?:.+\.)?experts)\.(?P<expert_id>[0-9]+)\.'
    r'(?P<projection>gate_proj|up_proj|down_proj)\.(?P<suffix>[^.]+)$')
_COMPRESSED_TENSORS_EXPERT_SUFFIXES = frozenset({
    'weight_packed',
    'weight_scale',
    'weight_shape',
})
_COMPRESSED_TENSORS_EXPERT_PROJECTIONS = {
    'gate_proj': ('gate_up', 'gate'),
    'up_proj': ('gate_up', 'up'),
    'down_proj': ('down', 'down'),
}


def _use_kimi_fused_qkv_a_proj(
    config: Any,
    dtype: torch.dtype | None,
    prefix: str,
) -> bool:
    """Check the Kimi BF16/FP16 replicated MLA A-projection contract."""
    if not getattr(config, 'fuse_qkv_a_proj', False):
        return False
    resolved_dtype = dtype if dtype is not None else getattr(config, 'dtype', None)
    dtype_name = str(resolved_dtype).removeprefix('torch.')
    if (
        getattr(config, 'model_type', None) != 'kimi_k2'
        or getattr(config, 'q_lora_rank', None) is None
        or getattr(config, 'hidden_size', None) != 7168
        or config.q_lora_rank != 1536
        or getattr(config, 'kv_lora_rank', None) != 512
        or getattr(config, 'qk_rope_head_dim', None) != 64
        or dtype_name not in {'bfloat16', 'float16'}
    ):
        return False

    quant_config = get_build_model_context().quant_config
    source_prefixes = (
        add_prefix('q_a_proj', prefix),
        add_prefix('kv_a_proj_with_mqa', prefix),
    )
    return all(
        quant_config.get_quant_method(source_prefix, module_kind='linear') is None
        for source_prefix in source_prefixes
    )


class KimiK2Attention(nn.Module):
    """Deepseekv2 attention."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        num_key_value_heads = getattr(config, 'num_key_value_heads', 1)
        use_flash_mla = getattr(config, 'use_flash_mla', False)
        self.fuse_qkv_a_proj = _use_kimi_fused_qkv_a_proj(config, dtype, prefix)

        if self.q_lora_rank is None:
            self.q_proj = build_colwise_linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                is_tp=True,
                quant_config=quantization_config,
                dp_disable_tp=True,
                prefix=add_prefix('q_proj', prefix),
            )
        else:
            if self.fuse_qkv_a_proj:
                self.fused_qkv_a_proj_with_mqa = build_merged_colwise_linear(
                    self.hidden_size,
                    [config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim],
                    bias=config.attention_bias,
                    dtype=dtype,
                    device=device,
                    is_tp=False,
                    out_names=['q', 'kv'],
                    quant_config=quantization_config,
                    check_dist=False,
                    layer_type='attn',
                    prefix=add_prefix('fused_qkv_a_proj_with_mqa', prefix),
                )
            else:
                self.q_a_proj = build_colwise_linear(
                    self.hidden_size,
                    config.q_lora_rank,
                    bias=config.attention_bias,
                    dtype=dtype,
                    device=device,
                    is_tp=False,
                    quant_config=quantization_config,
                    prefix=add_prefix('q_a_proj', prefix),
                )
            self.q_a_layernorm = RMSNorm(config.q_lora_rank,
                                         1e-6,
                                         quant_config=quantization_config,
                                         dtype=dtype,
                                         device=device,
                                         prefix=add_prefix('q_a_layernorm', prefix))
            self.q_b_proj = build_colwise_linear(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                is_tp=True,
                quant_config=quantization_config,
                dp_disable_tp=True,
                prefix=add_prefix('q_b_proj', prefix),
            )

        if not self.fuse_qkv_a_proj:
            self.kv_a_proj_with_mqa = build_colwise_linear(
                self.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                bias=config.attention_bias,
                dtype=dtype,
                device=device,
                is_tp=False,
                quant_config=quantization_config,
                prefix=add_prefix('kv_a_proj_with_mqa', prefix),
            )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank,
                                      1e-6,
                                      quant_config=quantization_config,
                                      dtype=dtype,
                                      device=device,
                                      prefix=add_prefix('kv_a_layernorm', prefix))
        self.kc = KimiK2BMM(self.num_heads,
                                config.qk_nope_head_dim,
                                config.kv_lora_rank,
                                dtype=dtype,
                                device=device)

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        self.softmax_scale = self.q_head_dim**(-0.5)

        rope_scaling = get_rope_parameters(config)
        if rope_scaling is not None:
            mscale_all_dim = rope_scaling.get('mscale_all_dim', 0)
            scaling_factor = rope_scaling.get('factor', 1.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_fwd = Attention(self.num_heads,
                                  config.kv_lora_rank + self.qk_rope_head_dim,
                                  scale=self.softmax_scale,
                                  num_kv_heads=num_key_value_heads,
                                  v_head_size=config.kv_lora_rank,
                                  num_replicate_kv_heads=num_replicate_kv_heads,
                                  use_flash_mla=use_flash_mla)

        self.vc = KimiK2BMM(self.num_heads, config.kv_lora_rank, self.v_head_dim, dtype=dtype, device=device)
        self.o_proj = build_o_proj(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
            prefix=add_prefix('o_proj', prefix),
        )

    def _q_proj(self,
                hidden_states,
                num_heads: int,
                nope_size: int,
                pe_size: int,
                q_a_states: torch.Tensor | None = None):
        """Q proj."""
        q_len = hidden_states.size(1)

        query_states = hidden_states.new_empty(q_len, num_heads, nope_size + pe_size)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            if q_a_states is None:
                q_a_states = self.q_a_proj(hidden_states)
            q = self.q_b_proj(self.q_a_layernorm(q_a_states))
        q = q.view(q_len, num_heads, self.q_head_dim)
        # q_pe: (q_len, num_heads, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope: (q_len, num_heads, kv_lora_rank)
        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe

    def _kv_proj(self, hidden_states, nope_size: int, kv_a_states: torch.Tensor | None = None):
        """Kv proj."""
        # (q_len, 1, nope_size + pe_size)
        if kv_a_states is None:
            key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
        else:
            key_states = kv_a_states[0, :, None]
        # (q_len, 1, pe_size)
        k_pe = key_states[..., nope_size:]
        # kv_a_layernorm
        value_states = key_states[..., :nope_size]
        value_states = self.kv_a_layernorm(value_states)
        key_states[..., :nope_size] = value_states
        return key_states, value_states, k_pe

    def _qkv_proj(self, hidden_states: torch.Tensor, num_heads: int):
        """Qkv proj."""
        nope_size = self.kv_lora_rank
        pe_size = self.qk_rope_head_dim
        q_a_states = None
        kv_a_states = None
        if getattr(self, 'fuse_qkv_a_proj', False):
            fused_states = self.fused_qkv_a_proj_with_mqa(hidden_states)
            q_a_states, kv_a_states = fused_states.split([self.q_lora_rank, nope_size + pe_size], dim=-1)
        query_states, q_pe = self._q_proj(
            hidden_states,
            num_heads,
            nope_size,
            pe_size,
            q_a_states=q_a_states,
        )
        key_states, value_states, k_pe = self._kv_proj(
            hidden_states,
            nope_size,
            kv_a_states=kv_a_states,
        )

        return query_states, key_states, value_states, q_pe, k_pe

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: tuple[torch.Tensor] | None = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        dist_config = get_dist_manager().current_config()
        if dist_config.dp > 1:
            num_heads = self.num_heads
        else:
            world_size = dist_config.world_size
            num_heads = self.num_heads // world_size
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(1)

        # qkv_proj
        query_states, key_states, value_states, q_pe, k_pe = self._qkv_proj(
            hidden_states, num_heads=num_heads)

        cos, sin = rotary_pos_emb
        q_pe, k_pe = self.apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            inplace=False,
        )
        query_states[..., nope_size:] = q_pe
        key_states[..., nope_size:] = k_pe

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[0][..., :nope_size],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)

        self.vc(attn_output, attn_bmm_out)
        attn_output = attn_bmm_out.flatten(-2, -1)[None]
        attn_output = self.o_proj(attn_output)

        return attn_output


class KimiK2MoE(nn.Module):
    """Deepseek v2 MoE."""

    def __init__(
        self,
        config: Any,
        layer_idx,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.renormalize = self.top_k > 1 and self.norm_topk_prob
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        dp = dist_config.dp
        world_size = dist_config.world_size
        self._shared_expert_tp_group = getattr(
            getattr(dist_ctx, 'mlp_tp_group', None), 'gpu_group', None)
        moe_all_reduce = dp > 1 and dist_config.tp > 1
        if get_dist_manager().current_context().dist_config.enable_eplb:
            eplb_dispatch_info = EPLBManager.get_dispatch_info(
                ep_rank=dist_ctx.ep_rank,
                layer_idx=layer_idx,
            )
            self.num_experts = EPLBManager.num_physical_experts()
            self.gate = MoEGate(config, dtype=dtype, device=device, info=eplb_dispatch_info)
        else:
            self.gate = MoEGate(config, dtype=dtype, device=device, info=None)
        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            renormalize=False,
            dtype=dtype,
            device=device,
            all_reduce=moe_all_reduce,
            quant_config=quantization_config,
            layer_idx=layer_idx,
            prefix=add_prefix('experts', prefix),
        )
        self.shared_experts = None
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size * config.n_shared_experts)
            self.shared_experts = KimiK2MLP(
                config=config,
                intermediate_size=intermediate_size,
                dtype=dtype,
                device=device,
                is_shared_expert=True,
                prefix=add_prefix('shared_experts', prefix),
            )

        if dp == 1 and world_size > 1:
            self._all_reduce = True
        else:
            self._all_reduce = False

    def forward(self, hidden_states: torch.Tensor, all_routed_experts: torch.Tensor = None):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        routed_experts = None
        if all_routed_experts is not None:
            routed_experts = all_routed_experts[:, self.layer_idx, :]
        topk_weights, topk_ids = self.gate(hidden_states, routed_experts=routed_experts)

        out_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        shared_states = None
        if self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
            shared_states = shared_states.reshape(batch_size, sequence_length,
                                                  -1)
        out_states = out_states.reshape(batch_size, sequence_length, -1)
        out_states = self._combine_expert_outputs(out_states, shared_states)

        return out_states

    def _combine_expert_outputs(
        self,
        out_states: torch.Tensor,
        shared_states: torch.Tensor | None,
    ):
        """Combine routed and shared outputs under their distribution
        contract."""
        output_dtype = out_states.dtype
        tp_reduce_dtype = getattr(self.experts, 'tp_reduce_dtype', None)
        use_promoted_tp_reduce = self._all_reduce and tp_reduce_dtype is not None

        # DeepEP dispatch/combine returns a complete routed result to each
        # source rank, while the shared expert remains TP-sharded for DP=1.
        # Reducing their sum would therefore multiply the routed result by EP.
        if getattr(self.experts, 'ep', 1) > 1:
            if shared_states is None:
                return out_states
            if use_promoted_tp_reduce:
                out_states = out_states.to(tp_reduce_dtype)
                shared_states = shared_states.to(tp_reduce_dtype)
            if self._all_reduce:
                # moe_tp is one under pure EP, so its process group cannot be
                # used to reduce the attention/MLP-TP shared expert shards.
                if self._shared_expert_tp_group is None:
                    raise RuntimeError(
                        'EP shared-expert reduction requires an MLP TP process group')
                dist.all_reduce(
                    shared_states,
                    group=self._shared_expert_tp_group,
                )
            if use_promoted_tp_reduce:
                out_states += shared_states
                return out_states.to(output_dtype)
            out_states += shared_states
            return out_states

        if use_promoted_tp_reduce:
            out_states = out_states.to(tp_reduce_dtype)

        if shared_states is not None:
            if use_promoted_tp_reduce:
                shared_states = shared_states.to(tp_reduce_dtype)
            out_states += shared_states

        if self._all_reduce:
            if use_promoted_tp_reduce:
                dist.all_reduce(out_states, group=self.experts.tp_group)
                out_states = out_states.to(output_dtype)
            else:
                dist.all_reduce(out_states)

        return out_states


class KimiK2MLP(nn.Module):
    """Deepseek v2 mlp."""

    def __init__(self,
                 config: Any,
                 intermediate_size: int = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_shared_expert: bool = False,
                 prefix: str = ''):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        if is_shared_expert:
            dist_config = get_dist_manager().current_config()
            dp = dist_config.dp
            if dp == 1:
                # split weight, do all reduce in moe
                is_tp = True
                all_reduce = False
            else:
                # do not split weight on dp
                # TODO: support dp+tp?
                is_tp = False
                all_reduce = False
        else:
            all_reduce = True
            is_tp = True

        # gate up
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.gate_up_proj = build_gateup_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=is_tp,
            prefix=add_prefix('gate_up_proj', prefix),
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_down_linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
            all_reduce=all_reduce,
            prefix=add_prefix('down_proj', prefix),
        )

    def forward(self,
                x: torch.Tensor,
                all_routed_experts: torch.Tensor | None = None):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class KimiK2DecoderLayer(nn.Module):
    """Deepseekv2 decoder layer."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = None

        # build attention layer
        if getattr(config, 'use_mla', True):
            self.self_attn = KimiK2Attention(
                config,
                dtype=dtype,
                device=device,
                prefix=add_prefix('self_attn', prefix),
            )
        else:
            # deepseek-vl2-tiny uses MHA LlamaAttention structure
            from lmdeploy.pytorch.models.llama import LlamaAttention
            self.self_attn = LlamaAttention(config, dtype=dtype, device=device)

        # mlp
        self.mlp = (KimiK2MoE(
            config,
            layer_idx,
            dtype=dtype,
            device=device,
            prefix=add_prefix('mlp', prefix),
        ) if (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
              and layer_idx % config.moe_layer_freq == 0) else KimiK2MLP(
                  config,
                  dtype=dtype,
                  device=device,
                  prefix=add_prefix('mlp', prefix),
              ))

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       prefix=add_prefix('input_layernorm', prefix))

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('post_attention_layernorm', prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: list[torch.FloatTensor] | None,
        residual: torch.Tensor | None = None,
        attn_metadata: Any = None,
        all_routed_experts: torch.Tensor | None = None,
        capture_input_residual: bool = False,
    ) -> tuple[Any, ...]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        input_residual = (residual.clone()
                          if capture_input_residual else None)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(
            hidden_states, all_routed_experts=all_routed_experts)

        outputs = (hidden_states, residual)
        if input_residual is not None:
            outputs += (input_residual, )
        return outputs

    def forward_yield(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: list[torch.FloatTensor] | None,
        residual: torch.Tensor | None = None,
        attn_metadata: Any = None,
        tag: Any = None,
    ):
        """forward_yield."""
        is_decoding = attn_metadata.is_decoding
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # yield for attn0 and attn1
        yield
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MOE
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_idx = self.mlp.gate(hidden_states)

        topk_weights = self.mlp.experts.renormalize(topk_weights)
        topk_weights = topk_weights.to(torch.float32)
        topk_idx = topk_idx.to(torch.int64)
        hidden_shape = hidden_states.shape
        shared_states = None

        state = {
            'hidden_states': hidden_states,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'raw_hidden_shape': hidden_shape,
            'moe_type': MoeType.DSAsyncDecode if is_decoding else MoeType.DSAsyncPrefill,
        }

        self.mlp.experts.before_dispatch(state)

        # yield for attn1, dis (+share)
        yield
        recv_state = self.mlp.experts.dispatch(state)
        if self.mlp.shared_experts is not None and is_decoding:
            shared_states = self.mlp.shared_experts(hidden_states)
        # yield for dis, dis_wait
        yield
        self.mlp.experts.wait(recv_state)
        # yield for dis_wait, moe
        yield
        gemm_state = self.mlp.experts.gemm(recv_state)
        # yield for moe, comb
        yield
        out_state = self.mlp.experts.combine(gemm_state)
        # yield for comb, (+share) comb_wait
        yield
        if self.mlp.shared_experts is not None and not is_decoding:
            shared_states = self.mlp.shared_experts(hidden_states)
        self.mlp.experts.wait(out_state)
        # yield for (+share) comb_wait, (+share) attn0
        yield
        out_hidden_states = out_state['hidden_states'].view(hidden_shape)
        if shared_states is None and self.mlp.shared_experts is not None:
            shared_states = self.mlp.shared_experts(hidden_states)
        out_hidden_states = self.mlp._combine_expert_outputs(
            out_hidden_states, shared_states)
        out_hidden_states = out_hidden_states.reshape(batch_size, sequence_length, -1)
        outputs = (out_hidden_states, residual)
        return outputs


class KimiK2Model(nn.Module):
    """Mixtral model."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype = None,
        device: torch.device = None,
        prefix: str = '',
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(config.vocab_size,
                                              config.hidden_size,
                                              self.padding_idx,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

        if get_dist_manager().current_context().dist_config.enable_eplb:
            ep_size_, _ = get_ep_world_rank()
            EPLBManager.init_global_eplb_metadata(ep_size_, config.n_routed_experts, config.num_hidden_layers)
        self.layers = nn.ModuleList([
            KimiK2DecoderLayer(
                config,
                layer_idx,
                dtype=dtype,
                device=device,
                prefix=add_prefix(f'layers.{layer_idx}', prefix),
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.aux_hidden_state_layers: tuple[int, ...] = tuple(
            getattr(config, 'aux_hidden_state_layers', ()) or ())

        # build norm
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=None,
            dtype=dtype,
            device=device,
            prefix=add_prefix('norm', prefix),
        )

        emb_type = RopeType.LinearScaling
        rope_dim = config.qk_rope_head_dim if getattr(config, 'use_mla', True) else (config.hidden_size //
                                                                                     config.num_attention_heads)
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = get_rope_theta(config)

        rope_params = dict(emb_type=emb_type, dim=rope_dim, max_position_embeddings=rope_max_pos_emb, base=rope_base)
        update_params = build_rotary_params(config)
        rope_params.update(update_params)
        self.rotary_emb = build_rotary_embedding(**rope_params)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        all_routed_experts: torch.Tensor | None = None,
    ):
        """forward."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        aux_hidden_state_layers = self.aux_hidden_state_layers
        aux_hidden_states_by_layer: dict[int, torch.Tensor] = {}
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            capture_input_residual = idx in aux_hidden_state_layers
            layer_output = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
                all_routed_experts=all_routed_experts,
                capture_input_residual=capture_input_residual,
            )
            hidden_states, residual = layer_output[:2]
            if capture_input_residual:
                aux_hidden_states_by_layer[idx] = layer_output[2]

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_state_layers:
            aux_hidden_states = torch.cat([
                aux_hidden_states_by_layer[idx]
                for idx in aux_hidden_state_layers
            ], dim=-1)
            return dict(
                hidden_states=hidden_states,
                aux_hidden_states=aux_hidden_states,
            )
        return hidden_states

    def forward_microbatch(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ):
        """forward_microbatch."""
        assert self.config.moe_layer_freq == 1
        moe_start_idx = min(self.config.first_k_dense_replace, len(self.layers))

        # embed and mlplayers
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers[:moe_start_idx]):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        if moe_start_idx < len(self.layers):
            # run two micro batch
            num = 2
            input_list, exec_type, delta_stages, extern_tag = split_input(hidden_states,
                                                                          rotary_pos_emb,
                                                                          past_key_values,
                                                                          residual,
                                                                          attn_metadata,
                                                                          moe_start_idx,
                                                                          len(self.layers),
                                                                          num=num)

            output_list = execute_batch(inputs=input_list,
                                        fn=self.forward_yieldlayers,
                                        delta_stages=delta_stages,
                                        exec_type=exec_type,
                                        extern_tag=extern_tag)
            hidden_states, residual = merge_output(output_list)

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def forward_yieldlayers(self,
                            hidden_states: torch.Tensor,
                            rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
                            past_key_values: list[torch.FloatTensor] | None = None,
                            residual: torch.Tensor | None = None,
                            attn_metadata: Any = None,
                            start_idx: int = -1,
                            end_idx: int = -1,
                            tag: Any = None):
        """forward_yieldlayers."""
        for idx in range(start_idx, end_idx):
            past_key_value = past_key_values[idx]
            hidden_states, residual = yield from self.layers[idx].forward_yield(hidden_states,
                                                                                rotary_pos_emb=rotary_pos_emb,
                                                                                past_key_value=past_key_value,
                                                                                residual=residual,
                                                                                attn_metadata=attn_metadata,
                                                                                tag=tag)
        return hidden_states, residual

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class KimiK2ForCausalLM(nn.Module, CudaGraphMixin):
    """Mixture model for causalLM."""

    packed_modules_mapping = {}

    def __init__(self,
                 config: Any,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = KimiK2Model(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('model', prefix),
        )
        self.packed_modules_mapping = {}
        if any(getattr(layer.self_attn, 'fuse_qkv_a_proj', False) for layer in self.model.layers):
            self.packed_modules_mapping['fused_qkv_a_proj_with_mqa'] = [
                'q_a_proj',
                'kv_a_proj_with_mqa',
            ]
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device,
                                            prefix=add_prefix('lm_head', prefix))
        self._load_buffers = dict()
        self.enable_return_routed_experts = (
            get_build_model_context().enable_return_routed_experts)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        aux_hidden_states = None
        all_routed_experts = None
        if self.enable_return_routed_experts:
            num_tokens = (inputs_embeds.size(1)
                          if inputs_embeds is not None else input_ids.size(1))
            all_routed_experts = position_ids.new_zeros(
                (num_tokens, self.config.num_hidden_layers,
                 self.config.num_experts_per_tok),
                dtype=torch.uint16,
            )
        if get_step_ctx_manager().current_context().enable_microbatch:
            if all_routed_experts is not None:
                raise RuntimeError(
                    'KimiK2 routed-expert output is not supported with microbatch execution'
                )
            if self.model.aux_hidden_state_layers:
                raise RuntimeError(
                    'KimiK2 auxiliary hidden-state capture is not supported with microbatch execution'
                )
            hidden_states = self.model.forward_microbatch(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attn_metadata=attn_metadata,
                inputs_embeds=inputs_embeds,
            )
        else:
            model_output = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attn_metadata=attn_metadata,
                inputs_embeds=inputs_embeds,
                all_routed_experts=all_routed_experts,
            )
            if isinstance(model_output, dict):
                hidden_states = model_output['hidden_states']
                aux_hidden_states = model_output.get('aux_hidden_states')
            else:
                hidden_states = model_output
        if all_routed_experts is None and aux_hidden_states is None:
            return hidden_states
        output = {'hidden_states': hidden_states}
        if aux_hidden_states is not None:
            output['aux_hidden_states'] = aux_hidden_states
        if all_routed_experts is not None:
            output['all_routed_experts'] = all_routed_experts
        return output

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def get_outputs_cudagraph(self,
                              output_buffers: dict[str, torch.Tensor],
                              input_ids: torch.Tensor,
                              **kwargs):
        """Preserve auxiliary and routed-expert outputs on graph replay."""
        num_tokens = input_ids.size(-1)
        outputs = {
            'hidden_states': output_buffers['hidden_states'][:, :num_tokens]
        }
        if output_buffers.get('aux_hidden_states') is not None:
            outputs['aux_hidden_states'] = output_buffers[
                'aux_hidden_states'][:, :num_tokens]
        if output_buffers.get('all_routed_experts') is not None:
            outputs['all_routed_experts'] = output_buffers[
                'all_routed_experts'][:num_tokens, ...].clone()
        return outputs

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

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                             expert_params_mapping: list):
        """Load weight experts."""
        match = _COMPRESSED_TENSORS_EXPERT_WEIGHT_RE.fullmatch(name)
        if match is not None:
            suffix = match.group('suffix')
            if suffix in _COMPRESSED_TENSORS_EXPERT_SUFFIXES:
                projection = match.group('projection')
                param_group, shard_id = _COMPRESSED_TENSORS_EXPERT_PROJECTIONS[projection]
                param_name = f"{match.group('prefix')}.{param_group}.{suffix}"
                try:
                    param = params_dict[param_name]
                except KeyError:
                    raise KeyError(
                        f'Compressed-tensors expert weight {name!r} resolved to missing parameter {param_name!r}. '
                        'Check the routed-expert module prefix and quantization configuration.') from None
                load_weight(
                    param,
                    loaded_weight,
                    expert_id=int(match.group('expert_id')),
                    shard_id=shard_id,
                )
                return

            quantization_config = self.quantization_config
            quant_method = None
            if quantization_config is not None:
                quant_method = quantization_config.get('quant_method')
            if quant_method == 'compressed-tensors' and suffix.startswith('weight_'):
                supported = ', '.join(sorted(_COMPRESSED_TENSORS_EXPERT_SUFFIXES))
                raise ValueError(f'Unsupported compressed-tensors expert weight suffix {suffix!r} in {name!r}; '
                                 f'supported suffixes are: {supported}.')

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

        fused_a_proj_mapping = (
            ('q_a_proj', 'q'),
            ('kv_a_proj_with_mqa', 'kv'),
        )
        for source_name, shard_id in fused_a_proj_mapping:
            source_marker = f'.{source_name}.'
            if source_marker not in name:
                continue
            fused_name = name.replace(source_marker, '.fused_qkv_a_proj_with_mqa.')
            if fused_name not in params_dict:
                break
            weight = loaded_weight
            if source_name == 'kv_a_proj_with_mqa' and not name.endswith('.weight_scale_inv'):
                weight = __update_pe(
                    loaded_weight.to(device),
                    self.config.kv_lora_rank + self.config.qk_rope_head_dim,
                    self.config.kv_lora_rank,
                )
            load_weight(params_dict[fused_name], weight, shard_id=shard_id)
            return

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
                fp8_quant_scope = None
                if quantization_config is not None:
                    quant_method = quantization_config.get('quant_method')
                    fp8_quant_scope = quantization_config.get('fp8_quant_scope')

                loaded_weight = loaded_weight.to(device)
                if quant_method == 'fp8' and fp8_quant_scope != 'moe_only':
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

        def __skip_layers():
            """We might change the number of layers so we can debug the model
            with less gpus."""
            import re
            matches = re.findall(r'\.layers\.(\d+)\.', name)
            layer_id = int(matches[0])
            return layer_id >= self.config.num_hidden_layers

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        config = self.config

        update_pe_mapping = []
        if getattr(config, 'use_mla', True):
            qk_rope_head_dim = config.qk_rope_head_dim
            kv_lora_rank = config.kv_lora_rank
            qk_nope_head_dim = config.qk_nope_head_dim
            q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
            kv_dim = kv_lora_rank + qk_rope_head_dim
            update_pe_mapping = [('q_proj', q_head_dim, qk_nope_head_dim), ('q_b_proj', q_head_dim, qk_nope_head_dim),
                                 ('kv_a_proj_with_mqa', kv_dim, kv_lora_rank)]
        else:
            # deepseek-vl2-tiny uses MHA LlamaAttention, weight loading differs from MLA
            stacked_params_mapping.extend([
                # (param_name, shard_name, shard_id)
                ('.qkv_proj', '.q_proj', 'q'),
                ('.qkv_proj', '.k_proj', 'k'),
                ('.qkv_proj', '.v_proj', 'v'),
            ])

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
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if '.layers' in name:
                # skip nextn
                if __skip_nextn(name, nextn_keys):
                    continue

                if __skip_layers():
                    continue

            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            if '.experts' in name:
                self._load_weight_experts(name, loaded_weight, params_dict, expert_params_mapping=expert_params_mapping)
            elif '.self_attn' in name and getattr(config, 'use_mla', True):
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
