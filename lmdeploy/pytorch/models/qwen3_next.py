# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_merged_colwise_linear, build_o_proj, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.nn.moe import SoftmaxTopK, build_fused_moe
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


class CausalConv1d:

    def __init__(self, activation: str = 'silu'):
        try:
            import causal_conv1d
            self.causal_conv1d_fn = causal_conv1d.causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d.causal_conv1d_update
        except Exception:
            raise RuntimeError(
                'causal_conv1d is not installed, please refer to https://github.com/Dao-AILab/causal-conv1d')
        self.activation = activation

    def conv1d_func(self, x: torch.Tensor, conv1d: nn.Conv1d, seqlens: torch.Tensor):
        weight = conv1d.weight
        bias = conv1d.bias
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]
        num_tokens = x.size(-1)
        batch_size = seqlens.numel()
        device = seqlens.device
        batch_idx = torch.arange(0, batch_size, dtype=torch.int32, device=device)
        seq_idx = torch.repeat_interleave(batch_idx, seqlens, output_size=num_tokens)

        return self.causal_conv1d_fn(
            x,
            weight,
            bias,
            seq_idx,
            return_final_states=True,
            activation=self.activation,
        )

    def conv1d_update(self, x: torch.Tensor, conv1d: nn.Conv1d, conv_state: torch.Tensor):
        weight = conv1d.weight
        bias = conv1d.bias
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]
        return self.causal_conv1d_update(x, conv_state, weight, bias, activation=self.activation)

    def __call__(self, x: torch.Tensor, conv1d: nn.Conv1d, conv_state: torch.Tensor, seqlens: torch.Tensor):
        num_tokens = x.size(-1)
        batch_size = seqlens.numel()

        if num_tokens == batch_size:
            return self.conv1d_update(x, conv1d, conv_state), conv_state
        return self.conv1d_func(x, conv1d, seqlens)


class GatedDelta:

    def __init__(self, use_qk_l2norm_in_kernel: bool = True):
        try:
            import fla
            self.chunk_gated_delta_rule = fla.chunk_gated_delta_rule
            self.fused_recurrent_gated_delta_rule = fla.fused_recurrent_gated_delta_rule
        except Exception:
            raise RuntimeError(
                'fla is not installed, please refer to https://github.com/fla-org/flash-linear-attention')
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        is_decoding: bool,
        cu_seqlens: torch.Tensor,
    ):
        """call."""
        if not is_decoding:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )

        else:
            core_attn_out, last_recurrent_state = self.fused_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
            )
        return core_attn_out, last_recurrent_state


class Qwen3NextGatedDeltaNet(nn.Module):
    """Gated deltanet."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.kv_ratio = self.num_v_heads // self.num_k_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            dtype=dtype,
            device=device,
        )

        # projection of the input hidden states
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = build_colwise_linear(self.hidden_size,
                                                 projection_size_qkvz,
                                                 bias=False,
                                                 dtype=dtype,
                                                 device=device,
                                                 is_tp=True)
        self.in_proj_ba = build_colwise_linear(self.hidden_size,
                                               projection_size_ba,
                                               bias=False,
                                               dtype=dtype,
                                               device=device,
                                               is_tp=True)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, device=device))

        A = torch.empty(self.num_v_heads, device=device).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log_exp = None

        self.norm = RMSNorm(self.head_v_dim, eps=self.layer_norm_epsilon, dtype=dtype, device=device)
        self.out_proj = build_o_proj(self.value_dim,
                                     self.hidden_size,
                                     bias=False,
                                     dtype=dtype,
                                     device=device,
                                     is_tp=True)

        self.causal_conv1d = CausalConv1d(activation='silu')
        self.gated_delta = GatedDelta()

    def get_A_log_exp(self):
        if self.A_log_exp is None:
            self.A_log_exp = -self.A_log.float().exp()

        return self.A_log_exp

    def fix_query_key_value_ordering(self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor):
        """Derives `query`, `key` and `value` tensors from `mixed_qkvz` and
        `mixed_ba`."""
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            -1,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.kv_ratio,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (-1, 2 * self.kv_ratio)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.kv_ratio * self.head_v_dim),
            (self.kv_ratio * self.head_v_dim),
        ]
        split_arg_list_ba = [self.kv_ratio, self.kv_ratio]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=-1)
        # [..., ng, np/ng * hn] -> [..., np, hn]
        value = value.reshape(*value.shape[:-2], -1, self.head_v_dim)
        z = z.reshape(*z.shape[:-2], -1, self.head_v_dim)
        b = b.reshape(*b.shape[:-2], self.num_v_heads)
        a = a.reshape(*a.shape[:-2], self.num_v_heads)
        return query, key, value, z, b, a

    def _load_state(self, past_key_value: Tuple[torch.Tensor, torch.Tensor], state_ids: torch.Tensor):
        """Load states from cache."""
        conv_cache, recurrent_cache = past_key_value[:2]

        return conv_cache.index_select(0, state_ids), recurrent_cache.index_select(0, state_ids)

    def _store_state(self, conv_state: torch.Tensor, recurrent_state: torch.Tensor,
                     past_key_value: Tuple[torch.Tensor, torch.Tensor], state_ids: torch.Tensor):
        """Store states to cache."""
        conv_cache, recurrent_cache = past_key_value[:2]

        conv_cache = conv_cache.index_copy(0, state_ids, conv_state)
        recurrent_cache = recurrent_cache.index_copy(0, state_ids, recurrent_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        state_ids: torch.Tensor,
        attn_metadata: Any,
    ):
        """forward."""
        seqlens = attn_metadata.q_seqlens
        cu_seqlens = attn_metadata.cu_seqlens_q
        is_decoding = attn_metadata.is_decoding

        # load states
        conv_state, recurrent_state = self._load_state(past_key_value, state_ids)

        # inputs proj
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(*x.shape[:-2], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(-2, -1)

        mixed_qkv, conv_state = self.causal_conv1d(mixed_qkv, self.conv1d, conv_state, seqlens=seqlens)

        mixed_qkv = mixed_qkv.transpose(-2, -1)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.unflatten(-1, (-1, self.head_k_dim))
        key = key.unflatten(-1, (-1, self.head_k_dim))
        value = value.unflatten(-1, (-1, self.head_v_dim))

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = self.get_A_log_exp() * F.softplus(a.float() + self.dt_bias)
        if self.kv_ratio > 1:
            query = query.repeat_interleave(self.kv_ratio, dim=-2)
            key = key.repeat_interleave(self.kv_ratio, dim=-2)

        core_attn_out, recurrent_state = self.gated_delta(
            query,
            key,
            value,
            g=g,
            beta=beta,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
            is_decoding=is_decoding,
        )

        # store states
        self._store_state(conv_state, recurrent_state, past_key_value, state_ids)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        return output


class Qwen3NextAttention(nn.Module):
    """Rewrite module of Qwen3MoeAttention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        self.head_dim = head_dim
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)

        # packed qkv
        # Qwen3 uses 'config.attention_bias = False' for q/k/o projections
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads * 2,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=config.attention_bias,
            quant_config=quantization_config,
            num_replicate_kv_heads=num_replicate_kv_heads,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
        )

        # o_proj
        self.o_proj = build_o_proj(num_heads * head_dim,
                                   hidden_size,
                                   bias=config.attention_bias,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   is_tp=True)

        # q, k norm
        self.q_norm = RMSNorm(head_dim,
                              config.rms_norm_eps,
                              quant_config=quantization_config,
                              dtype=dtype,
                              device=device)
        self.k_norm = RMSNorm(head_dim,
                              config.rms_norm_eps,
                              quant_config=quantization_config,
                              dtype=dtype,
                              device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)
        query_states, gate = query_states.view(*query_states.shape[:-2], -1, 2 * self.head_dim).chunk(2, dim=-1)

        # apply q, k norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        gate = gate.reshape(*hidden_states.shape[:-1], -1)
        attn_output = attn_output * gate.sigmoid()

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3NextMLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 intermediate_size: int = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_tp: bool = True,
                 all_reduce: bool = True):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=is_tp,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(intermediate_size,
                                              config.hidden_size,
                                              bias=False,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=is_tp,
                                              all_reduce=all_reduce)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class Qwen3NextSparseMoeBlock(nn.Module):
    """Moe block."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
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

        self.softmax_topk = SoftmaxTopK(self.top_k)

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

        self.shared_expert = Qwen3NextMLP(
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
        dp = dist_ctx.dp
        world_size = dist_ctx.world_size
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


class Qwen3NextDecoderLayer(nn.Module):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx, dtype=dtype, device=device)
        elif self.layer_type == 'full_attention':
            self.self_attn = Qwen3NextAttention(config, dtype=dtype, device=device)

        # build MLP
        if (layer_idx not in config.mlp_only_layers) and (config.num_experts
                                                          > 0) and ((layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3NextSparseMoeBlock(config, layer_idx=layer_idx, dtype=dtype, device=device)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor],
        attn_metadata: Any,
        state_ids: torch.Tensor,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        if self.layer_type == 'linear_attention':
            past_states = past_key_value
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                past_states=past_states,
                state_ids=state_ids,
                attn_metadata=attn_metadata,
            )
        elif self.layer_type == 'full_attention':
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
            )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Qwen3NextModel(nn.Module):
    """Qwen3 next model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
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
        self.layers = nn.ModuleList(
            [Qwen3NextDecoderLayer(config, layer_idx, dtype=dtype, device=device) for layer_idx in range(4)])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = build_rotary_embedding_from_config(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: List[torch.FloatTensor],
        attn_metadata: Any,
        state_ids: torch.Tensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_values[idx],
                residual=residual,
                attn_metadata=attn_metadata,
                state_ids=state_ids,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class Qwen3NextForCausalLM(nn.Module, CudaGraphMixin):
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
        self.ctx_mgr = ctx_mgr
        # build model
        self.model = Qwen3NextModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        state_ids: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=state_ids,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

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

        # make past_key_values
        state_caches = list(cache.permute(0, 1) for cache in context.state_caches)
        state_caches = list(zip(state_caches[0], state_caches[1]))
        past_key_values = list(past_key_values)
        new_past_key_values = []
        for layer_type in self.config.layer_types:
            if layer_type == 'linear_attention':
                new_past_key_values.append(state_caches.pop(0))
            elif layer_type == 'full_attention':
                new_past_key_values.append(past_key_values.pop(0))

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
        )

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        max_batchs = graph_meta.max_batchs
        device = graph_meta.device

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        state_ids = torch.zeros(max_batchs, dtype=torch.long, device=device)
        input_buffers['state_ids'] = state_ids

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Fill cudagraph buffers from forward inputs."""
        input_buffers = graph_meta.input_buffers

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        state_ids = kwargs['state_ids']
        input_buffers['state_ids'][:state_ids.size(0)].copy_(state_ids)
        new_inputs['state_ids'] = input_buffers['state_ids']

        return new_inputs

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
        # TODO: remove this after debug
        return
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

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if '.linear_attn' in name:
                # TODO: support linear attn
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
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
