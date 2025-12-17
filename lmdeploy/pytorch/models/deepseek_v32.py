# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank
from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType, build_rotary_embedding,
                                 build_rotary_params)
from lmdeploy.pytorch.nn.eplb import EPLBManager
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_o_proj, build_rowwise_linear
from lmdeploy.pytorch.nn.nsa import IndexerTopKFP8

from .deepseek_v2 import (DeepseekV2Attention, DeepseekV2BMM, DeepseekV2DecoderLayer, DeepseekV2ForCausalLM,
                          DeepseekV2MLP, DeepseekV2Model, DeepseekV2MoE, yarn_get_mscale)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class LayerNorm(nn.Module):
    """Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device = None):
        super().__init__()
        if device is None:
            device = 'cuda'
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim, ), self.weight, self.bias, self.eps).type_as(x)


class Indexer(nn.Module):

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        try:
            import fast_hadamard_transform  # noqa: F401
        except ImportError:
            raise ImportError('Please install fast_hadamard_transform package.')
        quant_config = getattr(config, 'quantization_config', None)
        self.layer_idx = layer_idx
        # self.dim: int = 2048
        self.dim: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.n_local_heads = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.wq_b = build_colwise_linear(self.q_lora_rank,
                                         self.n_heads * self.head_dim,
                                         bias=False,
                                         dtype=dtype,
                                         device=device,
                                         is_tp=False,
                                         quant_config=quant_config)
        self.wk = build_colwise_linear(self.dim,
                                       self.head_dim,
                                       bias=False,
                                       dtype=dtype,
                                       device=device,
                                       is_tp=False,
                                       quant_config=quant_config)
        self.k_norm = LayerNorm(self.head_dim, device=device)
        self.weights_proj = build_colwise_linear(self.dim,
                                                 self.n_heads,
                                                 bias=False,
                                                 dtype=dtype,
                                                 device=device,
                                                 is_tp=False)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = quant_config['scale_fmt']
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.indexer_topk = IndexerTopKFP8(self.index_topk, self.softmax_scale, block_size=128, fill=-1)

    def forward(self,
                x: torch.Tensor,
                qr: torch.Tensor,
                freqs_cis: torch.Tensor,
                index_cache: Tuple[torch.Tensor, torch.Tensor],
                attn_metadata: Any = None):
        q = self.wq_b(qr)
        q = q.unflatten(-1, (-1, self.head_dim))
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        # apply rotary embedding
        cos, sin = freqs_cis
        q_pe, k_pe = self.apply_rotary_pos_emb(
            q_pe,
            k_pe[..., None, :],
            cos,
            sin,
            inplace=False,
        )
        k_pe = k_pe[0, :]
        k_nope = k_nope[0, :, None]
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)

        weights = self.weights_proj(x) * self.n_heads**-0.5

        return self.indexer_topk(q[0], k[:, 0], weights[0], index_cache[0], index_cache[1], attn_metadata=attn_metadata)


class DeepseekV32Attention(DeepseekV2Attention):

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
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
            )
            self.q_a_layernorm = RMSNorm(config.q_lora_rank,
                                         1e-6,
                                         quant_config=quantization_config,
                                         dtype=torch.float32,
                                         device=device)
            self.q_b_proj = build_colwise_linear(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                is_tp=True,
                quant_config=quantization_config,
                dp_disable_tp=True,
            )

        self.kv_a_proj_with_mqa = build_colwise_linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank,
                                      1e-6,
                                      quant_config=quantization_config,
                                      dtype=torch.float32,
                                      device=device)
        self.kc = DeepseekV2BMM(self.num_heads,
                                config.qk_nope_head_dim,
                                config.kv_lora_rank,
                                dtype=dtype,
                                device=device)

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        self.softmax_scale = self.q_head_dim**(-0.5)

        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get('mscale_all_dim', 0)
            scaling_factor = config.rope_scaling['factor']
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

        self.vc = DeepseekV2BMM(self.num_heads, config.kv_lora_rank, self.v_head_dim, dtype=dtype, device=device)
        self.o_proj = build_o_proj(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
        )

        self.indexer = Indexer(config, layer_idx, dtype=dtype, device=device)

    def _q_proj(self, hidden_states, num_heads: int, nope_size: int, pe_size: int):
        """Q proj."""
        q_len = hidden_states.size(1)

        query_states = hidden_states.new_empty(q_len, num_heads, nope_size + pe_size)

        if self.q_lora_rank is None:
            qr = hidden_states
            q = self.q_proj(hidden_states)
        else:
            qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(qr)
        q = q.view(q_len, num_heads, self.q_head_dim)
        # q_pe: (q_len, num_heads, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope: (q_len, num_heads, kv_lora_rank)
        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe, qr

    def _kv_proj(self, hidden_states, nope_size: int):
        """Kv proj."""
        # (q_len, 1, nope_size + pe_size)
        key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
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
        query_states, q_pe, qr = self._q_proj(hidden_states, num_heads, nope_size, pe_size)
        key_states, value_states, k_pe = self._kv_proj(hidden_states, nope_size)

        return query_states, key_states, value_states, q_pe, k_pe, qr

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Sequence[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        dist_ctx = get_dist_manager().current_context()
        tp_world_size = dist_ctx.dist_config.attn_tp
        num_heads = self.num_heads // tp_world_size
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(1)

        # qkv_proj
        query_states, key_states, value_states, q_pe, k_pe, qr = self._qkv_proj(hidden_states, num_heads=num_heads)

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

        topk_indices = self.indexer(hidden_states, qr, rotary_pos_emb, past_key_value[-2:], attn_metadata=attn_metadata)

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[0][..., :nope_size],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            nsa_indices=topk_indices,
        )
        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)

        self.vc(attn_output, attn_bmm_out)
        attn_output = attn_bmm_out.flatten(-2, -1)[None]
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepseekV32DecoderLayer(DeepseekV2DecoderLayer):

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        quantization_config = None

        # build attention layer
        if getattr(config, 'use_mla', True):
            self.self_attn = DeepseekV32Attention(config, layer_idx, dtype=dtype, device=device)
        else:
            # deepseek-vl2-tiny uses MHA LlamaAttention structure
            from lmdeploy.pytorch.models.llama import LlamaAttention
            self.self_attn = LlamaAttention(config, dtype=dtype, device=device)

        # mlp
        self.mlp = (DeepseekV2MoE(config, layer_idx, dtype=dtype, device=device) if
                    (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
                     and layer_idx % config.moe_layer_freq == 0) else DeepseekV2MLP(config, dtype=dtype, device=device))

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=torch.float32,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                dtype=torch.float32,
                                                device=device)


class DeepseekV32Model(DeepseekV2Model):

    def __init__(self, config: Any, dtype: torch.dtype = None, device: torch.device = None):
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        if get_dist_manager().current_context().dist_config.enable_eplb:
            ep_size_, _ = get_ep_world_rank()
            EPLBManager.init_global_eplb_metadata(ep_size_, config.n_routed_experts, config.num_hidden_layers)
        self.layers = nn.ModuleList([
            DeepseekV32DecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=None,
                            dtype=torch.float32,
                            device=device)

        emb_type = RopeType.LinearScaling
        rope_dim = config.qk_rope_head_dim if getattr(config, 'use_mla', True) else (config.hidden_size //
                                                                                     config.num_attention_heads)
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta

        rope_params = dict(emb_type=emb_type, dim=rope_dim, max_position_embeddings=rope_max_pos_emb, base=rope_base)
        update_params = build_rotary_params(config)
        rope_params.update(update_params)
        self.rotary_emb = build_rotary_embedding(**rope_params)


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):

    def __init__(self,
                 config: Any,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        nn.Module.__init__(self)
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = DeepseekV32Model(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)
        self._load_buffers = dict()
