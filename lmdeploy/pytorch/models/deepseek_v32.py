# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank
from lmdeploy.pytorch.model_inputs import StepContextManager, get_step_ctx_manager
from lmdeploy.pytorch.nn import (
    ApplyRotaryEmb,
    Attention,
    RMSNorm,
    RopeType,
    build_rotary_embedding,
    build_rotary_params,
)
from lmdeploy.pytorch.nn.eplb import EPLBManager
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_o_proj, build_rowwise_linear
from lmdeploy.pytorch.nn.nsa import IndexerTopKFP8
from lmdeploy.pytorch.nn.rotary_embedding import get_rope_parameters, get_rope_theta

from .deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2BMM,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
    yarn_get_mscale,
)
from .patch import get_build_model_context


def get_layer_indexer_type(config: Any, layer_idx: int | None) -> str:
    """Return whether a DSA layer computes or reuses top-k indices."""
    indexer_types = getattr(config, 'indexer_types', None)
    if indexer_types is None or layer_idx is None or layer_idx >= len(indexer_types):
        return 'full'
    return indexer_types[layer_idx]


def get_layer_idx_from_weight_name(name: str) -> int | None:
    """Parse a transformer layer index from a checkpoint parameter name."""
    for marker in ('.layers.', 'layers.'):
        if marker not in name:
            continue
        try:
            return int(name.split(marker, 1)[1].split('.', 1)[0])
        except ValueError:
            return None
    return None


class DSATopKIndicesBuffer(nn.Module):
    """Persistent DSA top-k buffer shared by full and reuse layers."""

    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk
        self.register_buffer('indices', None, persistent=False)

    def _target_capacity(self, num_tokens: int) -> int:
        capacity = num_tokens
        ctx_mgr = get_step_ctx_manager()
        if ctx_mgr is None:
            return capacity

        context = ctx_mgr.current_context()
        cache_config = getattr(context, 'cache_config', None)
        max_prefill_token_num = getattr(cache_config, 'max_prefill_token_num', None)
        if max_prefill_token_num is not None:
            capacity = max(capacity, max_prefill_token_num)
        return capacity

    def ensure(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Return a stable top-k slice with enough capacity for the current
        forward."""
        capacity = self._target_capacity(num_tokens)
        if (self.indices is None or self.indices.size(0) < capacity or self.indices.device != device):
            self.indices = torch.empty(capacity, self.topk, dtype=torch.int32, device=device)
        return self.indices[:num_tokens]

    def write(self, topk_indices: torch.Tensor) -> torch.Tensor:
        """Copy freshly computed top-k indices into the shared buffer."""
        buffer = self.ensure(topk_indices.size(0), topk_indices.device)
        buffer.copy_(topk_indices)
        return buffer

    def read(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Read top-k indices previously written by a full indexer layer."""
        if self.indices is None or self.indices.size(0) < num_tokens or self.indices.device != device:
            raise RuntimeError('DSA top-k indices are reused before the shared buffer is populated.')
        return self.indices[:num_tokens]

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved RoPE pairs."""
    out = torch.empty_like(x)
    out[..., ::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., ::2]
    return out


def apply_interleaved_rotary_pos_emb(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply GPT-J style interleaved RoPE to query and key."""

    def _to_interleaved(freqs: torch.Tensor):
        return freqs[..., :freqs.size(-1) // 2].repeat_interleave(2, dim=-1)

    cos = _to_interleaved(cos).unsqueeze(-2)
    sin = _to_interleaved(sin).unsqueeze(-2)
    query = query * cos + rotate_interleaved(query) * sin
    key = key * cos + rotate_interleaved(key) * sin
    return query, key


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
        self.rope_interleave: bool = getattr(config, 'indexer_rope_interleave', False)
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
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.indexer_topk = IndexerTopKFP8(self.index_topk, self.softmax_scale, block_size=128, fill=-1)

    def _apply_rotary_pos_emb(self, q_pe: torch.Tensor, k_pe: torch.Tensor,
                              freqs_cis: tuple[torch.Tensor, torch.Tensor]):
        """Apply the indexer's RoPE layout."""
        cos, sin = freqs_cis
        k_pe = k_pe[..., None, :]
        if self.rope_interleave:
            return apply_interleaved_rotary_pos_emb(q_pe, k_pe, cos, sin)
        return self.apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            inplace=False,
        )

    def forward(self,
                x: torch.Tensor,
                qr: torch.Tensor,
                freqs_cis: torch.Tensor,
                index_cache: tuple[torch.Tensor, torch.Tensor],
                attn_metadata: Any = None):
        q = self.wq_b(qr)
        q = q.unflatten(-1, (-1, self.head_dim))
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        # apply rotary embedding
        q_pe, k_pe = self._apply_rotary_pos_emb(q_pe, k_pe, freqs_cis)
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

        rope_scaling = get_rope_parameters(config)
        if rope_scaling is not None:
            mscale_all_dim = rope_scaling.get('mscale_all_dim', 0)
            if mscale_all_dim:
                scaling_factor = rope_scaling['factor']
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

        self.indexer_type = get_layer_indexer_type(config, layer_idx)
        self.indexer = None
        if self.indexer_type == 'full':
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
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Sequence[torch.Tensor] = None,
        attn_metadata: Any = None,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        all_indexer_topk: torch.Tensor | None = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        dist_config = get_dist_manager().current_config()
        if dist_config.dp > 1:
            num_heads = self.num_heads
        else:
            num_heads = self.num_heads // dist_config.attn_tp
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

        if self.indexer is None:
            if topk_indices_buffer is None:
                raise RuntimeError(f'Layer {self.layer_idx} reuses DSA top-k indices but none were provided.')
            topk_indices = topk_indices_buffer.read(q_len, hidden_states.device)
        else:
            topk_indices = self.indexer(hidden_states,
                                        qr,
                                        rotary_pos_emb,
                                        past_key_value[-2:],
                                        attn_metadata=attn_metadata)
            if topk_indices_buffer is not None:
                topk_indices = topk_indices_buffer.write(topk_indices)

        if all_indexer_topk is not None:
            all_indexer_topk[:, self.layer_idx, :].copy_(topk_indices)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: list[torch.FloatTensor] | None,
        residual: torch.Tensor | None = None,
        attn_metadata: Any = None,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        all_routed_experts: torch.Tensor | None = None,
        all_indexer_topk: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if isinstance(self.self_attn, DeepseekV32Attention):
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
                topk_indices_buffer=topk_indices_buffer,
                all_indexer_topk=all_indexer_topk,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
            )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states, all_routed_experts=all_routed_experts)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


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
        self.topk_indices_buffer = DSATopKIndicesBuffer(config.index_topk)

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
        all_indexer_topk: torch.Tensor | None = None,
    ):
        """forward."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
                topk_indices_buffer=self.topk_indices_buffer,
                all_routed_experts=all_routed_experts,
                all_indexer_topk=all_indexer_topk,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def forward_microbatch(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        all_routed_experts: torch.Tensor | None = None,
        all_indexer_topk: torch.Tensor | None = None,
    ):
        """forward_microbatch."""
        # DSA shared top-k indices are model-global; use normal forward until
        # microbatching has per-microbatch top-k buffers.
        return self.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            all_routed_experts=all_routed_experts,
            all_indexer_topk=all_indexer_topk,
        )


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
        bm_ctx = get_build_model_context()
        self.enable_return_routed_experts = bm_ctx.enable_return_routed_experts
        self.enable_return_indexer_topk = bm_ctx.enable_return_indexer_topk

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward with optional exact DSA indexer capture."""
        step_ctx = get_step_ctx_manager().current_context()
        num_tokens = inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1)
        all_routed_experts = None
        if self.enable_return_routed_experts:
            all_routed_experts = position_ids.new_full(
                (num_tokens, self.config.num_hidden_layers, self.config.num_experts_per_tok),
                torch.iinfo(torch.uint16).max,
                dtype=torch.uint16,
            )
        all_indexer_topk = None
        if self.enable_return_indexer_topk:
            all_indexer_topk = position_ids.new_empty(
                (num_tokens, self.config.num_hidden_layers, self.config.index_topk), dtype=torch.int32)

        forward = self.model.forward_microbatch if step_ctx.enable_microbatch else self.model.forward
        hidden_states = forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            all_routed_experts=all_routed_experts,
            all_indexer_topk=all_indexer_topk,
        )
        if all_routed_experts is None and all_indexer_topk is None:
            return hidden_states
        outputs = dict(hidden_states=hidden_states)
        if all_routed_experts is not None:
            outputs['all_routed_experts'] = all_routed_experts
        if all_indexer_topk is not None:
            outputs['all_indexer_topk'] = all_indexer_topk
        return outputs

    def _load_weight_attention(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                               update_pe_mapping: list):
        """Load attention weights."""
        if '.self_attn.indexer.' in name and name not in params_dict:
            layer_idx = get_layer_idx_from_weight_name(name)
            # Shared DSA layers reuse cached top-k indices and have no local indexer.
            if get_layer_indexer_type(self.config, layer_idx) == 'shared':
                return
        return super()._load_weight_attention(name, loaded_weight, params_dict, update_pe_mapping)
