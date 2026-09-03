# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank
from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import (
    ApplyRotaryEmb,
    Attention,
    ParallelLMHead,
    RMSNorm,
    RopeType,
    build_rotary_embedding,
    build_rotary_params,
)
from lmdeploy.pytorch.nn.eplb import EPLBManager
from lmdeploy.pytorch.nn.linear import (
    build_colwise_linear,
    build_merged_colwise_linear,
    build_o_proj,
)
from lmdeploy.pytorch.nn.nsa import IndexerTopKFP8
from lmdeploy.pytorch.nn.rotary_embedding import get_rope_parameters, get_rope_theta
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

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


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def _dequantize_blocked_fp8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a 2D block-FP8 checkpoint tensor."""
    dim_w0, dim_w1 = weight.shape
    dim_s0, dim_s1 = scale.shape
    assert dim_w0 % dim_s0 == 0 and dim_w1 % dim_s1 == 0
    weight = weight.reshape(dim_s0, dim_w0 // dim_s0, dim_s1, dim_w1 // dim_s1)
    weight = weight.float() * scale.reshape(dim_s0, 1, dim_s1, 1)
    return weight.to(dtype).reshape(dim_w0, dim_w1)


def _load_fused_indexer_weight(name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                               load_buffers: dict) -> bool:
    """Load separate checkpoint projections into one fused BF16 weight."""
    is_wk = '.self_attn.indexer.wk.' in name
    is_gate = '.self_attn.indexer.weights_proj.' in name
    if not (is_wk or is_gate):
        return False

    indexer_prefix = name.rsplit('.indexer.', 1)[0] + '.indexer'
    fused_param = params_dict.get(f'{indexer_prefix}.wk_weights_proj.weight')
    if fused_param is None:
        return False

    if is_gate:
        if not name.endswith('.weight'):
            return False
        gate = loaded_weight.to(device=fused_param.device, dtype=fused_param.dtype)
        fused_param.data[-gate.size(0):].copy_(gate)
        return True

    if name.endswith('.weight') and loaded_weight.dtype != torch.float8_e4m3fn:
        wk = loaded_weight.to(device=fused_param.device, dtype=fused_param.dtype)
        fused_param.data[:wk.size(0)].copy_(wk)
        return True

    is_weight = name.endswith('.weight')
    is_scale = name.endswith('.weight_scale_inv')
    if not (is_weight or is_scale):
        return False

    buffer = load_buffers.setdefault(f'{indexer_prefix}.wk', {})
    buffer['weight' if is_weight else 'scale'] = loaded_weight.to(fused_param.device)
    if 'weight' in buffer and 'scale' in buffer:
        wk = _dequantize_blocked_fp8(buffer['weight'], buffer['scale'], fused_param.dtype)
        fused_param.data[:wk.size(0)].copy_(wk)
        load_buffers.pop(f'{indexer_prefix}.wk')
    return True


def _load_fused_qkv_a_weight(name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                             config: Any) -> bool:
    """Load separate Q/KV-A checkpoint tensors into their fused projection."""
    shard_id = None
    if '.self_attn.q_a_proj.' in name:
        shard_id = 0
        fused_name = name.replace('.q_a_proj.', '.fused_qkv_a_proj.')
    elif '.self_attn.kv_a_proj_with_mqa.' in name:
        shard_id = 1
        fused_name = name.replace('.kv_a_proj_with_mqa.', '.fused_qkv_a_proj.')
    else:
        return False

    param = params_dict.get(fused_name)
    if param is None:
        return False

    if shard_id == 1 and not name.endswith('.weight_scale_inv'):
        kv_dim = config.kv_lora_rank + config.qk_rope_head_dim
        loaded_weight = loaded_weight.to(param.device).unflatten(0, (-1, kv_dim))
        rope_weight = loaded_weight[:, config.kv_lora_rank:]
        rope_weight = rope_weight.unflatten(1, (-1, 2)).transpose(1, 2).flatten(1, 2)
        loaded_weight[:, config.kv_lora_rank:] = rope_weight
        loaded_weight = loaded_weight.flatten(0, 1)

    load_weight(param, loaded_weight, shard_id=shard_id)
    return True


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
        self.use_fusion = not _envs.disable_dsa_indexer_fusion
        if self.use_fusion:
            self.wk_weights_proj = build_colwise_linear(self.dim,
                                                        self.head_dim + self.n_heads,
                                                        bias=False,
                                                        dtype=torch.bfloat16,
                                                        device=device,
                                                        is_tp=False)
        else:
            self.wk = build_colwise_linear(self.dim,
                                           self.head_dim,
                                           bias=False,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=False,
                                           quant_config=quant_config)
            self.weights_proj = build_colwise_linear(self.dim,
                                                     self.n_heads,
                                                     bias=False,
                                                     dtype=dtype,
                                                     device=device,
                                                     is_tp=False)
        self.k_norm = LayerNorm(self.head_dim, device=device)
        self.softmax_scale = self.head_dim**-0.5
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.indexer_topk = IndexerTopKFP8(self.index_topk,
                                           self.softmax_scale,
                                           self.head_dim,
                                           block_size=128,
                                           fill=-1,
                                           # MTP may reuse its first iteration's indices in later drafts.
                                           allow_short_prefill_scoring_skip=layer_idx
                                           < config.num_hidden_layers)

    def forward(self,
                x: torch.Tensor,
                qr: torch.Tensor,
                freqs_cis: torch.Tensor,
                attn_metadata: Any = None):
        q = self.wq_b(qr)
        q = q.unflatten(-1, (-1, self.head_dim))
        if self.use_fusion:
            kw = self.wk_weights_proj(x)
            k, weights = kw.split([self.head_dim, self.n_heads], dim=-1)
            cos, sin = freqs_cis
            return self.indexer_topk.forward_fused(q[0],
                                                   k[0],
                                                   weights[0],
                                                   self.k_norm.weight,
                                                   self.k_norm.bias,
                                                   cos,
                                                   sin,
                                                   norm_eps=self.k_norm.eps,
                                                   head_gate_scale=self.n_heads**-0.5,
                                                   rope_interleaved=False,
                                                   attn_metadata=attn_metadata)

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

        return self.indexer_topk(q[0], k[:, 0], weights[0], attn_metadata=attn_metadata)


class DeepseekV32Attention(DeepseekV2Attention):

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 all_reduce: bool = True):
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
            )
        else:
            self.fused_qkv_a_proj = build_merged_colwise_linear(
                self.hidden_size,
                [config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim],
                bias=config.attention_bias,
                dtype=dtype,
                device=device,
                is_tp=False,
                quant_config=quantization_config,
                out_names=[0, 1],
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
            )

        if self.q_lora_rank is None:
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
                                  use_flash_mla=use_flash_mla,
                                  mla_index_topk=config.index_topk)

        self.vc = DeepseekV2BMM(self.num_heads, config.kv_lora_rank, self.v_head_dim, dtype=dtype, device=device)
        self.o_proj = build_o_proj(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
            all_reduce=all_reduce,
        )

        self.indexer = self._build_indexer(config, layer_idx, dtype, device)

    def _build_indexer(self, config: Any, layer_idx: int, dtype: torch.dtype, device: torch.device):
        return Indexer(config, layer_idx, dtype=dtype, device=device)

    def _q_proj(self, q_a_states, num_heads: int, nope_size: int, pe_size: int):
        """Q proj."""
        q_len = q_a_states.size(1)

        query_states = q_a_states.new_empty(q_len, num_heads, nope_size + pe_size)

        if self.q_lora_rank is None:
            qr = q_a_states
            q = self.q_proj(q_a_states)
        else:
            qr = self.q_a_layernorm(q_a_states)
            q = self.q_b_proj(qr)
        q = q.view(q_len, num_heads, self.q_head_dim)
        # q_pe: (q_len, num_heads, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope: (q_len, num_heads, kv_lora_rank)
        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe, qr

    def _kv_proj(self, key_states, nope_size: int):
        """Kv proj."""
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
        if self.q_lora_rank is None:
            q_a_states = hidden_states
            key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
        else:
            q_a_states, key_states = self.fused_qkv_a_proj(hidden_states).split(
                [self.q_lora_rank, nope_size + pe_size], dim=-1)
            key_states = key_states[0, :, None]

        query_states, q_pe, qr = self._q_proj(q_a_states, num_heads, nope_size, pe_size)
        key_states, value_states, k_pe = self._kv_proj(key_states, nope_size)

        return query_states, key_states, value_states, q_pe, k_pe, qr

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Sequence[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        num_heads = self.attn_fwd.num_heads
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

        topk_indices = self.indexer(hidden_states, qr, rotary_pos_emb, attn_metadata=attn_metadata)

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
    attention_cls = DeepseekV32Attention

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        quantization_config = None
        # Row-wise TP outputs normally reduce inside their projections. An
        # optimized communicator lets the following RMSNorm consume that
        # reduction instead. Attention is consumed in this layer.
        defer_attn_all_reduce = RMSNorm.can_handle_all_reduce('attn')
        # MLP is consumed by the next target layer, so terminal and MTP blocks
        # must still reduce their outputs.
        defer_mlp_all_reduce = (layer_idx < config.num_hidden_layers - 1
                                and RMSNorm.can_handle_all_reduce('mlp'))

        # build attention layer
        self.self_attn = self.attention_cls(
            config, layer_idx, dtype=dtype, device=device, all_reduce=not defer_attn_all_reduce)

        # mlp
        self.mlp = (DeepseekV2MoE(
            config, layer_idx, dtype=dtype, device=device, all_reduce=not defer_mlp_all_reduce) if
                    (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
                     and layer_idx % config.moe_layer_freq == 0) else DeepseekV2MLP(
                         config, dtype=dtype, device=device, all_reduce=not defer_mlp_all_reduce))

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=torch.float32,
                                       device=device,
                                       all_reduce_group='mlp')

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                dtype=torch.float32,
                                                device=device,
                                                all_reduce_group='attn')

class DeepseekV32Model(DeepseekV2Model):
    decoder_layer_cls = DeepseekV32DecoderLayer

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
            self.decoder_layer_cls(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=None,
                            dtype=torch.float32,
                            device=device)

        emb_type = RopeType.LinearScaling
        rope_dim = config.qk_rope_head_dim
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = get_rope_theta(config)

        rope_params = dict(emb_type=emb_type, dim=rope_dim, max_position_embeddings=rope_max_pos_emb, base=rope_base)
        update_params = build_rotary_params(config)
        rope_params.update(update_params)
        self.rotary_emb = build_rotary_embedding(**rope_params)


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    model_cls = DeepseekV32Model

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
        self.model = self.model_cls(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      bias=False,
                                      dtype=dtype,
                                      device=device)
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.get_input_embeddings())
        self._load_buffers = dict()

    def _load_weight_attention(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                               update_pe_mapping: list):
        if _load_fused_indexer_weight(name, loaded_weight, params_dict, self._load_buffers):
            return
        if _load_fused_qkv_a_weight(name, loaded_weight, params_dict, self.config):
            return
        return super()._load_weight_attention(name, loaded_weight, params_dict, update_pe_mapping)
