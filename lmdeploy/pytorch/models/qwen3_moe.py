# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.eplb import EPLBManager
from lmdeploy.pytorch.nn.linear import build_merged_colwise_linear, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.nn.moe import SoftmaxTopK, build_fused_moe
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class Qwen3MoeAttention(nn.Module):
    """Rewrite module of Qwen3MoeAttention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)

        # packed qkv
        # Qwen3 uses 'config.attention_bias = False' for q/k/o projections
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
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
            sliding_window=getattr(config, 'sliding_window', None),
        )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
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

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3MoeMLP(nn.Module):
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


class Qwen3MoeSparseMoeBlock(nn.Module):
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

        world_size, _ = get_tp_world_rank()
        _all_reduce = world_size > 1
        if get_dist_manager().current_context().dist_config.enable_eplb:
            dist_ctx = get_dist_manager().current_context()
            self.eplb_dispatch_info = EPLBManager.get_dispatch_info(
                ep_rank=dist_ctx.ep_rank,
                layer_idx=layer_idx,
            )
            self.num_experts = EPLBManager.num_physical_experts()
        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            renormalize=self.renormalize,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            all_reduce=_all_reduce,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self.softmax_topk(router_logits)
        if get_dist_manager().current_context().dist_config.enable_eplb:
            topk_ids = EPLBManager.topk_ids_logical_to_physical(topk_ids, self.eplb_dispatch_info)
        out_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        out_states = out_states.reshape(batch_size, sequence_length, -1)
        return out_states


class Qwen3MoeDecoderLayer(nn.Module):
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
        self.self_attn = Qwen3MoeAttention(config, dtype=dtype, device=device)

        # build MLP
        if (layer_idx not in config.mlp_only_layers) and (config.num_experts
                                                          > 0) and ((layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx=layer_idx, dtype=dtype, device=device)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size, dtype=dtype, device=device)

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
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
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


class Qwen3MoeModel(nn.Module):
    """model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        if get_dist_manager().current_context().dist_config.enable_eplb:
            ep_size, _ = get_ep_world_rank()
            EPLBManager.init_global_eplb_metadata(
                ep_size=ep_size,
                num_routed_experts=config.num_experts,
                num_hidden_layers=config.num_hidden_layers,
            )

        # build all decode layers
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = self._build_rotary_embedding(config)

    def _build_rotary_embedding(self, config: PretrainedConfig):
        """Build rotary embedding."""
        return build_rotary_embedding_from_config(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
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
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class Qwen3MoeForCausalLM(nn.Module, CudaGraphMixin):
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
        self.model = Qwen3MoeModel(config, dtype=dtype, device=device)
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
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                             expert_params_mapping: List):
        """Load weight experts."""
        # load fused weights
        if any([k in name for k in ['fused_w1w3', 'fused_w2']]):
            return self._load_weight_fused_experts(name, loaded_weight, params_dict)

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

    def _load_weight_fused_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter]):
        """Load weight of fused expert weights."""
        num_experts = self.config.num_experts
        fused_gateup_name = 'fused_w1w3'
        fused_down_name = 'fused_w2'
        if fused_gateup_name in name:
            chunk_size = loaded_weight.shape[0] // num_experts

            for expert_id in range(num_experts):
                param_name = name.replace(f'experts.{fused_gateup_name}', 'experts.gate_up')
                param = params_dict[param_name]
                w1 = loaded_weight.narrow(dim=0, start=chunk_size * expert_id, length=chunk_size // 2)
                w3 = loaded_weight.narrow(dim=0, start=chunk_size * expert_id + chunk_size // 2, length=chunk_size // 2)
                load_weight(param, w1, expert_id=expert_id, shard_id='gate')
                load_weight(param, w3, expert_id=expert_id, shard_id='up')

        elif fused_down_name in name:
            chunk_size = loaded_weight.shape[0] // num_experts

            for expert_id in range(num_experts):
                param_name = name.replace(f'experts.{fused_down_name}', 'experts.down')
                param = params_dict[param_name]
                w2 = loaded_weight.narrow(dim=0, start=chunk_size * expert_id, length=chunk_size)
                load_weight(param, w2, expert_id=expert_id, shard_id='down')

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
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            name = name.replace('.block_sparse_moe.', '.mlp.')
            if '.experts' in name:
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
