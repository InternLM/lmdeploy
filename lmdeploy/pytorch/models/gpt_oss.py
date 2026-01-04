# Copyright (c) OpenMMLab. All rights reserved.

import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import build_o_proj, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.nn.moe import build_fused_moe
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import get_build_model_context
from .utils.cudagraph import CudaGraphMixin


class GptOssAttention(nn.Module):
    """attention."""

    def __init__(self,
                 config: PretrainedConfig,
                 attention_type: str,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.layer_idx = layer_idx
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        scaling = head_dim**-0.5

        self.qkv_proj = build_qkv_proj(hidden_size,
                                       num_q_heads=num_attention_heads,
                                       num_kv_heads=num_key_value_heads,
                                       head_size=head_dim,
                                       bias=config.attention_bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       num_replicate_kv_heads=num_replicate_kv_heads)

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        if attention_type == 'sliding_attention':
            sliding_window = config.sliding_window
        elif attention_type == 'full_attention':
            sliding_window = None
        else:
            raise ValueError(f'Unsupported attention type: {attention_type}')
        # attention
        self.attn_fwd = Attention(
            num_attention_heads,
            head_dim,
            scale=scaling,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
            sliding_window=sliding_window,
            learnable_sink=True,
        )

        # o_proj
        self.o_proj = build_o_proj(num_attention_heads * head_dim,
                                   hidden_size,
                                   bias=config.attention_bias,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   is_tp=True)

        # sinks
        self.sinks = self.build_sinks(config, device)

    @classmethod
    def build_sinks(cls, config: PretrainedConfig, device):
        """Build sinks."""
        from lmdeploy.pytorch.distributed import get_tp_world_rank
        world_size, _ = get_tp_world_rank()
        num_attention_heads = config.num_attention_heads
        assert num_attention_heads % world_size == 0, (
            f'num_attention_heads={num_attention_heads} should be divisible by TP={world_size}')
        num_attention_heads = num_attention_heads // world_size
        sinks = nn.Parameter(torch.empty(num_attention_heads, device=device))
        sinks.weight_loader = cls.weight_loader_sinks
        return sinks

    @classmethod
    def weight_loader_sinks(cls, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load weight of sinks."""
        from lmdeploy.pytorch.distributed import get_tp_world_rank
        world_size, rank = get_tp_world_rank()
        loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
        param.data.copy_(loaded_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

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
            s_aux=self.sinks,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class GateupAct:

    def __init__(self, limit: float = 7.0, alpha: float = 1.702):
        self.limit = limit
        self.alpha = alpha
        self._run: Callable = None

    def _impl(self, gateup: torch.Tensor) -> torch.Tensor:
        """Moe act."""
        gate, up = gateup.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1) * glu

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def build(limit: float, alpha: float):
        return GateupAct(limit, alpha)

    def _try_compile(self, gateup: torch.Tensor) -> Callable:
        try:
            run = torch.compile(self._impl, dynamic=True)
            run(gateup)
            self._run = run
        except Exception:
            self._run = self._impl

    def __call__(self, gateup: torch.Tensor) -> torch.Tensor:
        """Call the act function."""
        if self._run is None:
            self._try_compile(gateup)

        return self._run(gateup)


class GptOssExperts(nn.Module):
    """experts."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.layer_idx = layer_idx
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.top_k = config.num_experts_per_tok
        self.alpha = 1.702
        self.limit = 7.0
        self._gateup_act = GateupAct.build(self.limit, self.alpha)

        self.experts = build_fused_moe(
            self.hidden_size,
            self.expert_dim,
            self.num_experts,
            bias=True,
            top_k=self.top_k,
            renormalize=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            all_reduce=True,
            layer_idx=layer_idx,
            act_func=self._gateup_act,
        )

    def forward(self, hidden_states: torch.Tensor, router_indices, routing_weights) -> torch.Tensor:
        """forward."""
        batch_size, sequence_length, _ = hidden_states.shape
        out_states = self.experts(
            hidden_states[0],
            routing_weights,
            router_indices,
        )

        out_states = out_states.reshape(batch_size, sequence_length, -1)
        return out_states


class GptOssTopKRouter(nn.Module):
    """Gate + topk + softmax."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.empty(self.num_experts, dtype=dtype, device=device))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = router_top_value
        return router_scores, router_indices


class GptOssMLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.router = GptOssTopKRouter(config, dtype=dtype, device=device)
        self.experts = GptOssExperts(config, layer_idx, dtype=dtype, device=device)

    def forward(self, hidden_states, all_routed_experts: torch.Tensor = None):
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        if all_routed_experts is not None:
            all_routed_experts[:, self.layer_idx, :] = router_indices
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out


class GptOssDecoderLayer(nn.Module):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = GptOssAttention(config, self.attention_type, layer_idx=layer_idx, dtype=dtype, device=device)

        # build MLP
        self.mlp = GptOssMLP(config, layer_idx, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                quant_config=quantization_config,
                                                dtype=dtype,
                                                device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        all_routed_experts: torch.Tensor = None,
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
        hidden_states = self.mlp(hidden_states, all_routed_experts=all_routed_experts)

        outputs = (hidden_states, residual)
        return outputs


class GptOssModel(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         config.pad_token_id,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            GptOssDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = build_rotary_embedding_from_config(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        all_routed_experts: torch.Tensor = None,
    ):
        """Rewrite of forward."""

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
                all_routed_experts=all_routed_experts,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class GptOssForCausalLM(nn.Module, CudaGraphMixin):
    """ModelForCausalLM."""

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
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
        self.model = GptOssModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

        # for router replay
        bm_ctx = get_build_model_context()
        self.enable_return_routed_experts = bm_ctx.enable_return_routed_experts

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
        # router replay
        all_routed_experts = None
        if self.enable_return_routed_experts:
            if inputs_embeds is not None:
                num_tokens = inputs_embeds.size(1)
            else:
                num_tokens = input_ids.size(1)
            all_routed_experts = position_ids.new_empty(
                (num_tokens, self.config.num_hidden_layers, self.config.num_experts_per_tok), dtype=torch.uint16)

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            all_routed_experts=all_routed_experts,
        )

        if all_routed_experts is None:
            return hidden_states
        return dict(hidden_states=hidden_states, all_routed_experts=all_routed_experts)

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

    def update_weights(self):
        """Update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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

    def _load_weight_experts_gate_up(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str,
                                                                                                     nn.Parameter]):
        """Load weight of experts gate up."""
        num_experts = self.config.num_local_experts

        loaded_weight = loaded_weight.cuda()
        if 'gate_up_proj_bias' in name:
            param_name = name.replace('experts.gate_up_proj_bias', 'experts.experts.gate_up.bias')
        elif 'gate_up_proj' in name:
            param_name = name.replace('experts.gate_up_proj', 'experts.experts.gate_up.weight')
            loaded_weight = loaded_weight.transpose(1, 2)
        param = params_dict[param_name]
        for expert_id in range(num_experts):
            w1 = loaded_weight[expert_id, ::2]
            w3 = loaded_weight[expert_id, 1::2]
            load_weight(param, w1, expert_id=expert_id, shard_id='gate')
            load_weight(param, w3, expert_id=expert_id, shard_id='up')

    def _load_weight_experts_down(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter]):
        """Load weight of experts down."""
        num_experts = self.config.num_local_experts

        loaded_weight = loaded_weight.cuda()
        if 'down_proj_bias' in name:
            param_name = name.replace('experts.down_proj_bias', 'experts.experts.down.bias')
        elif 'down_proj' in name:
            param_name = name.replace('experts.down_proj', 'experts.experts.down.weight')
            loaded_weight = loaded_weight.transpose(1, 2)
        param = params_dict[param_name]
        for expert_id in range(num_experts):
            w2 = loaded_weight[expert_id]
            load_weight(param, w2, expert_id=expert_id, shard_id='down')

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter]):
        """Load weight of fused expert weights."""
        if 'gate_up' in name:
            self._load_weight_experts_gate_up(name, loaded_weight, params_dict)

        elif 'down' in name:
            self._load_weight_experts_down(name, loaded_weight, params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]

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
                self._load_weight_experts(name, loaded_weight, params_dict)
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
