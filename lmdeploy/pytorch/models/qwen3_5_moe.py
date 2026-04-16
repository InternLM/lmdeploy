# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import RMSNorm
from lmdeploy.pytorch.nn.moe import build_fused_moe
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import add_prefix, get_build_model_context
from .qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)
from .qwen3_5 import Qwen3_5VisionModel as Qwen3_5MoeVisionModel
from .qwen3_vl import Qwen3VLInputProcessor as Qwen3_5MoeInputProcessor


class Qwen3_5MoeTopKRouter(nn.Module):

    def __init__(self, config, dtype: torch.dtype | None = None, device: torch.device | None = None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim, dtype=dtype, device=device))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = '',
                 is_tp: bool = True):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = Qwen3_5MoeTopKRouter(config, dtype=dtype, device=device)

        # get all reduce flags
        dist_ctx = get_dist_manager().current_context()
        dp = dist_ctx.dist_config.dp
        world_size = dist_ctx.dist_config.moe_tp
        attn_tp = dist_ctx.dist_config.attn_tp or 1
        if dp == 1 and world_size > 1:
            self._all_reduce = True
        else:
            self._all_reduce = False
        shared_expert_all_reduce = (attn_tp > 1 and not self._all_reduce)

        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            renormalize=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            all_reduce=False,
            layer_idx=layer_idx,
            prefix=add_prefix('experts', prefix),
        )

        self.shared_expert = Qwen3_5MLP(
            config=config,
            intermediate_size=config.shared_expert_intermediate_size,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
            all_reduce=shared_expert_all_reduce,
            prefix=add_prefix('shared_expert', prefix),
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, all_routed_experts: torch.Tensor | None = None):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits, topk_weights, topk_ids = self.gate(hidden_states)
        if all_routed_experts is not None:
            all_routed_experts[:, self.layer_idx, :] = topk_ids
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


class Qwen3_5MoeDecoderLayer(Qwen3_5DecoderLayer):
    """Decoder layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = Qwen3_5GatedDeltaNet(config,
                                                    layer_idx,
                                                    dtype=dtype,
                                                    device=device,
                                                    prefix=add_prefix('linear_attn', prefix))
        elif self.layer_type == 'full_attention':
            self.self_attn = Qwen3_5Attention(config,
                                              layer_idx,
                                              dtype=dtype,
                                              device=device,
                                              prefix=add_prefix('self_attn', prefix))

        # build MLP
        self.mlp = Qwen3_5MoeSparseMoeBlock(config,
                                            layer_idx,
                                            dtype=dtype,
                                            device=device,
                                            prefix=add_prefix('mlp', prefix))

        # build input layer norm
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('input_layernorm', prefix),
        )

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)


class Qwen3_5MoeTextModel(Qwen3_5TextModel):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        nn.Module.__init__(self)
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
        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoderLayer(config,
                                   layer_idx,
                                   dtype=dtype,
                                   device=device,
                                   prefix=add_prefix(f'layers.{layer_idx}', prefix))
            for layer_idx in range(self.config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=device)


class Qwen3_5MoeModel(Qwen3_5Model):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        nn.Module.__init__(self)

        self.visual = Qwen3_5MoeVisionModel(config.vision_config,
                                            dtype=dtype,
                                            device=device,
                                            prefix=add_prefix('visual', prefix))
        self.language_model = Qwen3_5MoeTextModel(config.text_config,
                                                  dtype=dtype,
                                                  device=device,
                                                  prefix=add_prefix('language_model', prefix))


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
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
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        nn.Module.__init__(self)
        self.config = config
        self.ctx_mgr = ctx_mgr

        # build preprocessor
        self.input_processor = Qwen3_5MoeInputProcessor(self.config)

        # build model
        self.model = Qwen3_5MoeModel(config, dtype=dtype, device=device, prefix=add_prefix('model', prefix))
        # build lm_head
        self.lm_head = self.build_lm_head(config.text_config.hidden_size,
                                          config.text_config.vocab_size,
                                          bias=False,
                                          dtype=dtype,
                                          device=device)
        # for router replay
        bm_ctx = get_build_model_context()
        self.enable_return_routed_experts = bm_ctx.enable_return_routed_experts
        self.is_spec_decoding = get_build_model_context().num_spec_tokens > 0

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter]):
        """Load weight experts."""
        import re
        # fused weights (bf16): experts.gate_up_proj / experts.down_proj
        if any(k in name for k in ['experts.gate_up_proj', 'experts.down_proj']):
            return self._load_weight_fused_experts(name, loaded_weight, params_dict)

        # non-fused weights (fp8): experts.<id>.<gate|up|down>_proj.*
        proj_map = {
            'gate_proj': ('.experts.gate_up', 'gate'),
            'up_proj': ('.experts.gate_up', 'up'),
            'down_proj': ('.experts.down', 'down'),
        }
        m = re.search(r'\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)', name)
        if m:
            expert_id = int(m.group(1))
            param_name, shard_id = proj_map[m.group(2)]
            # e.g. .experts.42.gate_proj.weight -> .experts.gate_up.weight
            suffix = name[m.end():]
            param_key = name[:m.start()] + param_name + suffix
            param = params_dict[param_key]
            load_weight(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
        else:
            param = params_dict[name]
            load_weight(param, loaded_weight)

    def _load_weight_fused_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter]):
        """Load weight of fused expert weights."""
        num_experts = self.config.text_config.num_experts
        fused_gateup_name = 'gate_up_proj'
        fused_down_name = 'down_proj'
        if fused_gateup_name in name:

            for expert_id in range(num_experts):
                param_name = name.replace(f'experts.{fused_gateup_name}', 'experts.gate_up.weight')
                param = params_dict[param_name]
                weight = loaded_weight[expert_id]
                w1, w3 = weight.chunk(2, 0)
                load_weight(param, w1, expert_id=expert_id, shard_id='gate')
                load_weight(param, w3, expert_id=expert_id, shard_id='up')

        elif fused_down_name in name:

            for expert_id in range(num_experts):
                param_name = name.replace(f'experts.{fused_down_name}', 'experts.down.weight')
                param = params_dict[param_name]
                w2 = loaded_weight[expert_id]
                load_weight(param, w2, expert_id=expert_id, shard_id='down')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights."""

        def __skip_layers(name):
            """We might change the number of layers so we can debug the model
            with less gpus."""
            import re
            if '.layers.' not in name:
                return False
            matches = re.findall(r'\.layers\.(\d+)\.', name)
            layer_id = int(matches[0])
            return layer_id >= self.config.text_config.num_hidden_layers

        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
            ('.in_proj_ba', '.in_proj_b', 'b'),
            ('.in_proj_ba', '.in_proj_a', 'a'),
        ]

        rms_norm_keys = ['model.norm', '.input_layernorm', '.post_attention_layernorm', '.q_norm', '.k_norm']

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            if __skip_layers(name):
                continue

            if 'mtp.' in name:
                continue
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            if '.experts' in name and '.shared_expert' not in name:
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
                    if '.qkv.' in name:
                        # vl attention
                        param = params_dict[name]
                        q, k, v = param.weight_spliter(loaded_weight)
                        load_weight(param, q, shard_id='q')
                        load_weight(param, k, shard_id='k')
                        load_weight(param, v, shard_id='v')
                    else:
                        for rms_norm_key in rms_norm_keys:
                            if rms_norm_key in name and 'weight' in name:
                                loaded_weight = loaded_weight + 1
                                break
                        param = params_dict[name]
                        load_weight(param, loaded_weight)
