# Copyright (c) OpenMMLab. All rights reserved.
"""EAGLE3.1 draft model for Kimi-K2.x multi-latent attention."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_group, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (
    ParallelEmbedding,
    RMSNorm,
    RopeType,
    build_rotary_embedding,
    build_rotary_params,
)
from lmdeploy.pytorch.nn.linear import (
    build_colwise_linear,
    build_merged_colwise_linear,
)
from lmdeploy.pytorch.nn.rotary_embedding import get_rope_theta
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .kimi_k2_language import KimiK2Attention, KimiK2MLP
from .patch import add_prefix
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


def _get_eagle_aux_layer_ids(config: PretrainedConfig) -> tuple[int, ...]:
    """Return target layer boundaries in checkpoint concatenation order."""
    eagle_config = getattr(config, 'eagle_config', None)
    if isinstance(eagle_config, dict):
        layer_ids = eagle_config.get('eagle_aux_hidden_state_layer_ids')
    else:
        layer_ids = getattr(eagle_config,
                            'eagle_aux_hidden_state_layer_ids', None)
    if layer_ids is None:
        layer_ids = getattr(config, 'eagle_aux_hidden_state_layer_ids', None)
    if layer_ids is None:
        return (2, 30, 58)
    layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
    if len(layer_ids) != 3 or len(set(layer_ids)) != 3:
        raise ValueError(
            'EAGLE3 auxiliary layer ids must contain exactly three unique '
            'layer ids')
    return layer_ids


def _reorder_rope_weight(weight: torch.Tensor, head_dim: int,
                         pe_dim_offset: int) -> torch.Tensor:
    """Convert checkpoint interleaved RoPE rows to LMDeploy's layout."""
    if (head_dim <= pe_dim_offset
            or (head_dim - pe_dim_offset) % 2 != 0
            or weight.shape[0] % head_dim != 0):
        raise ValueError(
            f'invalid RoPE weight shape={tuple(weight.shape)}, '
            f'head_dim={head_dim}, pe_dim_offset={pe_dim_offset}')
    weight = weight.unflatten(0, (-1, head_dim)).clone()
    rope_weight = weight[:, pe_dim_offset:]
    rope_weight = rope_weight.unflatten(1, (-1, 2)).transpose(1, 2)
    weight[:, pe_dim_offset:] = rope_weight.flatten(1, 2)
    return weight.flatten(0, 1)


def _split_kv_b_weight(weight: torch.Tensor, qk_nope_head_dim: int,
                       v_head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the checkpoint MLA B projection into absorbed K/V matrices."""
    head_width = qk_nope_head_dim + v_head_dim
    if weight.shape[0] % head_width != 0:
        raise ValueError(
            f'kv_b output size {weight.shape[0]} is not divisible by '
            f'{head_width}')
    w_kc, w_vc = weight.unflatten(0, (-1, head_width)).split(
        [qk_nope_head_dim, v_head_dim], dim=1)
    return w_kc, w_vc.transpose(1, 2).contiguous()


def _all_gather_last_dim(hidden_states: torch.Tensor,
                         layer_type: str = 'attn') -> torch.Tensor:
    """Gather a column-parallel projection in rank order."""
    tp, _ = get_tp_world_rank(layer_type)
    if tp == 1:
        return hidden_states
    local = hidden_states.movedim(-1, 0).contiguous()
    gathered = local.new_empty((local.shape[0] * tp, *local.shape[1:]))
    group = get_dist_group(layer_type).gpu_group
    dist.all_gather_into_tensor(gathered, local, group=group)
    return gathered.movedim(0, -1).contiguous()


def _identity_token_map(vocab_size: int,
                        device: torch.device | str = None) -> torch.Tensor:
    """Build the no-d2t full-vocabulary mapping used by Kimi EAGLE3.1."""
    return torch.arange(vocab_size, dtype=torch.long, device=device)


class Eagle3MLAAttention(KimiK2Attention):
    """DeepSeek MLA with the EAGLE3 doubled pre-attention input width."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        if config.q_lora_rank is None:
            raise ValueError('Kimi EAGLE3 MLA requires q_lora_rank')

        # The published draft is BF16 even when its target is quantized.
        config.fuse_qkv_a_proj = False
        super().__init__(config, dtype=dtype, device=device, prefix=prefix)
        for module_name in ('q_a_proj', 'kv_a_proj_with_mqa',
                            'fused_qkv_a_proj_with_mqa'):
            if hasattr(self, module_name):
                delattr(self, module_name)
        self.fused_qkv_a_proj_with_mqa = build_merged_colwise_linear(
            2 * config.hidden_size,
            [config.q_lora_rank,
             config.kv_lora_rank + config.qk_rope_head_dim],
            bias=getattr(config, 'attention_bias', False),
            dtype=dtype,
            device=device,
            is_tp=False,
            out_names=['q', 'kv'],
            quant_config=None,
            check_dist=False,
            layer_type='attn',
            prefix=add_prefix('fused_qkv_a_proj_with_mqa', prefix),
        )
        # DeepseekV2Attention defaults these two norms to 1e-6.  This draft
        # checkpoint was trained with the model-wide epsilon (1e-5 for K2.6).
        self.q_a_layernorm = RMSNorm(
            config.q_lora_rank,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('q_a_layernorm', prefix),
        )
        self.kv_a_layernorm = RMSNorm(
            config.kv_lora_rank,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('kv_a_layernorm', prefix),
        )
        self.fuse_qkv_a_proj = True


class Eagle3MLADecoderLayer(nn.Module):
    """One dense EAGLE3 decoder layer using Kimi's MLA cache layout."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int = 0,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Eagle3MLAAttention(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('self_attn', prefix),
        )
        self.mlp = KimiK2MLP(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('mlp', prefix),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('input_layernorm', prefix),
        )
        self.hidden_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('hidden_norm', prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('post_attention_layernorm', prefix),
        )

    def forward(
        self,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: list[torch.FloatTensor] | None,
        attn_metadata: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        attention_input = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(
            hidden_states=attention_input,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Eagle3MLAModel(nn.Module):
    """Kimi EAGLE3.1 draft backbone."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=dtype,
            device=device,
            is_tp=True,
        )

        self.aux_layer_ids = _get_eagle_aux_layer_ids(config)
        self.num_aux_hidden_states = len(self.aux_layer_ids)
        self.target_hidden_size = getattr(config, 'target_hidden_size',
                                          config.hidden_size)
        self.fc = build_colwise_linear(
            self.target_hidden_size * self.num_aux_hidden_states,
            config.hidden_size,
            bias=getattr(config, 'bias', False),
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=None,
            layer_type='attn',
            prefix=add_prefix('fc', prefix),
        )

        if getattr(config, 'fc_norm', False) or getattr(
                config, 'use_aux_norm', False):
            self.fc_norm = nn.ModuleList([
                RMSNorm(
                    self.target_hidden_size,
                    config.rms_norm_eps,
                    dtype=dtype,
                    device=device,
                    prefix=add_prefix(f'fc_norm.{idx}', prefix),
                ) for idx in range(self.num_aux_hidden_states)
            ])
        else:
            self.fc_norm = None

        if config.num_hidden_layers != 1:
            raise ValueError('Kimi EAGLE3 requires exactly one draft layer')
        self.midlayer = Eagle3MLADecoderLayer(
            config,
            layer_idx=0,
            dtype=dtype,
            device=device,
            prefix=add_prefix('midlayer', prefix),
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('norm', prefix),
        )
        self.norm_output = bool(getattr(config, 'norm_output', False))

        rope_params = dict(
            emb_type=RopeType.LinearScaling,
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=get_rope_theta(config),
        )
        rope_params.update(build_rotary_params(config))
        self.rotary_emb = build_rotary_embedding(**rope_params)

    def _project_target_hidden_states(
            self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected_width = self.target_hidden_size * self.num_aux_hidden_states
        if hidden_states.shape[-1] == self.config.hidden_size:
            return hidden_states
        if hidden_states.shape[-1] != expected_width:
            raise ValueError(
                'Kimi EAGLE3 target hidden width must be either recurrent '
                f'{self.config.hidden_size} or auxiliary {expected_width}, got '
                f'{hidden_states.shape[-1]}')
        if self.fc_norm is not None:
            chunks = hidden_states.split(self.target_hidden_size, dim=-1)
            hidden_states = torch.cat([
                norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)
            ], dim=-1)
        hidden_states = self.fc(hidden_states)
        return _all_gather_last_dim(hidden_states, layer_type='attn')

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        previous_hidden_states: torch.FloatTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError('input_ids or inputs_embeds is required')
            inputs_embeds = self.embed_tokens(input_ids)
        if self.dtype is not None:
            inputs_embeds = inputs_embeds.to(self.dtype)
        if previous_hidden_states is None:
            raise ValueError('previous_hidden_states is required for EAGLE3')
        previous_hidden_states = previous_hidden_states.to(inputs_embeds)
        previous_hidden_states = self._project_target_hidden_states(
            previous_hidden_states)

        cos, sin = self.rotary_emb(previous_hidden_states, position_ids)
        rotary_pos_emb = (cos[0], sin[0])
        hidden_states, residual = self.midlayer(
            inputs_embeds,
            previous_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_values[0],
            attn_metadata=attn_metadata,
        )
        hidden_states, hidden_states_prenorm = self.norm(
            hidden_states, residual)
        draft_aux_hidden_states = (
            hidden_states if self.norm_output else hidden_states_prenorm)
        return dict(
            hidden_states=hidden_states,
            hidden_states_prenorm=hidden_states_prenorm,
            draft_aux_hidden_states=draft_aux_hidden_states,
        )

    def get_input_embeddings(self):
        """Get token embeddings."""
        return self.embed_tokens


class Eagle3DeepseekV2ForCausalLM(nn.Module, CudaGraphMixin):
    """Kimi-K2.x EAGLE3.1 causal draft model with MLA attention."""

    packed_modules_mapping = {
        'fused_qkv_a_proj_with_mqa': [
            'q_a_proj',
            'kv_a_proj_with_mqa',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    @classmethod
    def update_quant_config(cls, _quant_config):
        """The published draft checkpoint is BF16, independent of target."""
        return None

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        nn.Module.__init__(self)
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype
        if not hasattr(config, 'attention_bias'):
            config.attention_bias = False
        self.model = Eagle3MLAModel(
            config, dtype=dtype, device=device, prefix='model')

        draft_vocab_size = getattr(config, 'draft_vocab_size',
                                   config.vocab_size)
        config.draft_vocab_size = draft_vocab_size
        self.lm_head = build_colwise_linear(
            config.hidden_size,
            draft_vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=None,
            layer_type='attn',
            prefix='lm_head',
        )
        self.draft_id_to_target_id = nn.Parameter(
            _identity_token_map(draft_vocab_size, device=device),
            requires_grad=False,
        )
        # Match SGLang: always share the target embedding for this checkpoint.
        self.include_embed_tokens = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        target_hidden_states: torch.Tensor = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            previous_hidden_states=target_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ) -> dict[str, Any]:
        return dict(
            input_ids=context.input_ids,
            position_ids=context.position_ids,
            past_key_values=past_key_values,
            attn_metadata=context.attn_metadata,
            inputs_embeds=inputs_embeds,
            target_hidden_states=context.target_hidden_states,
        )

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute gathered full-vocabulary draft logits."""
        local_logits = self.lm_head(hidden_states)
        return _all_gather_last_dim(local_logits, layer_type='attn')

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make CUDA graph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens
        target_hidden_states = kwargs.get('target_hidden_states')
        if target_hidden_states is None:
            raise ValueError(
                'target_hidden_states is required for EAGLE3 graph capture')
        input_buffers = super().make_buffers_cudagraph(
            graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers[
            'input_ids'].new_zeros(
                1,
                max_tokens,
                target_hidden_states.size(-1),
                dtype=self.dtype,
            )
        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Fill CUDA graph buffers from forward inputs."""
        new_inputs = super().fill_buffers_cudagraph(
            graph_meta=graph_meta, **kwargs)
        num_tokens = kwargs['input_ids'].size(-1)
        target_hidden_states = kwargs.get('target_hidden_states')
        if target_hidden_states is None:
            raise ValueError(
                'target_hidden_states is required for EAGLE3 graph replay')
        target_buffer = graph_meta.input_buffers['target_hidden_states']
        target_buffer[:, :num_tokens] = target_hidden_states
        new_inputs['target_hidden_states'] = target_buffer
        return new_inputs

    def get_outputs_cudagraph(self, output_buffers: dict[str, torch.Tensor],
                              input_ids: torch.Tensor,
                              **kwargs) -> dict[str, torch.Tensor]:
        """Slice valid outputs from CUDA graph buffers."""
        num_tokens = input_ids.size(-1)
        return {
            'hidden_states': output_buffers['hidden_states'][:, :num_tokens],
            'hidden_states_prenorm':
            output_buffers['hidden_states_prenorm'][:, :num_tokens],
            'draft_aux_hidden_states':
            output_buffers['draft_aux_hidden_states'][:, :num_tokens],
        }

    def get_input_embeddings(self):
        """Get the target-shared embedding module."""
        return self.model.get_input_embeddings()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load the published one-layer EAGLE3.1 MLA checkpoint."""
        stacked_params_mapping = (
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        )
        params_dict = dict(self.named_parameters())

        for checkpoint_name, loaded_weight in weights:
            name = checkpoint_name.removeprefix('model.')
            if name == 't2d' or name.endswith('.t2d'):
                continue
            if name == 'd2t' or name.endswith('.d2t'):
                if loaded_weight.numel() != self.config.draft_vocab_size:
                    raise ValueError(
                        f'd2t has {loaded_weight.numel()} entries, expected '
                        f'{self.config.draft_vocab_size}')
                token_ids = loaded_weight.to(torch.long) + torch.arange(
                    self.config.draft_vocab_size,
                    dtype=torch.long,
                    device=loaded_weight.device,
                )
                load_weight(self.draft_id_to_target_id, token_ids)
                continue
            if 'rotary_emb.' in name:
                continue
            if name == 'embed_tokens.weight':
                # The runtime embedding is deliberately shared with target.
                continue

            name = name.replace('layers.0.', 'midlayer.', 1)
            if not name.startswith('lm_head.'):
                name = f'model.{name}'

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                target_name = name.replace(weight_name, param_name)
                load_weight(
                    params_dict[target_name], loaded_weight, shard_id=shard_id)
                handled = True
                break
            if handled:
                continue

            if '.self_attn.q_a_proj.' in name:
                target_name = name.replace(
                    '.q_a_proj.', '.fused_qkv_a_proj_with_mqa.')
                load_weight(
                    params_dict[target_name], loaded_weight, shard_id='q')
                continue
            if '.self_attn.kv_a_proj_with_mqa.' in name:
                target_name = name.replace(
                    '.kv_a_proj_with_mqa.',
                    '.fused_qkv_a_proj_with_mqa.')
                if name.endswith('.weight'):
                    loaded_weight = _reorder_rope_weight(
                        loaded_weight,
                        self.config.kv_lora_rank +
                        self.config.qk_rope_head_dim,
                        self.config.kv_lora_rank,
                    )
                load_weight(
                    params_dict[target_name], loaded_weight, shard_id='kv')
                continue
            if '.self_attn.q_b_proj.' in name and name.endswith('.weight'):
                loaded_weight = _reorder_rope_weight(
                    loaded_weight,
                    self.config.qk_nope_head_dim +
                    self.config.qk_rope_head_dim,
                    self.config.qk_nope_head_dim,
                )
            if '.self_attn.kv_b_proj.' in name:
                if not name.endswith('.weight'):
                    raise ValueError(
                        f'unsupported quantized draft kv_b tensor '
                        f'{checkpoint_name}')
                w_kc, w_vc = _split_kv_b_weight(
                    loaded_weight,
                    self.config.qk_nope_head_dim,
                    self.config.v_head_dim,
                )
                load_weight(
                    params_dict[name.replace('.kv_b_proj.', '.kc.')], w_kc)
                load_weight(
                    params_dict[name.replace('.kv_b_proj.', '.vc.')], w_vc)
                continue

            try:
                param = params_dict[name]
            except KeyError:
                raise KeyError(
                    f'unexpected Kimi EAGLE3 checkpoint weight '
                    f'{checkpoint_name!r} mapped to {name!r}') from None
            load_weight(param, loaded_weight)
