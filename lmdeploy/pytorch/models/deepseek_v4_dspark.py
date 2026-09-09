# Copyright (c) OpenMMLab. All rights reserved.
"""Bundled DeepSeek-V4 DSpark draft stages."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.compressor import V4CompressorMetadata
from lmdeploy.pytorch.backends.rotary_embedding import RopeType
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, rms_scale
from lmdeploy.pytorch.nn.linear import build_colwise_linear
from lmdeploy.pytorch.nn.rotary_embedding import build_rotary_embedding
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.utils import get_logger

from .deepseek_v4 import (
    Block,
    DeepseekV4ForCausalLM,
    V4Args,
    V4Caches,
    _set_default_dtype,
)
from .dspark_heads import build_dspark_head, compute_dspark_proposal_ids
from .patch import get_build_model_context
from .utils.cudagraph import CudaGraphMixin

logger = get_logger('lmdeploy')


class DeepseekV4ForCausalLMDSpark(nn.Module, CudaGraphMixin):
    """The checkpoint's ``mtp.*`` V4 stages used as a DSpark draft model."""

    def __init__(self,
                 config,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype or torch.bfloat16
        self.device = torch.device(device) if isinstance(device, str) else device
        self.quantization_config = getattr(config, 'quantization_config', None)
        target_layer_ids = tuple(
            get_build_model_context().spec_model_ctx.target_aux_hidden_state_layers)
        if not target_layer_ids:
            raise ValueError('Bundled DeepSeek-V4 DSpark requires target auxiliary layers.')
        self.target_layer_ids = target_layer_ids
        self.num_stages = len(target_layer_ids)
        self.dspark_sample_from_anchor = bool(
            getattr(config, 'dspark_sample_from_anchor'))
        self.dspark_draft_query_len = int(
            getattr(config, 'dspark_draft_query_len'))
        self.dspark_num_speculative_tokens = int(
            getattr(config, 'dspark_num_speculative_tokens'))

        compress_rope_params = config.rope_parameters['compress']
        self.args = V4Args(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_layers=self.num_stages,
            # Bundled DSpark stage gates carry learned weights/biases rather
            # than target hash-routing tables.
            mlp_layer_types=tuple('moe' for _ in range(self.num_stages)),
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            score_func=config.scoring_func,
            route_scale=config.routed_scaling_factor,
            swiglu_limit=config.swiglu_limit,
            q_lora_rank=config.q_lora_rank,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            norm_eps=config.rms_norm_eps,
            o_groups=config.o_groups,
            o_lora_rank=config.o_lora_rank,
            window_size=config.sliding_window,
            ring_storage_capacity=getattr(config, 'v4_ring_storage_capacity',
                                          config.sliding_window),
            compress_ratios=tuple(0 for _ in range(self.num_stages)),
            compress_rope_theta=config.compress_rope_theta,
            original_seq_len=compress_rope_params['original_max_position_embeddings'],
            rope_theta=config.rope_theta,
            rope_factor=compress_rope_params['factor'],
            beta_fast=compress_rope_params['beta_fast'],
            beta_slow=compress_rope_params['beta_slow'],
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
        )
        self.layers = nn.ModuleList([
            Block(config, idx, self.args, dtype=self.dtype, device=self.device)
            for idx in range(self.num_stages)
        ])
        self.main_proj = build_colwise_linear(
            config.hidden_size * len(target_layer_ids),
            config.hidden_size,
            bias=False,
            dtype=self.dtype,
            device=self.device,
            is_tp=False,
            quant_config=self.quantization_config,
            check_dist=False,
        )
        self.main_norm = RMSNorm(config.hidden_size,
                                 config.rms_norm_eps,
                                 dtype=self.dtype,
                                 device=self.device)
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            dtype=self.dtype,
                            device=self.device)
        hc_dim = config.hc_mult * config.hidden_size
        with _set_default_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(
                torch.empty(config.hc_mult, hc_dim, device=self.device),
                requires_grad=False)
            self.hc_head_base = nn.Parameter(
                torch.empty(config.hc_mult, device=self.device),
                requires_grad=False)
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, device=self.device), requires_grad=False)

        self.rotary_emb_plain = build_rotary_embedding(
            dim=self.args.rope_head_dim,
            max_position_embeddings=self.args.original_seq_len,
            base=self.args.rope_theta,
            emb_type=RopeType.Default,
            device=self.device,
        )
        self.markov_head = build_dspark_head(config,
                                             dtype=self.dtype,
                                             device=self.device)
        self.embed_tokens: nn.Module | None = None
        self.output_head: nn.Module | None = None
        self._load_buffers = {}

    def set_input_embeddings(self, embed_tokens: nn.Module):
        self.embed_tokens = embed_tokens

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_output_head(self, output_head: nn.Module):
        self.output_head = output_head

    def embed_input_ids(self, input_ids: torch.Tensor):
        if self.embed_tokens is None:
            raise RuntimeError('DeepSeek-V4 DSpark requires target-shared embeddings.')
        hidden = self.embed_tokens(input_ids)
        return hidden.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)

    def _runtime(self, h, position_ids, attn_metadata, state_ids):
        context = self.ctx_mgr.current_context()
        safe_state_ids = state_ids.to(torch.long)
        v4_meta_cls = get_backend().get_v4_attention_metadata_cls()
        v4_meta = v4_meta_cls.from_step_context(
            attn_metadata, context, window_size=self.args.window_size,
            ring_storage_capacity=self.args.ring_storage_capacity,
            slot=safe_state_ids, causal=False)
        v4_indexer_meta = v4_meta.build_indexer_metadata()
        v4_compressor_meta = V4CompressorMetadata(
            cu_q_seqlens=v4_meta.cu_q_seqlens,
            kv_seqlens=v4_meta.kv_seqlens,
            block_offsets=v4_meta.block_offsets,
            block_size=v4_meta.block_size,
            max_q_seqlen=v4_meta.max_q_seqlen,
        )
        caches = V4Caches(context.named_state_caches, context.block_caches)
        rd = self.args.rope_head_dim
        cos, sin = self.rotary_emb_plain(h, position_ids)
        cos = cos[0, :, :rd // 2]
        sin = sin[0, :, :rd // 2]
        return (v4_meta, v4_indexer_meta, v4_compressor_meta, caches,
                (cos, sin, -sin), safe_state_ids)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list | None = None,
                attn_metadata=None,
                inputs_embeds: torch.Tensor | None = None,
                state_ids: torch.Tensor | None = None,
                **kwargs):
        del past_key_values, kwargs
        if state_ids is None:
            raise RuntimeError('DeepSeek-V4 DSpark requires state_ids.')
        h = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        runtime = self._runtime(h, position_ids, attn_metadata, state_ids)
        v4_meta, index_meta, compressor_meta, caches, rotary, safe_ids = runtime
        for layer in self.layers:
            h = layer(h, v4_meta, index_meta, compressor_meta, input_ids,
                      safe_ids, caches, rotary_pos_emb=rotary,
                      compress_pos_emb=None)
        if not attn_metadata.is_decoding:
            return h
        draft_token_ids = compute_dspark_proposal_ids(self, h, input_ids)
        return dict(hidden_states=h, draft_token_ids=draft_token_ids)

    def project_target_hidden(self, target_hidden: torch.Tensor):
        expected = self.config.hidden_size * len(self.target_layer_ids)
        if target_hidden.ndim != 2 or target_hidden.size(-1) != expected:
            raise ValueError('DeepSeek-V4 DSpark target hidden width mismatch: '
                             f'expected {expected}, got {tuple(target_hidden.shape)}.')
        return self.main_norm(self.main_proj(target_hidden))

    @torch.inference_mode()
    def precompute_and_store_context_kv(self,
                                        target_hidden: torch.Tensor,
                                        position_ids: torch.Tensor,
                                        past_key_values=None,
                                        attn_metadata=None,
                                        max_q_seqlen: int | None = None):
        """Materialize target-derived K/V into every DSpark stage cache.

        The initial correctness path reuses the V4 attention operator to write its FP8 ring cache and discards the
        attention output.  This is more expensive than a dedicated KV-only kernel but preserves the exact cache format;
        optimize only after E2E parity is established.
        """
        del past_key_values, max_q_seqlen
        context = self.ctx_mgr.current_context()
        state_ids = context.state_offsets
        if state_ids is None:
            raise RuntimeError('DeepSeek-V4 DSpark context materialization requires state offsets.')
        main_x = self.project_target_hidden(target_hidden).unsqueeze(0)
        position_ids = position_ids.reshape(1, -1)
        runtime = self._runtime(main_x, position_ids, attn_metadata, state_ids)
        v4_meta, _, _, caches, rotary, safe_ids = runtime
        # Only K/V side effects matter.  The returned attention output is
        # deliberately discarded.
        for layer in self.layers:
            layer.attn(main_x, v4_meta, v4_meta.build_indexer_metadata(),
                       V4CompressorMetadata(cu_q_seqlens=v4_meta.cu_q_seqlens,
                                            kv_seqlens=v4_meta.kv_seqlens,
                                            block_offsets=v4_meta.block_offsets,
                                            block_size=v4_meta.block_size,
                                            max_q_seqlen=v4_meta.max_q_seqlen),
                       safe_ids, caches, rotary_pos_emb=rotary,
                       compress_pos_emb=None)

    def _collapse_hc(self, h: torch.Tensor):
        shape, dtype = h.size(), h.dtype
        flat = h.flatten(2).float()
        mixes = rms_scale(torch.nn.functional.linear(flat, self.hc_head_fn),
                          flat, eps=self.config.rms_norm_eps)
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base)
        pre = pre + self.config.hc_eps
        return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(dtype)

    def compute_base_logits(self, hidden_states: torch.Tensor):
        if self.output_head is None:
            raise RuntimeError('DeepSeek-V4 DSpark requires the target output head.')
        hidden_states = self.norm(self._collapse_hc(hidden_states))
        return self.output_head(hidden_states)

    def init_sequential_state(self, batch_size: int, dtype: torch.dtype,
                              device: torch.device):
        return self.markov_head.init_state(batch_size, dtype, device)

    def sequential_head_step(self, prev_token_ids: torch.Tensor,
                             hidden_states: torch.Tensor,
                             state: torch.Tensor | None):
        # Gated/RNN checkpoints consume the ordinary post-hc hidden state.
        if hidden_states.ndim == 3:
            hidden_states = self._collapse_hc(hidden_states.unsqueeze(1)).squeeze(1)
        return self.markov_head.step(prev_token_ids, hidden_states, state)

    @staticmethod
    def map_draft_to_target(draft_ids: torch.Tensor):
        return draft_ids

    def get_outputs_cudagraph(self, output_buffers: dict[str, torch.Tensor],
                              input_ids: torch.Tensor, **kwargs):
        """Return only live proposal rows from the captured batch bucket."""
        outputs = super().get_outputs_cudagraph(
            output_buffers, input_ids, **kwargs)
        draft_token_ids = output_buffers.get('draft_token_ids')
        if draft_token_ids is not None:
            batch_size = input_ids.numel() // self.dspark_draft_query_len
            outputs['draft_token_ids'] = draft_token_ids[:batch_size]
        return outputs

    def prepare_inputs_for_generation(self,
                                      past_key_values,
                                      inputs_embeds=None,
                                      context: StepContext = None):
        return dict(input_ids=context.input_ids,
                    position_ids=context.position_ids,
                    past_key_values=past_key_values,
                    attn_metadata=context.attn_metadata,
                    inputs_embeds=inputs_embeds,
                    state_ids=context.state_offsets)

    def update_model_metas(self, past_key_values, inputs_embeds=None,
                           context: StepContext = None):
        return None

    # Reuse the target's quantized V4 attention/expert loading primitives.
    _load_weights_attn = DeepseekV4ForCausalLM._load_weights_attn
    _load_expert = DeepseekV4ForCausalLM._load_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith('mtp.'):
                continue
            parts = name.split('.', 2)
            if len(parts) != 3:
                continue
            stage_id = int(parts[1])
            if stage_id >= self.num_stages:
                continue
            rest = parts[2]
            if rest.startswith('confidence_head.'):
                continue
            if rest.startswith('markov_head.'):
                mapped = rest
            elif stage_id == 0 and rest.startswith(('main_proj.', 'main_norm.')):
                mapped = rest
            elif stage_id == self.num_stages - 1 and rest.startswith(
                    ('norm.', 'hc_head_')):
                mapped = rest
            else:
                mapped = f'layers.{stage_id}.{rest}'

            if mapped.startswith('main_proj.') and mapped.endswith('.scale'):
                mapped = mapped.replace('.scale', '.weight_scale_inv')
            if mapped.startswith('markov_head.'):
                if mapped in params:
                    load_weight(params[mapped], loaded_weight)
                continue
            if '.ffn.' in mapped:
                self._load_expert(mapped, loaded_weight, params)
                continue
            if '.attn.' in mapped:
                self._load_weights_attn(mapped, loaded_weight, params)
                continue
            if mapped.endswith('.scale'):
                mapped = mapped.replace('.scale', '.weight_scale_inv')
            if mapped not in params:
                logger.warning(f'Skip unknown DeepSeek-V4 DSpark weight: {name} -> {mapped}')
                continue
            load_weight(params[mapped], loaded_weight)
