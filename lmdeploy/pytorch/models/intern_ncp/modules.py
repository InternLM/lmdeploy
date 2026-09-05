# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.nn import (
    ApplyRotaryEmb,
    Attention,
    RMSNorm,
    SiluAndMul,
    build_rotary_embedding_from_config,
)
from lmdeploy.pytorch.nn.linear import (
    build_down_linear,
    build_gateup_linear,
    build_merged_colwise_linear,
    build_o_proj,
    build_qkv_proj,
)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from ..patch import add_prefix
from ..utils.model import build_embedding
from .weight import _repack_olmo_qkv_weight

_CONFIG_VALUE = object()
_HistoryStates = torch.Tensor
_SourceStates = torch.Tensor | None


def _get_configured_window(config: PretrainedConfig):
    """Return the reference OLMo window setting as an int or None."""
    window_size = getattr(config, 'window_size', None)
    if window_size is None:
        return None
    if isinstance(window_size, (list, tuple)):
        window_size = window_size[0]
    if window_size is None:
        return None
    window_size = int(window_size)
    return window_size if window_size > 0 else None


def _make_olmo_rotary_embedding(config: PretrainedConfig,
                                device: torch.device = None) -> nn.Module:
    """Build ConceptLM/OLMo rotary embedding from its native config fields."""
    rotary_interleaved = bool(getattr(config, 'rotary_interleaved', False))
    if rotary_interleaved:
        raise NotImplementedError('ConceptLM rotary_interleaved=True is not supported by the LMDeploy block yet.')

    return build_rotary_embedding_from_config(config, device=device)


def _qk_rmsnorm_variance(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Local Q/K squared sums before TP all-reduce."""
    query = query.float()
    key = key.float()
    query_var = (query * query).sum(-1, keepdim=True)
    key_var = (key * key).sum(-1, keepdim=True)
    return torch.stack([query_var, key_var], dim=0)


def _qk_rmsnorm_apply(query: torch.Tensor,
                      key: torch.Tensor,
                      variance: torch.Tensor,
                      query_weight: torch.Tensor,
                      key_weight: torch.Tensor,
                      hidden_size: int,
                      eps: float):
    """Apply whole-hidden Q/K RMSNorm from an already all-reduced variance."""
    dtype = query.dtype
    query_var, key_var = variance / hidden_size + eps
    query = (query.float() * torch.rsqrt(query_var)).to(dtype) * query_weight
    key = (key.float() * torch.rsqrt(key_var)).to(dtype) * key_weight
    return query, key


class Embedding(nn.Module):
    """Token embedding container.

    Mirrors ``self.embedding`` in the reference: a plain ``nn.Module`` holding
    a ``word_embeddings`` table. Keeping the attribute name ``word_embeddings``
    makes the checkpoint key ``embedding.word_embeddings.weight`` line up
    directly with ``self.embedding.word_embeddings``.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.word_embeddings = build_embedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, 'pad_token_id', None),
            dtype=dtype,
            device=device,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """forward."""
        return self.word_embeddings(input_ids)


class Quantizer(nn.Module):
    """Rewrite of ``_Quantizer``.

    The reference keeps codebooks as a ``ParameterList`` only to produce
    checkpoint keys ``codebook.0`` ... ``codebook.N``. Runtime computation only
    needs the stacked tensor returned by ``transformed_codebook()``.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.num_codebooks = int(config.concept_v22_vq_num_codebooks)
        self.codebook_size = int(config.concept_v22_vq_codebook_size)
        hidden_size = int(config.hidden_size)
        assert hidden_size % self.num_codebooks == 0, (
            f'hidden_size={hidden_size} must be divisible by num_codebooks={self.num_codebooks}')
        self.codebook_dim = hidden_size // self.num_codebooks
        self.hidden_size = hidden_size
        self.codebook = nn.Parameter(
            torch.empty(
                self.num_codebooks,
                self.codebook_size,
                self.codebook_dim,
                dtype=dtype,
                device=device,
            ),
            requires_grad=False,
        )

    def transformed_codebook(self):
        """Return codebook as ``[num_codebooks, codebook_size,
        codebook_dim]``."""
        return self.codebook

    def forward(self, concept_logits: torch.Tensor) -> torch.Tensor:
        """Convert per-codebook logits to hidden vectors.

        Args:
            concept_logits: ``[..., num_codebooks, codebook_size]``. In the
                LMDeploy engine the leading dims may be a packed continuous
                batching dimension, e.g. ``[num_concepts_total]``.

        Returns:
            Quantized/predicted vectors with shape ``[..., hidden_size]``.
        """
        codebook = self.transformed_codebook().to(concept_logits.dtype)
        vectors = torch.einsum('...hk,hkd->...hd', concept_logits, codebook)
        return vectors.flatten(-2, -1)


class DepthDD(nn.Module):
    """Rewrite of ``_DepthDD``.

    This is a small replicated per-token depth mixer. It does not mix tokens;
    it computes ``num_prev`` route weights from the current hidden state and
    combines the matching per-layer history states. The reference only handles
    dense ``[seq, batch, hidden]`` tensors because it stacks history at
    ``dim=2``. LMDeploy continuous batching commonly uses packed
    ``[num_tokens, hidden]`` tensors. Runtime passes preallocated tensor
    history to avoid repeated ``torch.stack`` copies.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 use_softmax: bool,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.num_prev = int(layer_idx) + 2
        route_hidden_size = self.num_prev
        self.w1 = nn.Linear(config.hidden_size, route_hidden_size, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(route_hidden_size, self.num_prev, bias=False, dtype=dtype, device=device)
        self.static_a = nn.Parameter(torch.zeros(self.num_prev, dtype=dtype, device=device), requires_grad=False)
        self.use_softmax = bool(use_softmax)
        for param in self.parameters():
            param.requires_grad_(False)

    def _history_tensor(self, hidden_states: torch.Tensor, history_states: _HistoryStates, history_dim: int):
        """Return history in shape ``[..., num_prev, hidden_size]``."""
        history_dim = history_dim if history_dim >= 0 else history_dim + history_states.dim()
        if history_dim != history_states.dim() - 2:
            history_states = history_states.movedim(history_dim, -2)
        return history_states

    def forward(self,
                hidden_states: torch.Tensor,
                history_states: _HistoryStates,
                history_dim: int = -2) -> torch.Tensor:
        """forward."""
        history = self._history_tensor(hidden_states, history_states, history_dim)
        weights = self.w2(F.gelu(self.w1(hidden_states)))
        weights = weights + self.static_a.to(dtype=weights.dtype)
        if self.use_softmax:
            weights = weights.softmax(dim=-1)
        return torch.einsum('...l,...lh->...h', weights, history)


class SelfDD(nn.Module):
    """Rewrite of ``_SelfDD``."""

    def __init__(self,
                 config: PretrainedConfig,
                 num_layers: int,
                 use_softmax: bool = False,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.num_layers = int(num_layers)
        self.depth_dds = nn.ModuleList([
            DepthDD(config, layer_idx, use_softmax, dtype=dtype, device=device)
            for layer_idx in range(self.num_layers)
        ])

    def make_history_buffer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Allocate layer-major history buffer ``[num_layers + 1,
        *hidden_shape]``."""
        return hidden_states.new_empty((self.num_layers + 1, *hidden_states.shape))

    @staticmethod
    def write_history(history_buffer: torch.Tensor, slot_idx: int, hidden_states: torch.Tensor):
        """Copy one history block into a layer-major history buffer."""
        history_buffer[int(slot_idx)].copy_(hidden_states)
        return history_buffer

    @staticmethod
    def history_view(history_buffer: torch.Tensor, layer_idx: int):
        """Return layer-major history needed by ``layer_idx`` without copying.

        This is CUDA-graph safe when ``layer_idx`` is a Python constant for the
        current layer and ``history_buffer`` has the fixed graph-capture shape.
        """
        return history_buffer[:int(layer_idx) + 2]

    def forward_from_buffer(self,
                            layer_idx: int,
                            hidden_states: torch.Tensor,
                            history_buffer: torch.Tensor):
        """Runtime path: fixed buffer, no list materialization."""
        layer_idx = int(layer_idx)
        return self.depth_dds[layer_idx](
            hidden_states,
            self.history_view(history_buffer, layer_idx),
            history_dim=0,
        )

    def forward(self,
                layer_idx: int,
                hidden_states: torch.Tensor,
                history_states: _HistoryStates):
        """forward."""
        return self.depth_dds[int(layer_idx)](hidden_states, history_states)


class ResidualRoute(nn.Module):
    """Rewrite of ``_ResidualRoute``.

    This module computes a source-state mixture and adds it as a gated residual update to the target hidden state. It is
    small and replicated.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 num_source_states: int,
                 use_softmax: bool = True,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.num_source_states = int(num_source_states)
        route_hidden_size = max(1, self.num_source_states)
        self.w1 = nn.Linear(config.hidden_size, route_hidden_size, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(route_hidden_size, self.num_source_states, bias=False, dtype=dtype, device=device)
        self.residual_diag = nn.Parameter(torch.zeros(config.hidden_size, dtype=dtype, device=device),
                                          requires_grad=False)
        self.use_softmax = bool(use_softmax)
        for param in self.parameters():
            param.requires_grad_(False)

    def _source_tensor(self,
                       target_hidden: torch.Tensor,
                       source_states: _SourceStates,
                       source_dim: int):
        """Return source states in shape ``[..., active_sources,
        hidden_size]``."""
        if source_states is None:
            return None

        source_dim = source_dim if source_dim >= 0 else source_dim + source_states.dim()
        if source_states.shape[source_dim] == 0:
            return None
        if source_dim != source_states.dim() - 2:
            source_states = source_states.movedim(source_dim, -2)
        return source_states

    def _route_weights(self, target_hidden: torch.Tensor, active_sources: int):
        """Compute route weights and keep the last active source logits."""
        weights = self.w2(F.gelu(self.w1(target_hidden)))
        weights = weights[..., -active_sources:]
        if self.use_softmax:
            weights = weights.softmax(dim=-1)
        return weights

    def _add_update(self,
                    target_hidden: torch.Tensor,
                    source_mix: torch.Tensor,
                    residual_scale: torch.Tensor | None = None):
        """Apply residual diagonal and optional scale, then add to target."""
        update = source_mix * self.residual_diag.to(source_mix.dtype)
        if residual_scale is not None:
            update = update * residual_scale.to(update.dtype)
        return target_hidden + update.to(target_hidden.dtype)

    def forward(self,
                target_hidden: torch.Tensor,
                source_states: _SourceStates,
                residual_scale: torch.Tensor | None = None,
                source_dim: int = -2):
        """Mix source states into ``target_hidden``."""
        source_states = self._source_tensor(target_hidden, source_states, source_dim)
        if source_states is None:
            return target_hidden
        active_sources = source_states.shape[-2]
        weights = self._route_weights(target_hidden, active_sources)
        source_mix = torch.einsum('...m,...mh->...h', weights, source_states)
        return self._add_update(target_hidden, source_mix, residual_scale)


class ConceptRoute(nn.Module):
    """Rewrite of ``_ConceptRoute``.

    Applies LayerNorm to the final concept state, scales it elementwise with a learned diagonal, optionally applies a
    route scale, then adds the update to decoder hidden states. This is replicated and token-local.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.concept_norm = nn.LayerNorm(config.hidden_size,
                                         eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                         dtype=dtype,
                                         device=device)
        self.final_diag = nn.Parameter(torch.zeros(config.hidden_size, dtype=dtype, device=device),
                                       requires_grad=False)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self,
                hidden_states: torch.Tensor,
                final_concept_state: torch.Tensor,
                final_scale: torch.Tensor | None = None):
        """forward."""
        concept = self.concept_norm(final_concept_state)
        update = concept * self.final_diag.to(concept.dtype)
        if final_scale is not None:
            update = update * final_scale.to(update.dtype)
        return hidden_states + update.to(hidden_states.dtype)


class TwoRouteAdd(nn.Module):
    """Rewrite of ``_TwoRouteAdd``.

    It first applies decoder-side ``_DepthDD`` to the decoder history, then
    injects the final concept state with ``_ConceptRoute``.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.num_layers = int(config.concept_decoder_layers)
        use_softmax = bool(getattr(config, 'concept_dd_two_route_add_decoder_use_softmax', True))
        self.decoder_dds = nn.ModuleList([
            DepthDD(config, layer_idx, use_softmax, dtype=dtype, device=device)
            for layer_idx in range(self.num_layers)
        ])
        self.concept_routes = nn.ModuleList([
            ConceptRoute(config, dtype=dtype, device=device)
            for _ in range(self.num_layers)
        ])

    def make_history_buffer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Allocate layer-major decoder history buffer ``[num_layers + 1,
        *hidden_shape]``."""
        return hidden_states.new_empty((self.num_layers + 1, *hidden_states.shape))

    @staticmethod
    def write_history(history_buffer: torch.Tensor, slot_idx: int, hidden_states: torch.Tensor):
        """Copy one decoder history block into a layer-major history buffer."""
        return SelfDD.write_history(history_buffer, slot_idx, hidden_states)

    @staticmethod
    def history_view(history_buffer: torch.Tensor, layer_idx: int):
        """Return layer-major decoder history needed by ``layer_idx`` without
        copying."""
        return SelfDD.history_view(history_buffer, layer_idx)

    def forward_from_buffer(self,
                            layer_idx: int,
                            hidden_states: torch.Tensor,
                            history_buffer: torch.Tensor,
                            final_concept_state: torch.Tensor,
                            final_scale: torch.Tensor | None = None):
        """Runtime path: fixed decoder history buffer, no list materialization."""
        layer_idx = int(layer_idx)
        hidden_states = self.decoder_dds[layer_idx](
            hidden_states,
            self.history_view(history_buffer, layer_idx),
            history_dim=0,
        )
        return self.concept_routes[layer_idx](hidden_states, final_concept_state, final_scale)


class PredictionHeads(nn.Module):
    """Merged per-codebook prediction heads.

    Native checkpoints store one ``prediction_heads.N`` linear per concept
    codebook. Runtime uses one merged projection and loads each native head into
    a deterministic output shard through LMDeploy's ``param.weight_loader``
    pattern.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        codebook_size = int(config.concept_v22_vq_codebook_size)
        num_codebooks = int(config.concept_v22_vq_num_codebooks)
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        quantization_config = getattr(config, 'quantization_config', None)
        self.proj = build_merged_colwise_linear(
            config.hidden_size,
            [codebook_size] * num_codebooks,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            # Keep concept logits replicated for now. Output TP would make the
            # trailing ``[num_codebooks, codebook_size]`` contract sharded and
            # needs a defined distributed sampling/gather path first.
            is_tp=False,
            out_names=list(range(num_codebooks)),
            prefix=add_prefix('proj', prefix),
        )

    def forward(self, hidden_states: torch.Tensor):
        """Return logits in shape ``[..., num_codebooks, codebook_size]``."""
        logits = self.proj(hidden_states)
        return logits.unflatten(-1, (self.num_codebooks, self.codebook_size))


class ConceptPredictor(nn.Module):
    """Rewrite of ``_ConceptPredictor``.

    The predictor owns the high-level concept OLMo block, per-codebook prediction heads, concept self-DD, encoder-read
    routes, and the shared source LayerNorm used before encoder states are routed into the concept stream.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.num_layers = int(config.concept_special_layers)
        self.num_codebooks = int(config.concept_v22_vq_num_codebooks)
        self.codebook_size = int(config.concept_v22_vq_codebook_size)
        self.hlm_block = OlmoBlock(config,
                                   self.num_layers,
                                   post_layer_norm=True,
                                   dtype=dtype,
                                   device=device,
                                   prefix=add_prefix('hlm_block', prefix))
        self.prediction_heads = PredictionHeads(config,
                                                dtype=dtype,
                                                device=device,
                                                prefix=add_prefix('prediction_heads', prefix))
        self.concept_self_dd = SelfDD(config,
                                      self.num_layers,
                                      use_softmax=False,
                                      dtype=dtype,
                                      device=device)
        self.concept_read_encoder_routes = nn.ModuleList([
            ResidualRoute(config,
                          int(config.concept_encoder_layers) - 1,
                          use_softmax=True,
                          dtype=dtype,
                          device=device)
            for _ in range(self.num_layers)
        ])
        self.concept_read_encoder_shared_source_norm = nn.LayerNorm(
            config.hidden_size,
            eps=getattr(config, 'layernorm_epsilon', 1e-6),
            dtype=dtype,
            device=device)
        for param in self.concept_read_encoder_shared_source_norm.parameters():
            param.requires_grad_(False)

    def normalize_encoder_concept_states(self, encoder_concept_states: torch.Tensor):
        """Apply the shared source norm used before concept-read-encoder
        routes."""
        return self.concept_read_encoder_shared_source_norm(encoder_concept_states)

    def forward(self,
                concept_hidden: torch.Tensor,
                encoder_concept_states: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]] | None = None,
                attn_metadata: Any = None,
                encoder_source_dim: int = -2):
        """Concept predictor forward.

        This mirrors the reference control flow, but uses LMDeploy's OLMo layer
        rewrite and paged attention inputs. Full end-to-end use requires the
        caller to provide concept-stream ``past_key_values`` and
        ``attn_metadata`` matching the concept token layout.
        """
        hidden_states = concept_hidden
        history_buffer = self.concept_self_dd.make_history_buffer(hidden_states)
        self.concept_self_dd.write_history(history_buffer, 0, hidden_states)
        raw_states = []
        rotary_pos_emb = self.hlm_block._make_rotary_pos_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.hlm_block.layers):
            pkv = past_key_values[layer_idx] if past_key_values is not None else None
            raw = layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=pkv,
                attn_metadata=attn_metadata,
            )
            raw_states.append(raw)
            self.concept_self_dd.write_history(history_buffer, layer_idx + 1, raw)
            hidden_states = self.concept_self_dd.forward_from_buffer(layer_idx, raw, history_buffer)
            hidden_states = self.concept_read_encoder_routes[layer_idx](
                hidden_states,
                encoder_concept_states,
                source_dim=encoder_source_dim,
            )

        hidden_states = self.hlm_block.final_layernorm(hidden_states)
        logits = self.prediction_heads(hidden_states)
        return logits, raw_states


class OlmoAttention(nn.Module):
    """Rewrite of ``_OlmoSelfAttention``.

    Differences from the reference:
      - fused QKV uses lmdeploy's ``build_qkv_proj`` (standard [Q,K,V] packing,
        TP-aware). The reference stores QKV per attention head, so the native
        checkpoint weight is re-laid-out during ``load_weights``.
      - q/k RMSNorm operates on the whole hidden_size (as in the reference),
        not per-head. TP>1 needs an all-reduced variance; we follow internvl's
        qkv_norm pattern.
      - attention runs through lmdeploy's paged ``Attention`` primitive.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 sliding_window: int | None = None,
                 prefix: str = ''):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.kv_channels
        num_kv_heads = num_heads
        assert hidden_size == num_heads * head_dim, (
            f'ConceptLM OLMo attention expects hidden_size == num_heads * head_dim, '
            f'got {hidden_size} != {num_heads} * {head_dim}.')

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.layernorm_epsilon = config.layernorm_epsilon

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('qkv_proj', prefix),
        )

        # q, k norm over the whole hidden_size (NOT per-head). tp=True with
        # head_dim alignment so the weight shards correctly under TP.
        self.q_layernorm = RMSNorm(hidden_size,
                                   config.layernorm_epsilon,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   tp=True,
                                   align=head_dim,
                                   prefix=add_prefix('q_layernorm', prefix))
        self.k_layernorm = RMSNorm(hidden_size,
                                   config.layernorm_epsilon,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   tp=True,
                                   align=head_dim,
                                   prefix=add_prefix('k_layernorm', prefix))

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(num_heads,
                                  head_dim,
                                  num_kv_heads=num_kv_heads,
                                  v_head_size=head_dim,
                                  sliding_window=None if sliding_window is None else int(sliding_window),
                                  device=device)

        # o_proj
        self.o_proj = build_o_proj(num_heads * head_dim,
                                   hidden_size,
                                   bias=False,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   is_tp=True,
                                   prefix=add_prefix('o_proj', prefix))

    def _qkv_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm over the whole hidden_size, TP-correct via all-reduce."""
        import lmdeploy.pytorch.distributed as dist
        q_shape = q.shape
        k_shape = k.shape
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)

        tp, _ = dist.get_tp_world_rank('attn')
        if tp == 1:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)
            return q.view(q_shape), k.view(k_shape)

        # variance is computed over the full hidden_size, so it must be
        # all-reduced across TP ranks before normalizing the local shard.
        variance = _qk_rmsnorm_variance(q, k)
        dist.all_reduce(variance)
        q, k = _qk_rmsnorm_apply(q, k, variance, self.q_layernorm.weight,
                                 self.k_layernorm.weight, self.hidden_size,
                                 self.layernorm_epsilon)
        return q.view(q_shape), k.view(k_shape)

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
                past_key_value: list[torch.Tensor] | None = None,
                attn_metadata: Any = None):
        """Rewrite of _OlmoSelfAttention.forward."""
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.flatten(0, -2)  # [num_tokens, packed_qkv_dim]
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        # q, k norm (whole hidden_size)
        query_states, key_states = self._qkv_norm(query_states, key_states)

        # rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, inplace=True)

        # attention (paged)
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


class OlmoMLP(nn.Module):
    """Rewrite of ``_OlmoMLP`` (SwiGLU)."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.gate_up_proj = build_gateup_linear(
            config.hidden_size,
            [config.ffn_hidden_size, config.ffn_hidden_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            prefix=add_prefix('gate_up_proj', prefix),
        )
        self.act_fn = SiluAndMul(inplace=True)
        self.down_proj = build_down_linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            prefix=add_prefix('down_proj', prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        gate_up = self.gate_up_proj(hidden_states)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class OlmoLayer(nn.Module):
    """Rewrite of ``_OlmoLayer``.

    The reference uses post-norm residuals::
        h = h + post_attention_layernorm(attn(h))
        h = h + post_feedforward_layernorm(mlp(h))
    This block therefore calls RMSNorm without the residual argument and adds
    the residual explicitly.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 sliding_window: int | None = None,
                 prefix: str = ''):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.layer_number = int(layer_idx) + 1
        self.self_attention = OlmoAttention(config,
                                            dtype=dtype,
                                            device=device,
                                            sliding_window=sliding_window,
                                            prefix=add_prefix('self_attention', prefix))
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.layernorm_epsilon,
                                                quant_config=quantization_config,
                                                dtype=dtype,
                                                device=device,
                                                prefix=add_prefix('post_attention_layernorm', prefix))
        self.mlp = OlmoMLP(config, dtype=dtype, device=device, prefix=add_prefix('mlp', prefix))
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                  config.layernorm_epsilon,
                                                  quant_config=quantization_config,
                                                  dtype=dtype,
                                                  device=device,
                                                  prefix=add_prefix('post_feedforward_layernorm', prefix))

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
                past_key_value: list[torch.Tensor] | None = None,
                attn_metadata: Any = None):
        """forward.

        The reference uses post-norm residuals::
            h = h + post_attention_layernorm(attn(h))
            h = h + post_feedforward_layernorm(mlp(h))
        This is NOT the same as lmdeploy's pre-norm residual form
        ``norm(x, residual=h)`` (which normalizes x+h). So we call the norm
        without a residual and add explicitly.
        """
        attn_out = self.self_attention(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attn_out)

        mlp_out = self.mlp(hidden_states)
        hidden_states = hidden_states + self.post_feedforward_layernorm(mlp_out)
        return hidden_states


class OlmoBlock(nn.Module):
    """Rewrite of ``_OlmoBlock``.

    Holds a stack of ``_OlmoLayer`` and an optional final RMSNorm. Mirrors the
    reference's ``forward`` return contract: ``(hidden_states, layer_states)``.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 num_layers: int,
                 post_layer_norm: bool,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 window_size: Any = _CONFIG_VALUE,
                 skip_frequency: Any = _CONFIG_VALUE,
                 prefix: str = ''):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        if window_size is _CONFIG_VALUE:
            window_size = _get_configured_window(config)
        elif isinstance(window_size, (list, tuple)):
            window_size = window_size[0] if len(window_size) > 0 else None
        if window_size is not None:
            window_size = int(window_size)
        if skip_frequency is _CONFIG_VALUE:
            skip_frequency = getattr(config, 'window_attn_skip_freq', None)
        if skip_frequency is not None:
            skip_frequency = int(skip_frequency)

        self.layers = nn.ModuleList([
            OlmoLayer(config,
                      layer_idx,
                      dtype=dtype,
                      device=device,
                      sliding_window=self._layer_sliding_window(layer_idx + 1, window_size, skip_frequency),
                      prefix=add_prefix(f'layers.{layer_idx}', prefix))
            for layer_idx in range(num_layers)
        ])
        self.final_layernorm = (
            RMSNorm(config.hidden_size,
                    config.layernorm_epsilon,
                    quant_config=quantization_config,
                    dtype=dtype,
                    device=device,
                    prefix=add_prefix('final_layernorm', prefix)) if post_layer_norm else None)
        self.rotary_emb = _make_olmo_rotary_embedding(config, device=device)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.kv_channels)

    @staticmethod
    def _layer_sliding_window(layer_number: int, window_size: int | None, skip_frequency: int | None):
        """Match reference OLMo window/full attention alternation."""
        if window_size is None or skip_frequency is None:
            return None
        if layer_number % skip_frequency == 0:
            return None
        return window_size

    def _make_rotary_pos_emb(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        """Create RoPE cos/sin in the shape consumed by ``ApplyRotaryEmb``."""
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        if cos.dim() == 3 and cos.size(0) == 1:
            cos = cos[0]
            sin = sin[0]
        return cos, sin

    def forward(self,
                hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]] | None = None,
                attn_metadata: Any = None,
                collect: bool = False):
        """forward."""
        layer_states = []
        rotary_pos_emb = self._make_rotary_pos_emb(hidden_states, position_ids)
        for idx, layer in enumerate(self.layers):
            pkv = past_key_values[idx] if past_key_values is not None else None
            hidden_states = layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=pkv,
                attn_metadata=attn_metadata,
            )
            if collect:
                layer_states.append(hidden_states)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, layer_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], prefix: str = ''):
        """Load native ConceptLM OLMo block weights into the LMDeploy
        rewrite."""
        if prefix and not prefix.endswith('.'):
            prefix = f'{prefix}.'

        params_dict = dict(self.named_parameters())
        loaded_names = set()
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if prefix:
                if not name.startswith(prefix):
                    continue
                name = name[len(prefix):]

            if name.endswith('.self_attention.qkv_proj.weight'):
                param = params_dict[name]
                query, key, value = param.weight_spliter(loaded_weight)
                load_weight(param, query, shard_id='q')
                load_weight(param, key, shard_id='k')
                load_weight(param, value, shard_id='v')
                loaded_names.add(name)
                continue

            if name.endswith('.self_attention.linear_qkv.weight'):
                target_name = name.replace('.self_attention.linear_qkv.weight',
                                           '.self_attention.qkv_proj.weight')
                param = params_dict[target_name]
                loaded_weight = _repack_olmo_qkv_weight(loaded_weight, self.num_heads, self.head_dim)
                query, key, value = param.weight_spliter(loaded_weight)
                load_weight(param, query, shard_id='q')
                load_weight(param, key, shard_id='k')
                load_weight(param, value, shard_id='v')
                loaded_names.add(target_name)
                continue

            if name.endswith('.self_attention.linear_proj.weight'):
                target_name = name.replace('.self_attention.linear_proj.weight',
                                           '.self_attention.o_proj.weight')
            elif name.endswith('.mlp.gate_up_proj.weight'):
                param = params_dict[name]
                gate, up = param.weight_spliter(loaded_weight)
                load_weight(param, gate, shard_id=0)
                load_weight(param, up, shard_id=1)
                loaded_names.add(name)
                continue
            elif name.endswith('.mlp.linear_fc1.weight'):
                target_name = name.replace('.mlp.linear_fc1.weight', '.mlp.gate_up_proj.weight')
                param = params_dict[target_name]
                gate, up = param.weight_spliter(loaded_weight)
                load_weight(param, gate, shard_id=0)
                load_weight(param, up, shard_id=1)
                loaded_names.add(target_name)
                continue
            elif name.endswith('.mlp.linear_fc2.weight'):
                target_name = name.replace('.mlp.linear_fc2.weight', '.mlp.down_proj.weight')
            else:
                target_name = name

            param = params_dict.get(target_name)
            if param is None:
                continue
            load_weight(param, loaded_weight)
            loaded_names.add(target_name)
        return loaded_names
