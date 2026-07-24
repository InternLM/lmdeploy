# Copyright (c) OpenMMLab. All rights reserved.
"""lmdeploy adapter for ConceptLM V2.2-VQ.

Reference implementation:
``concept_olmo_stage_2_V1/modeling_conceptlm_v22_vq.py``

Modules are added incrementally. Current state:
  - token embedding + output projection (lm_head)
  - ``_OlmoBlock`` (encoder/decoder/concept_predictor backbone): attention,
    mlp, rmsnorm, rope. Wired to lmdeploy primitives so it is TP-correct and
    ready to plug into the engine's paged attention path.
  - ``_Quantizer``: stacked VQ codebook parameter, replicated across TP.
  - ``_SelfDD``: replicated per-token depth mixer for encoder hidden history.
  - ``_ResidualRoute``: replicated residual source mixer for decoder routes.
  - ``_TwoRouteAdd``: decoder depth mixing plus final-concept residual route.
  - ``_ConceptPredictor``: concept block container and prediction heads.
  - top-level encoder/decoder containers, fusion norms, route norms, and
    checkpoint loading for the implemented module tree.

Results are NOT yet correct end-to-end — the encoder/concept/decoder control
flow still needs runtime metadata for the compressed concept stream. The
module structure mirrors the reference for readability.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, ConceptLMRuntimeOps, RMSNorm, SiluAndMul,
                                 build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_down_linear, build_gateup_linear,
                                        build_merged_colwise_linear, build_o_proj, build_qkv_proj)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import add_prefix
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixinV1, build_embedding

_CONFIG_VALUE = object()
_HistoryStates = list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor
_SourceStates = list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor | None


@dataclass
class ConceptMetadata:
    """Layer-invariant ConceptLM runtime metadata.

    This mirrors the DSV4 pattern: ``StepContext`` is read at the top-level
    model boundary, then submodules receive explicit metadata instead of
    reaching back into the engine context. The dense/reference helpers below do
    not consume all fields yet; they are part of the serving decode contract.
    """

    chunk_size: int
    merge_method: str
    shift_feature: bool
    is_decoding: bool | None = None
    state_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    block_offsets: torch.Tensor | None = None
    q_seqlens: torch.Tensor | None = None
    kv_seqlens: torch.Tensor | None = None
    q_start_loc: torch.Tensor | None = None
    attn_metadata: Any = None

    @classmethod
    def build(cls,
              config: PretrainedConfig,
              position_ids: torch.Tensor | None = None,
              attn_metadata: Any = None,
              state_ids: torch.Tensor | None = None):
        """Build ConceptLM metadata from explicit forward inputs."""
        return cls(
            chunk_size=int(config.concept_chunk_size),
            merge_method=getattr(config, 'concept_chunk_merge_method', 'meanpooling'),
            shift_feature=bool(getattr(config, 'concept_shift_feature', True)),
            is_decoding=getattr(attn_metadata, 'is_decoding', None),
            state_ids=state_ids,
            position_ids=position_ids,
            block_offsets=getattr(attn_metadata, 'block_offsets', None),
            q_seqlens=getattr(attn_metadata, 'q_seqlens', None),
            kv_seqlens=getattr(attn_metadata, 'kv_seqlens', None),
            q_start_loc=getattr(attn_metadata, 'q_start_loc', None),
            attn_metadata=attn_metadata,
        )


@dataclass
class ConceptCaches:
    """ConceptLM cache views resolved once at the top-level model boundary."""

    encoder_past_key_values: list[list[torch.Tensor]] | None = None
    concept_past_key_values: list[list[torch.Tensor]] | None = None
    decoder_past_key_values: list[list[torch.Tensor]] | None = None
    state_caches: list[torch.Tensor] | None = None
    chunk_source_idx: int = 0
    last_raw_idx: int = 1
    last_final_idx: int = 2

    @classmethod
    def build(cls,
              config: PretrainedConfig,
              past_key_values: list[list[torch.Tensor]] | None = None,
              state_caches: list[torch.Tensor] | None = None):
        """Build ConceptLM cache views from engine-provided caches."""
        encoder_past_key_values, concept_past_key_values, decoder_past_key_values = (
            _split_concept_past_key_values(config, past_key_values))
        return cls(
            encoder_past_key_values=encoder_past_key_values,
            concept_past_key_values=concept_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
            state_caches=state_caches,
            chunk_source_idx=int(getattr(config, 'concept_state_chunk_source_idx', 0)),
            last_raw_idx=int(getattr(config, 'concept_state_last_raw_idx', 1)),
            last_final_idx=int(getattr(config, 'concept_state_last_final_idx', 2)),
        )

    def state_cache(self, state_idx: int) -> torch.Tensor | None:
        """Return one anonymous state-cache tensor by semantic index."""
        if self.state_caches is None:
            return None
        if state_idx < 0 or state_idx >= len(self.state_caches):
            return None
        return self.state_caches[state_idx]

    @property
    def chunk_source_state(self) -> torch.Tensor | None:
        """Current chunk source accumulator state cache."""
        return self.state_cache(self.chunk_source_idx)

    @property
    def last_raw_states(self) -> torch.Tensor | None:
        """Latest raw concept-layer state cache."""
        return self.state_cache(self.last_raw_idx)

    @property
    def last_final_state(self) -> torch.Tensor | None:
        """Latest final concept vector state cache."""
        return self.state_cache(self.last_final_idx)


@dataclass
class ConceptChunkStateUpdateResult:
    """Fixed-shape result of one decode chunk-source state update."""

    concept_input_states: torch.Tensor
    next_chunk_source_states: torch.Tensor
    concept_update_mask: torch.Tensor
    valid_state_mask: torch.Tensor
    state_ids: torch.Tensor


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

    head_dim = int(config.kv_channels)
    rotary_percent = float(getattr(config, 'rotary_percent', 1.0))
    rotary_dim = int(head_dim * rotary_percent)
    rotary_dim -= rotary_dim % 2
    if rotary_dim <= 0:
        raise ValueError(f'Invalid ConceptLM rotary dimension: head_dim={head_dim}, rotary_percent={rotary_percent}')

    partial_rotary_factor = rotary_dim / head_dim
    return build_rotary_embedding(
        dim=head_dim,
        max_position_embeddings=getattr(config, 'max_position_embeddings', getattr(config, 'max_sequence_length',
                                                                                   2048)),
        base=getattr(config, 'rotary_base', 10000),
        partial_rotary_factor=partial_rotary_factor,
        device=device,
    )


def _repack_olmo_qkv_weight(loaded_weight: torch.Tensor, num_heads: int, head_dim: int):
    """Convert native OLMo per-head [Q,K,V] QKV packing to LMDeploy [Q][K][V]."""
    leading_shape = loaded_weight.shape[1:]
    loaded_weight = loaded_weight.reshape(num_heads, 3, head_dim, *leading_shape)
    query = loaded_weight[:, 0].flatten(0, 1)
    key = loaded_weight[:, 1].flatten(0, 1)
    value = loaded_weight[:, 2].flatten(0, 1)
    return torch.cat([query, key, value], dim=0)


def _load_stacked_codebook_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor, codebook_idx: int):
    """Load one native ``codebook.N`` checkpoint tensor into a stacked codebook."""
    assert 0 <= codebook_idx < param.size(0), f'Invalid codebook index: {codebook_idx}'
    target = param.data[codebook_idx]
    assert target.size() == loaded_weight.size(), (
        f'Attempted to load codebook weight ({loaded_weight.size()}) into parameter slice ({target.size()})')
    target.copy_(loaded_weight)


def _split_concept_past_key_values(config: PretrainedConfig, past_key_values: list[list[torch.Tensor]] | None):
    """Split the flat LMDeploy KV-cache list into ConceptLM streams."""
    if past_key_values is None or len(past_key_values) == 0:
        return None, None, None
    enc_layers = int(config.concept_encoder_layers)
    concept_layers = int(config.concept_special_layers)
    dec_layers = int(config.concept_decoder_layers)
    total_layers = enc_layers + concept_layers + dec_layers
    assert len(past_key_values) >= total_layers, (
        f'ConceptLM requires {total_layers} KV-cache layers '
        f'({enc_layers} encoder + {concept_layers} concept + {dec_layers} decoder), '
        f'got {len(past_key_values)}.')
    enc_end = enc_layers
    concept_end = enc_end + concept_layers
    dec_end = concept_end + dec_layers
    return past_key_values[:enc_end], past_key_values[enc_end:concept_end], past_key_values[concept_end:dec_end]


def _flatten_decode_position_ids(position_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize decode position ids to one absolute position per batch row."""
    if position_ids.dim() == 0:
        position_ids = position_ids.view(1)
    if position_ids.dim() == 1:
        return position_ids.to(torch.long)
    position_ids = position_ids.reshape(-1)
    if position_ids.numel() == batch_size:
        return position_ids.to(torch.long)
    assert position_ids.numel() % batch_size == 0, (
        f'Cannot map position_ids with {position_ids.numel()} elements to batch size {batch_size}.')
    return position_ids.reshape(-1, batch_size)[-1].to(torch.long)


def _concept_decode_chunk_state_update(
    chunk_source_state_cache: torch.Tensor,
    current_source_states: torch.Tensor,
    state_ids: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_size: int,
    merge_method: str,
) -> ConceptChunkStateUpdateResult:
    """Compute one decode step's chunk-source state update.

    This helper is intentionally fixed-shape over the decode batch. It returns
    per-row next states plus a device-side boundary mask; it does not compact
    concept rows by ``num_concepts_total``. The future Triton/CUDA op should
    fuse this compute with the state write and skip ``state_id < 0`` rows.

    Args:
        chunk_source_state_cache: ``[num_state_slots, num_sources, hidden]``.
        current_source_states: ``[batch, num_sources, hidden]`` for the current
            decode token after encoder source selection. These should be the
            unnormalized states that are merged over the current concept chunk.
        state_ids: ``[batch]`` state-cache slot per row, with ``-1`` for
            padded CUDA-graph rows.
        position_ids: absolute token positions for the decode rows.
        chunk_size: ConceptLM chunk size.
        merge_method: ``meanpooling``, ``first``, or ``last``.
    """
    assert current_source_states.dim() == 3, (
        f'current_source_states must be [batch, num_sources, hidden], got {tuple(current_source_states.shape)}.')
    assert chunk_source_state_cache.dim() == 3, (
        f'chunk_source_state_cache must be [num_state_slots, num_sources, hidden], '
        f'got {tuple(chunk_source_state_cache.shape)}.')
    batch_size = current_source_states.size(0)
    assert current_source_states.shape[1:] == chunk_source_state_cache.shape[1:], (
        f'Current source state shape {tuple(current_source_states.shape[1:])} does not match state-cache shape '
        f'{tuple(chunk_source_state_cache.shape[1:])}.')

    state_ids = state_ids.to(device=current_source_states.device, dtype=torch.long)
    position_ids = _flatten_decode_position_ids(position_ids, batch_size).to(device=current_source_states.device)
    assert position_ids.numel() == batch_size, (
        f'Expected {batch_size} decode position ids, got {position_ids.numel()}.')

    valid_state_mask = state_ids >= 0
    safe_state_ids = state_ids.clamp(min=0)
    previous_rows = chunk_source_state_cache.index_select(0, safe_state_ids)

    chunk_size = int(chunk_size)
    chunk_pos = torch.remainder(position_ids, chunk_size)
    concept_update_mask = valid_state_mask & (torch.remainder(position_ids + 1, chunk_size) == 0)
    first_token_mask = valid_state_mask & (chunk_pos == 0)
    merge_method = str(merge_method)

    if merge_method == 'first':
        update_rows = torch.where(first_token_mask.view(batch_size, 1, 1), current_source_states, previous_rows)
        concept_input_states = update_rows
    elif merge_method == 'last':
        update_rows = current_source_states
        concept_input_states = current_source_states
    else:
        update_rows = previous_rows + current_source_states
        concept_input_states = update_rows / chunk_size

    zero_rows = torch.zeros_like(update_rows)
    next_rows = torch.where(concept_update_mask.view(batch_size, 1, 1), zero_rows, update_rows)
    next_rows = torch.where(valid_state_mask.view(batch_size, 1, 1), next_rows, previous_rows)
    concept_input_states = torch.where(concept_update_mask.view(batch_size, 1, 1), concept_input_states, zero_rows)
    return ConceptChunkStateUpdateResult(
        concept_input_states=concept_input_states,
        next_chunk_source_states=next_rows,
        concept_update_mask=concept_update_mask,
        valid_state_mask=valid_state_mask,
        state_ids=state_ids,
    )


def _apply_concept_chunk_state_update_reference_(chunk_source_state_cache: torch.Tensor,
                                                update: ConceptChunkStateUpdateResult):
    """Reference-only state write for tests.

    This uses a Python loop and may read scalar state ids on host. Do not call
    it from the serving hot path. The graph-safe implementation should write
    ``update.next_chunk_source_states`` inside a backend op that skips
    ``state_id < 0`` rows.
    """
    for batch_idx in range(update.state_ids.numel()):
        state_id = int(update.state_ids[batch_idx])
        if state_id < 0:
            continue
        chunk_source_state_cache[state_id].copy_(update.next_chunk_source_states[batch_idx])


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


class ConceptLMV22VQEmbedding(nn.Module):
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


class ConceptLMV22VQQuantizer(nn.Module):
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
        """Return codebook as ``[num_codebooks, codebook_size, codebook_dim]``."""
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
        assert concept_logits.shape[-2:] == (self.num_codebooks, self.codebook_size), (
            f'Expected concept logits trailing shape {(self.num_codebooks, self.codebook_size)}, '
            f'got {tuple(concept_logits.shape[-2:])}.')
        codebook = self.transformed_codebook().to(concept_logits.dtype)
        vectors = torch.einsum('...hk,hkd->...hd', concept_logits, codebook)
        return vectors.flatten(-2, -1)


class ConceptLMV22VQDepthDD(nn.Module):
    """Rewrite of ``_DepthDD``.

    This is a small replicated per-token depth mixer. It does not mix tokens;
    it computes ``num_prev`` route weights from the current hidden state and
    combines the matching per-layer history states. The reference only handles
    dense ``[seq, batch, hidden]`` tensors because it stacks history at
    ``dim=2``. LMDeploy continuous batching commonly uses packed
    ``[num_tokens, hidden]`` tensors. Runtime should pass a preallocated tensor
    history to avoid repeated ``torch.stack`` copies; list/tuple input remains
    only as a debug/parity convenience.
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
        if not isinstance(history_states, torch.Tensor):
            assert len(history_states) == self.num_prev, (
                f'Expected {self.num_prev} history states, got {len(history_states)}.')
            history_states = torch.stack(tuple(history_states), dim=-2)
            history_dim = -2

        history_dim = history_dim if history_dim >= 0 else history_dim + history_states.dim()
        assert 0 <= history_dim < history_states.dim(), f'Invalid history_dim={history_dim}.'
        assert history_states.shape[history_dim] == self.num_prev, (
            f'Expected history dimension {history_dim} to be {self.num_prev}, '
            f'got {history_states.shape[history_dim]}.')
        if history_dim != history_states.dim() - 2:
            history_states = history_states.movedim(history_dim, -2)
        assert history_states.shape[:-2] == hidden_states.shape[:-1], (
            f'Expected history leading shape {tuple(hidden_states.shape[:-1])}, '
            f'got {tuple(history_states.shape[:-2])}.')
        assert history_states.shape[-1] == hidden_states.shape[-1], (
            f'Expected history hidden size {hidden_states.shape[-1]}, got {history_states.shape[-1]}.')
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


class ConceptLMV22VQSelfDD(nn.Module):
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
            ConceptLMV22VQDepthDD(config, layer_idx, use_softmax, dtype=dtype, device=device)
            for layer_idx in range(self.num_layers)
        ])

    def make_history_buffer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Allocate layer-major history buffer ``[num_layers + 1, *hidden_shape]``."""
        return hidden_states.new_empty((self.num_layers + 1, *hidden_states.shape))

    @staticmethod
    def write_history(history_buffer: torch.Tensor, slot_idx: int, hidden_states: torch.Tensor):
        """Copy one history block into a layer-major history buffer."""
        assert history_buffer.dim() == hidden_states.dim() + 1, (
            f'Expected history buffer dim {hidden_states.dim() + 1}, got {history_buffer.dim()}.')
        assert history_buffer.shape[1:] == hidden_states.shape, (
            f'Expected history buffer trailing shape {tuple(hidden_states.shape)}, '
            f'got {tuple(history_buffer.shape[1:])}.')
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


class ConceptLMV22VQResidualRoute(nn.Module):
    """Rewrite of ``_ResidualRoute``.

    This module computes a source-state mixture and adds it as a gated residual
    update to the target hidden state. It is small and replicated.

    Runtime/static-graph code should use ``forward_from_buffer`` with a full
    fixed-size source buffer. Flexible active-source/list paths are retained for
    parity tests and WIP reference wiring, but they should not be used inside a
    captured CUDA graph because they can change intermediate shapes.
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
                       source_dim: int,
                       expected_leading_shape: tuple[int, ...] | None = None):
        """Return source states in shape ``[..., active_sources, hidden_size]``."""
        if source_states is None:
            return None
        if not isinstance(source_states, torch.Tensor):
            if len(source_states) == 0:
                return None
            source_states = torch.stack(tuple(source_states), dim=-2)
            source_dim = -2

        source_dim = source_dim if source_dim >= 0 else source_dim + source_states.dim()
        assert 0 <= source_dim < source_states.dim(), f'Invalid source_dim={source_dim}.'
        if source_states.shape[source_dim] == 0:
            return None
        if source_dim != source_states.dim() - 2:
            source_states = source_states.movedim(source_dim, -2)
        if expected_leading_shape is None:
            expected_leading_shape = target_hidden.shape[:-1]
        assert source_states.shape[:-2] == expected_leading_shape, (
            f'Expected source leading shape {tuple(expected_leading_shape)}, '
            f'got {tuple(source_states.shape[:-2])}.')
        assert source_states.shape[-1] == target_hidden.shape[-1], (
            f'Expected source hidden size {target_hidden.shape[-1]}, got {source_states.shape[-1]}.')
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

    def make_source_buffer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Allocate source-major buffer ``[num_source_states, *hidden_shape]``."""
        return hidden_states.new_empty((self.num_source_states, *hidden_states.shape))

    @staticmethod
    def write_source(source_buffer: torch.Tensor, slot_idx: int, source_state: torch.Tensor):
        """Copy one source state into a source-major buffer."""
        assert source_buffer.dim() == source_state.dim() + 1, (
            f'Expected source buffer dim {source_state.dim() + 1}, got {source_buffer.dim()}.')
        assert source_buffer.shape[1:] == source_state.shape, (
            f'Expected source buffer trailing shape {tuple(source_state.shape)}, '
            f'got {tuple(source_buffer.shape[1:])}.')
        source_buffer[int(slot_idx)].copy_(source_state)
        return source_buffer

    @staticmethod
    def source_view(source_buffer: torch.Tensor, active_sources: int | None = None):
        """Return active source-major view without copying.

        Passing ``active_sources`` is a flexible/debug path. Static graph
        runtime should pass full fixed buffers and leave ``active_sources`` as
        ``None``.
        """
        if active_sources is None:
            return source_buffer
        return source_buffer[:int(active_sources)]

    def forward(self,
                target_hidden: torch.Tensor,
                source_states: _SourceStates,
                residual_scale: torch.Tensor | None = None,
                source_dim: int = -2):
        """Flexible/debug forward path.

        This accepts lists, ``None``, and active source tensors for reference
        parity. Use ``forward_from_buffer`` for static-shape runtime.
        """
        source_states = self._source_tensor(target_hidden, source_states, source_dim)
        if source_states is None:
            return target_hidden
        active_sources = source_states.shape[-2]
        weights = self._route_weights(target_hidden, active_sources)
        source_mix = torch.einsum('...m,...mh->...h', weights, source_states)
        return self._add_update(target_hidden, source_mix, residual_scale)

    def forward_from_buffer(self,
                            target_hidden: torch.Tensor,
                            source_buffer: torch.Tensor,
                            residual_scale: torch.Tensor | None = None):
        """Runtime path: read a full fixed source-major buffer.

        ``source_buffer`` shape is ``[num_source_states, *target_hidden.shape]``.
        This keeps source count fixed across CUDA graph capture/replay.
        """
        assert source_buffer.shape[0] == self.num_source_states, (
            f'Expected full source buffer with {self.num_source_states} states, got {source_buffer.shape[0]}.')
        return self.forward(
            target_hidden,
            source_buffer,
            residual_scale=residual_scale,
            source_dim=0,
        )

    def forward_active_from_buffer(self,
                                   target_hidden: torch.Tensor,
                                   source_buffer: torch.Tensor,
                                   active_sources: int,
                                   residual_scale: torch.Tensor | None = None):
        """Flexible/debug buffer path with reduced active source count."""
        return self.forward(
            target_hidden,
            self.source_view(source_buffer, active_sources),
            residual_scale=residual_scale,
            source_dim=0,
        )

    def forward_repeated_chunks(self,
                                target_hidden: torch.Tensor,
                                source_states: _SourceStates,
                                chunk_size: int,
                                shift_feature: bool,
                                residual_scale: torch.Tensor | None = None,
                                source_dim: int = 2):
        """Reference repeated-chunk route used by decoder-read-concept.

        ``target_hidden`` is dense ``[seq, batch, hidden]`` and source states are
        chunk-level ``[chunks, batch, sources, hidden]`` after ``source_dim`` is
        normalized to 2. A packed continuous-batching runtime will need token to
        chunk metadata before using this path end-to-end. This method is not a
        complete CUDA-graph runtime path yet because chunk lengths still need a
        fixed-buffer/mask contract at the caller level.
        """
        if source_states is None:
            return target_hidden
        assert target_hidden.dim() == 3, (
            f'forward_repeated_chunks expects [seq, batch, hidden], got {tuple(target_hidden.shape)}.')

        if not isinstance(source_states, torch.Tensor):
            if len(source_states) == 0:
                return target_hidden
            source_states = torch.stack(tuple(source_states), dim=2)
            source_dim = 2
        source_dim = source_dim if source_dim >= 0 else source_dim + source_states.dim()
        assert 0 <= source_dim < source_states.dim(), f'Invalid source_dim={source_dim}.'
        if source_states.shape[source_dim] == 0:
            return target_hidden
        if source_dim != 2:
            source_states = source_states.movedim(source_dim, 2)

        seq_len, batch_size, hidden_size = target_hidden.shape
        assert source_states.dim() == 4, (
            f'Expected source states [chunks, batch, sources, hidden], got {tuple(source_states.shape)}.')
        assert source_states.shape[1] == batch_size, (
            f'Expected source batch size {batch_size}, got {source_states.shape[1]}.')
        assert source_states.shape[3] == hidden_size, (
            f'Expected source hidden size {hidden_size}, got {source_states.shape[3]}.')
        num_chunks, _, active_sources, _ = source_states.shape
        chunk_size = int(chunk_size)
        weights = self._route_weights(target_hidden, active_sources)
        if shift_feature:
            weights = torch.cat((weights.new_zeros(1, batch_size, active_sources), weights), dim=0)
        repeated_len = num_chunks * chunk_size
        if weights.shape[0] < repeated_len:
            pad_len = repeated_len - weights.shape[0]
            weights = torch.cat((weights, weights.new_zeros(pad_len, batch_size, active_sources)), dim=0)
        weights = weights[:repeated_len]

        source_mix = torch.einsum(
            'ckbm,cbmh->ckbh',
            weights.reshape(num_chunks, chunk_size, batch_size, active_sources),
            source_states,
        ).reshape(repeated_len, batch_size, hidden_size)
        if shift_feature:
            source_mix = source_mix[1:1 + seq_len]
        else:
            source_mix = source_mix[:seq_len]
        if source_mix.shape[0] < seq_len:
            pad_len = seq_len - source_mix.shape[0]
            source_mix = torch.cat((source_mix, source_mix.new_zeros(pad_len, batch_size, hidden_size)), dim=0)
        return self._add_update(target_hidden, source_mix, residual_scale)

    def forward_repeated_chunks_from_buffer(self,
                                            target_hidden: torch.Tensor,
                                            source_buffer: torch.Tensor,
                                            chunk_size: int,
                                            shift_feature: bool,
                                            residual_scale: torch.Tensor | None = None):
        """Full source-major buffer variant of ``forward_repeated_chunks``."""
        assert source_buffer.shape[0] == self.num_source_states, (
            f'Expected full source buffer with {self.num_source_states} states, got {source_buffer.shape[0]}.')
        return self.forward_repeated_chunks(
            target_hidden,
            source_buffer,
            chunk_size,
            shift_feature,
            residual_scale=residual_scale,
            source_dim=0,
        )

    def forward_repeated_chunks_active_from_buffer(self,
                                                   target_hidden: torch.Tensor,
                                                   source_buffer: torch.Tensor,
                                                   chunk_size: int,
                                                   shift_feature: bool,
                                                   active_sources: int,
                                                   residual_scale: torch.Tensor | None = None):
        """Flexible/debug repeated-chunk buffer path with reduced active source count."""
        return self.forward_repeated_chunks(
            target_hidden,
            self.source_view(source_buffer, active_sources),
            chunk_size,
            shift_feature,
            residual_scale=residual_scale,
            source_dim=0,
        )


class ConceptLMV22VQConceptRoute(nn.Module):
    """Rewrite of ``_ConceptRoute``.

    Applies LayerNorm to the final concept state, scales it elementwise with a
    learned diagonal, optionally applies a route scale, then adds the update to
    decoder hidden states. This is replicated and token-local.
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


class ConceptLMV22VQTwoRouteAdd(nn.Module):
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
            ConceptLMV22VQDepthDD(config, layer_idx, use_softmax, dtype=dtype, device=device)
            for layer_idx in range(self.num_layers)
        ])
        self.concept_routes = nn.ModuleList([
            ConceptLMV22VQConceptRoute(config, dtype=dtype, device=device)
            for _ in range(self.num_layers)
        ])

    def make_history_buffer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Allocate layer-major decoder history buffer ``[num_layers + 1, *hidden_shape]``."""
        return hidden_states.new_empty((self.num_layers + 1, *hidden_states.shape))

    @staticmethod
    def write_history(history_buffer: torch.Tensor, slot_idx: int, hidden_states: torch.Tensor):
        """Copy one decoder history block into a layer-major history buffer."""
        return ConceptLMV22VQSelfDD.write_history(history_buffer, slot_idx, hidden_states)

    @staticmethod
    def history_view(history_buffer: torch.Tensor, layer_idx: int):
        """Return layer-major decoder history needed by ``layer_idx`` without copying."""
        return ConceptLMV22VQSelfDD.history_view(history_buffer, layer_idx)

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

    def forward(self,
                layer_idx: int,
                hidden_states: torch.Tensor,
                history_states: _HistoryStates,
                final_concept_state: torch.Tensor,
                final_scale: torch.Tensor | None = None):
        """Flexible/debug forward path."""
        layer_idx = int(layer_idx)
        hidden_states = self.decoder_dds[layer_idx](hidden_states, history_states)
        return self.concept_routes[layer_idx](hidden_states, final_concept_state, final_scale)


class ConceptLMV22VQPredictionHeads(nn.Module):
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


class ConceptLMV22VQConceptPredictor(nn.Module):
    """Rewrite of ``_ConceptPredictor``.

    The predictor owns the high-level concept OLMo block, per-codebook
    prediction heads, concept self-DD, encoder-read routes, and the shared
    source LayerNorm used before encoder states are routed into the concept
    stream.
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
        self.hlm_block = ConceptLMV22VQOlmoBlock(config,
                                                 self.num_layers,
                                                 post_layer_norm=True,
                                                 dtype=dtype,
                                                 device=device,
                                                 prefix=add_prefix('hlm_block', prefix))
        self.prediction_heads = ConceptLMV22VQPredictionHeads(config,
                                                              dtype=dtype,
                                                              device=device,
                                                              prefix=add_prefix('prediction_heads', prefix))
        self.concept_self_dd = ConceptLMV22VQSelfDD(config,
                                                    self.num_layers,
                                                    use_softmax=False,
                                                    dtype=dtype,
                                                    device=device)
        self.concept_read_encoder_routes = nn.ModuleList([
            ConceptLMV22VQResidualRoute(config,
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

    def set_attention_window(self, window_size, skip_frequency):
        """Reference-compatible API.

        LMDeploy's rewrite bakes per-layer sliding-window policy into
        ``ConceptLMV22VQOlmoBlock`` at construction time, so this is retained as
        an explicit no-op for call-site compatibility.
        """
        self._window_size = window_size
        self._window_skip_frequency = skip_frequency

    def normalize_encoder_concept_states(self, encoder_concept_states: torch.Tensor):
        """Apply the shared source norm used before concept-read-encoder routes."""
        return self.concept_read_encoder_shared_source_norm(encoder_concept_states)

    def predict_logits(self, hidden_states: torch.Tensor):
        """Return logits in shape ``[..., num_codebooks, codebook_size]``."""
        return self.prediction_heads(hidden_states)

    def make_history_buffer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Allocate concept self-DD history buffer."""
        return self.concept_self_dd.make_history_buffer(hidden_states)

    @staticmethod
    def write_history(history_buffer: torch.Tensor, slot_idx: int, hidden_states: torch.Tensor):
        """Copy one concept history block into a layer-major history buffer."""
        return ConceptLMV22VQSelfDD.write_history(history_buffer, slot_idx, hidden_states)

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
        history_buffer = self.make_history_buffer(hidden_states)
        self.write_history(history_buffer, 0, hidden_states)
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
            self.write_history(history_buffer, layer_idx + 1, raw)
            hidden_states = self.concept_self_dd.forward_from_buffer(layer_idx, raw, history_buffer)
            hidden_states = self.concept_read_encoder_routes[layer_idx](
                hidden_states,
                encoder_concept_states,
                source_dim=encoder_source_dim,
            )

        hidden_states = self.hlm_block.final_layernorm(hidden_states)
        logits = self.predict_logits(hidden_states)
        return logits, raw_states


class ConceptLMV22VQOlmoAttention(nn.Module):
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
        # qkv proj -> (batch, seq, num_heads, head_dim) each
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.flatten(0, -2)  # (-1, heads_total, head_dim)
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


class ConceptLMV22VQOlmoMLP(nn.Module):
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


class ConceptLMV22VQOlmoLayer(nn.Module):
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
        self.self_attention = ConceptLMV22VQOlmoAttention(config,
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
        self.mlp = ConceptLMV22VQOlmoMLP(config, dtype=dtype, device=device, prefix=add_prefix('mlp', prefix))
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


class ConceptLMV22VQOlmoBlock(nn.Module):
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
            ConceptLMV22VQOlmoLayer(config,
                                    layer_idx,
                                    dtype=dtype,
                                    device=device,
                                    sliding_window=self._layer_sliding_window(layer_idx + 1, window_size,
                                                                              skip_frequency),
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
        if window_size is None:
            return None
        if skip_frequency is not None and layer_number % skip_frequency == 0:
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
        """Load native ConceptLM OLMo block weights into the LMDeploy rewrite."""
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


class ConceptLMV22VQForCausalLM(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    """Rewrote model of ConceptLMV22VQForCausalLM."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # token embedding — mirrors ``self.embedding`` in the reference.
        self.embedding = ConceptLMV22VQEmbedding(config, dtype=dtype, device=device)
        self.encoder = ConceptLMV22VQOlmoBlock(config,
                                               config.concept_encoder_layers,
                                               post_layer_norm=False,
                                               dtype=dtype,
                                               device=device,
                                               prefix=add_prefix('encoder', prefix))
        self.decoder = ConceptLMV22VQOlmoBlock(config,
                                               config.concept_decoder_layers,
                                               post_layer_norm=True,
                                               dtype=dtype,
                                               device=device,
                                               prefix=add_prefix('decoder', prefix))
        self.concept_vq_input_norm = nn.LayerNorm(config.hidden_size,
                                                  eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                                  dtype=dtype,
                                                  device=device)
        self.concept_quantizer = ConceptLMV22VQQuantizer(config, dtype=dtype, device=device)
        self.concept_predictor = ConceptLMV22VQConceptPredictor(config,
                                                                dtype=dtype,
                                                                device=device,
                                                                prefix=add_prefix('concept_predictor', prefix))
        self.concept_predictor.set_attention_window(tuple(getattr(config, 'window_size', (None, None))),
                                                    getattr(config, 'window_attn_skip_freq', None))
        self.fusion_tok_norm = nn.LayerNorm(config.hidden_size,
                                            eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                            dtype=dtype,
                                            device=device)
        self.fusion_hl_norm = nn.LayerNorm(config.hidden_size,
                                           eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                           dtype=dtype,
                                           device=device)
        self.fusion_norm_alpha = nn.Parameter(
            torch.tensor(getattr(config, 'concept_fusion_norm_alpha_init', 0.1), dtype=dtype, device=device),
            requires_grad=False)
        self.dd_encoder_self_dd = ConceptLMV22VQSelfDD(config,
                                                       config.concept_encoder_layers,
                                                       use_softmax=False,
                                                       dtype=dtype,
                                                       device=device)
        self.decoder_read_encoder_routes = nn.ModuleList([
            ConceptLMV22VQResidualRoute(config,
                                        config.concept_encoder_layers,
                                        use_softmax=True,
                                        dtype=dtype,
                                        device=device)
            for _ in range(config.concept_decoder_layers)
        ])
        self.decoder_read_encoder_shared_source_norm = nn.LayerNorm(config.hidden_size,
                                                                    eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                                                    dtype=dtype,
                                                                    device=device)
        self.decoder_read_concept_routes = nn.ModuleList([
            ConceptLMV22VQResidualRoute(config,
                                        config.concept_special_layers,
                                        use_softmax=True,
                                        dtype=dtype,
                                        device=device)
            for _ in range(config.concept_decoder_layers)
        ])
        self.decoder_read_concept_shared_source_norm = nn.LayerNorm(config.hidden_size,
                                                                    eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                                                    dtype=dtype,
                                                                    device=device)
        self.final_read_concept_gate_logits = nn.Parameter(
            torch.zeros(config.concept_decoder_layers, 2, dtype=dtype, device=device),
            requires_grad=False)
        self.dd_two_route_add = ConceptLMV22VQTwoRouteAdd(config, dtype=dtype, device=device)
        self.concept_ops = ConceptLMRuntimeOps()
        # output projection — mirrors ``output_layer`` in the reference. Built
        # via build_lm_head and named ``lm_head`` so DeployModelMixinV1.
        # get_logits picks it up directly; load_weights maps the checkpoint's
        # ``output_layer`` onto it.
        self.lm_head = self.build_lm_head(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]],
                attn_metadata=None,
                inputs_embeds: torch.Tensor = None,
                state_ids: torch.Tensor | None = None,
                state_caches: list[torch.Tensor] | None = None,
                **kwargs):
        """Model forward, return hidden_states (logits computed by runtime)."""
        concept_metadata = self._build_concept_metadata(position_ids, attn_metadata, state_ids)
        concept_caches = self._build_concept_caches(past_key_values, state_caches)
        _ = concept_metadata, concept_caches
        if inputs_embeds is None:
            # NOTE: placeholder. The real path is
            #   embed -> encoder -> concept(vq+predictor) -> fusion -> decoder
            # added incrementally. Returns raw embeddings as hidden_states.
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = inputs_embeds
        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embedding.word_embeddings

    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.lm_head

    def _split_past_key_values(self, past_key_values: list[list[torch.Tensor]] | None):
        """Split the flat LMDeploy KV-cache list into ConceptLM streams."""
        return _split_concept_past_key_values(self.config, past_key_values)

    def _build_concept_metadata(self,
                                position_ids: torch.Tensor | None,
                                attn_metadata: Any = None,
                                state_ids: torch.Tensor | None = None) -> ConceptMetadata:
        """Build top-level ConceptLM metadata for future serving paths."""
        return ConceptMetadata.build(
            self.config,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            state_ids=state_ids,
        )

    def _build_concept_caches(self,
                              past_key_values: list[list[torch.Tensor]] | None,
                              state_caches: list[torch.Tensor] | None = None) -> ConceptCaches:
        """Build top-level ConceptLM cache views for future serving paths."""
        return ConceptCaches.build(
            self.config,
            past_key_values=past_key_values,
            state_caches=state_caches,
        )

    def _decode_chunk_state_update(self,
                                   current_source_states: torch.Tensor,
                                   concept_metadata: ConceptMetadata,
                                   concept_caches: ConceptCaches) -> ConceptChunkStateUpdateResult:
        """Update decode chunk-source state and return fixed-shape rows.

        CUDA uses the Triton writer. CPU uses the reference writer for tests.
        The returned rows deliberately avoid dynamic concept-row compaction,
        matching the CUDA graph route in the design doc.
        """
        chunk_source_state = concept_caches.chunk_source_state
        if chunk_source_state is None:
            raise RuntimeError('ConceptLM decode chunk update requires concept chunk source state cache.')
        if concept_metadata.state_ids is None:
            raise RuntimeError('ConceptLM decode chunk update requires state_ids.')
        if concept_metadata.position_ids is None:
            raise RuntimeError('ConceptLM decode chunk update requires position_ids.')
        concept_input_states, next_rows, update_mask = self.concept_ops.decode_chunk_state_update(
            chunk_source_state,
            current_source_states,
            concept_metadata.state_ids,
            concept_metadata.position_ids,
            concept_metadata.chunk_size,
            concept_metadata.merge_method,
        )
        state_ids = concept_metadata.state_ids.to(device=current_source_states.device, dtype=torch.long)
        return ConceptChunkStateUpdateResult(
            concept_input_states=concept_input_states,
            next_chunk_source_states=next_rows,
            concept_update_mask=update_mask,
            valid_state_mask=state_ids >= 0,
            state_ids=state_ids,
        )

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Disable CUDA graph until ConceptLM decode state update is graph-safe.

        ``states_shapes`` makes the engine allocate graph-padded state ids. The
        current top-level ConceptLM helpers still include dense/reference-only
        chunk operations, so allowing the default decode graph capture would
        bake in the wrong execution contract.
        """
        return False

    def _merge_chunks(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reference chunk merge for dense ``[seq, batch, hidden]`` states."""
        assert hidden_states.dim() == 3, (
            f'_merge_chunks currently expects dense [seq, batch, hidden], got {tuple(hidden_states.shape)}.')
        seq_len, batch_size, hidden_size = hidden_states.shape
        chunk_size = int(self.config.concept_chunk_size)
        if seq_len < chunk_size:
            return hidden_states.mean(dim=0, keepdim=True)

        usable = (seq_len // chunk_size) * chunk_size
        chunks = hidden_states[:usable].reshape(usable // chunk_size, chunk_size, batch_size, hidden_size)
        merge_method = getattr(self.config, 'concept_chunk_merge_method', 'meanpooling')
        if merge_method == 'first':
            return chunks[:, 0]
        if merge_method == 'last':
            return chunks[:, -1]
        return chunks.mean(dim=1)

    def _repeat_shift(self, concept_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Repeat chunk-level concept states back to the token timeline."""
        chunk_size = int(self.config.concept_chunk_size)
        shifted = torch.cat((torch.zeros_like(concept_states[:1]), concept_states), dim=0)
        repeated = shifted.repeat_interleave(chunk_size, dim=0)[:int(seq_len) + 1]
        if bool(getattr(self.config, 'concept_shift_feature', True)):
            repeated = repeated[1:]
        else:
            repeated = repeated[:seq_len]
        if repeated.shape[0] < seq_len:
            pad_len = int(seq_len) - repeated.shape[0]
            repeated = torch.cat((repeated, repeated.new_zeros(pad_len, *repeated.shape[1:])), dim=0)
        return repeated[:seq_len]

    def _build_encoder_concept_states(self, encoder_raw_states: list[torch.Tensor]) -> torch.Tensor:
        """Build chunk-level encoder states used by the concept predictor."""
        chunks = [self._merge_chunks(state) for state in encoder_raw_states[:-1]]
        assert len(chunks) > 0, 'ConceptLM concept-read-encoder route requires at least one encoder source state.'
        states = torch.stack(chunks, dim=2)
        return self.concept_predictor.normalize_encoder_concept_states(states)

    def _route_gate(self, layer_idx: int) -> torch.Tensor:
        """Return decoder route gate ``[decoder_dd_scale, concept_route_scale]``."""
        return self.final_read_concept_gate_logits[int(layer_idx)].float().softmax(dim=-1)

    def _encode(self,
                hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]] | None = None,
                attn_metadata: Any = None):
        """Encoder stack plus encoder self-DD.

        This helper mirrors the reference flow but consumes LMDeploy attention
        inputs. It is wired for the future full forward path; continuous
        batching still needs caller-side concept metadata before the full model
        can use it safely.
        """
        raw_states = []
        history_buffer = self.dd_encoder_self_dd.make_history_buffer(hidden_states)
        self.dd_encoder_self_dd.write_history(history_buffer, 0, hidden_states)
        rotary_pos_emb = self.encoder._make_rotary_pos_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.encoder.layers):
            pkv = past_key_values[layer_idx] if past_key_values is not None else None
            raw = layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=pkv,
                attn_metadata=attn_metadata,
            )
            raw_states.append(raw)
            self.dd_encoder_self_dd.write_history(history_buffer, layer_idx + 1, raw)
            hidden_states = self.dd_encoder_self_dd.forward_from_buffer(layer_idx, raw, history_buffer)
        return hidden_states, raw_states

    def _decode(self,
                decoder_input: torch.Tensor,
                encoder_raw_states: list[torch.Tensor],
                final_concept_state: torch.Tensor,
                concept_raw_states: list[torch.Tensor],
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]] | None = None,
                attn_metadata: Any = None):
        """Decoder stack plus decoder DD and residual routes."""
        decoder_encoder_states = torch.stack(tuple(encoder_raw_states), dim=2)
        decoder_encoder_states = self.decoder_read_encoder_shared_source_norm(decoder_encoder_states)

        concept_states = torch.stack(tuple(concept_raw_states), dim=2)
        zero_chunk = torch.zeros_like(concept_states[:1])
        concept_states = torch.cat((zero_chunk, concept_states), dim=0)
        concept_states = self.decoder_read_concept_shared_source_norm(concept_states)

        chunk_size = int(self.config.concept_chunk_size)
        hidden_states = decoder_input
        history_buffer = self.dd_two_route_add.make_history_buffer(hidden_states)
        self.dd_two_route_add.write_history(history_buffer, 0, hidden_states)
        rotary_pos_emb = self.decoder._make_rotary_pos_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.decoder.layers):
            pkv = past_key_values[layer_idx] if past_key_values is not None else None
            raw = layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=pkv,
                attn_metadata=attn_metadata,
            )
            self.dd_two_route_add.write_history(history_buffer, layer_idx + 1, raw)
            gate = self._route_gate(layer_idx)
            hidden_states = self.dd_two_route_add.forward_from_buffer(
                layer_idx,
                raw,
                history_buffer,
                final_concept_state,
                gate[0],
            )
            hidden_states = self.decoder_read_encoder_routes[layer_idx](
                hidden_states,
                decoder_encoder_states,
                source_dim=2,
            )
            hidden_states = self.decoder_read_concept_routes[layer_idx].forward_repeated_chunks(
                hidden_states,
                concept_states,
                chunk_size,
                bool(getattr(self.config, 'concept_shift_feature', True)),
                residual_scale=gate[1],
                source_dim=2,
            )

        if self.decoder.final_layernorm is not None:
            hidden_states = self.decoder.final_layernorm(hidden_states)
        return hidden_states

    def prepare_inputs_for_generation(self,
                                      past_key_values: list[list[torch.Tensor]],
                                      inputs_embeds: torch.Tensor | None = None,
                                      context: StepContext = None):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
            state_caches=context.state_caches,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load native ConceptLM checkpoint weights into implemented modules."""
        # (checkpoint_name, target_name)
        weight_map = {
            'embedding.word_embeddings.weight': 'embedding.word_embeddings.weight',
            'output_layer.weight': 'lm_head.weight',
        }
        codebook_prefix = 'concept_quantizer.codebook.'
        prediction_head_prefix = 'concept_predictor.prediction_heads.'
        block_prefixes = (
            ('encoder.', self.encoder),
            ('decoder.', self.decoder),
            ('concept_predictor.hlm_block.', self.concept_predictor.hlm_block),
        )
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            loaded_by_block = False
            for block_prefix, block in block_prefixes:
                if name.startswith(block_prefix):
                    block.load_weights([(name, loaded_weight)], prefix=block_prefix[:-1])
                    loaded_by_block = True
                    break
            if loaded_by_block:
                continue
            if name.startswith(prediction_head_prefix):
                suffix = name[len(prediction_head_prefix):]
                parts = suffix.split('.')
                if len(parts) == 2 and parts[0].isdigit() and parts[1] in ('weight', 'bias'):
                    target_name = f'{prediction_head_prefix}proj.{parts[1]}'
                    param = params_dict.get(target_name)
                    if param is not None:
                        load_weight(param, loaded_weight, shard_id=int(parts[0]))
                    continue
                if suffix in ('proj.weight', 'proj.bias'):
                    param = params_dict.get(name)
                    if param is not None:
                        for shard_id, shard_weight in enumerate(param.weight_spliter(loaded_weight)):
                            load_weight(param, shard_weight, shard_id=shard_id)
                    continue
            if name.startswith(codebook_prefix):
                codebook_idx = int(name[len(codebook_prefix):])
                param = params_dict['concept_quantizer.codebook']
                _load_stacked_codebook_weight(param, loaded_weight, codebook_idx)
                continue
            target = weight_map.get(name)
            if target is None:
                target = name
            if target not in params_dict:
                # skip checkpoint metadata or tensors owned by future runtime-only paths
                continue
            param = params_dict[target]
            load_weight(param, loaded_weight)
