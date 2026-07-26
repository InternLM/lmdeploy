# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

_CONCEPT_STATE_CHUNK_SOURCE_NAME = 'concept_chunk_source_state'
_CONCEPT_STATE_LAST_NAME = 'concept_last_state'
_CONCEPT_STATE_LAST_RAW_NAME = 'concept_last_raw_states'
_CONCEPT_STATE_LAST_FINAL_NAME = 'concept_last_final_state'


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
    named_state_caches: Mapping[str, torch.Tensor] | None = None
    state_caches: list[torch.Tensor] | None = None
    chunk_source_name: str = _CONCEPT_STATE_CHUNK_SOURCE_NAME
    last_state_name: str = _CONCEPT_STATE_LAST_NAME
    last_raw_name: str = _CONCEPT_STATE_LAST_RAW_NAME
    last_final_name: str = _CONCEPT_STATE_LAST_FINAL_NAME
    chunk_source_idx: int = 0
    last_state_idx: int = 1
    last_raw_idx: int = 1
    last_final_idx: int = 2

    @classmethod
    def build(cls,
              config: PretrainedConfig,
              past_key_values: list[list[torch.Tensor]] | None = None,
              state_caches: list[torch.Tensor] | None = None,
              named_state_caches: Mapping[str, torch.Tensor] | None = None):
        """Build ConceptLM cache views from engine-provided caches."""
        encoder_past_key_values, concept_past_key_values, decoder_past_key_values = (
            _split_concept_past_key_values(config, past_key_values))
        state_names = tuple(getattr(config, 'concept_state_names', ()))

        def _find_state_idx(state_name: str, fallback: int) -> int:
            try:
                return state_names.index(state_name)
            except ValueError:
                return fallback

        chunk_source_idx = int(
            getattr(config, 'concept_state_chunk_source_idx', _find_state_idx(_CONCEPT_STATE_CHUNK_SOURCE_NAME, 0)))
        last_state_idx = int(getattr(config, 'concept_state_last_idx', _find_state_idx(_CONCEPT_STATE_LAST_NAME, -1)))
        last_raw_idx = int(getattr(config, 'concept_state_last_raw_idx',
                                   _find_state_idx(_CONCEPT_STATE_LAST_RAW_NAME, 1)))
        last_final_idx = int(
            getattr(config, 'concept_state_last_final_idx', _find_state_idx(_CONCEPT_STATE_LAST_FINAL_NAME, 2)))

        def _state_name(state_idx: int, fallback: str) -> str:
            if 0 <= state_idx < len(state_names):
                return str(state_names[state_idx])
            return fallback

        return cls(
            encoder_past_key_values=encoder_past_key_values,
            concept_past_key_values=concept_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
            named_state_caches=named_state_caches,
            state_caches=state_caches,
            chunk_source_name=_state_name(chunk_source_idx, _CONCEPT_STATE_CHUNK_SOURCE_NAME),
            last_state_name=_state_name(last_state_idx, _CONCEPT_STATE_LAST_NAME),
            last_raw_name=_state_name(last_raw_idx, _CONCEPT_STATE_LAST_RAW_NAME),
            last_final_name=_state_name(last_final_idx, _CONCEPT_STATE_LAST_FINAL_NAME),
            chunk_source_idx=chunk_source_idx,
            last_state_idx=last_state_idx,
            last_raw_idx=last_raw_idx,
            last_final_idx=last_final_idx,
        )

    def named_state_cache(self, state_name: str) -> torch.Tensor | None:
        """Return one named state-cache tensor when the engine provides it."""
        if self.named_state_caches is None or state_name not in self.named_state_caches:
            return None
        return self.named_state_caches[state_name]

    def state_cache(self, state_idx: int) -> torch.Tensor | None:
        """Return one anonymous state-cache tensor by semantic index."""
        if self.state_caches is None:
            return None
        if state_idx < 0 or state_idx >= len(self.state_caches):
            return None
        return self.state_caches[state_idx]

    def semantic_state_cache(self, state_name: str, state_idx: int) -> torch.Tensor | None:
        """Return a state cache by stable name, falling back to legacy
        index."""
        cache = self.named_state_cache(state_name)
        if cache is not None:
            return cache
        return self.state_cache(state_idx)

    @property
    def chunk_source_state(self) -> torch.Tensor | None:
        """Current chunk source accumulator state cache."""
        return self.semantic_state_cache(self.chunk_source_name, self.chunk_source_idx)

    @property
    def last_state(self) -> torch.Tensor | None:
        """Packed latest concept state cache.

        Shape is ``[num_state_slots, 1 + concept_layers, hidden]``. Row 0 is
        the final concept vector; rows 1: are raw concept-layer states.
        """
        return self.semantic_state_cache(self.last_state_name, self.last_state_idx)

    @property
    def last_raw_states(self) -> torch.Tensor | None:
        """Latest raw concept-layer state cache."""
        last_state = self.last_state
        if last_state is not None:
            return last_state[:, 1:]
        return self.semantic_state_cache(self.last_raw_name, self.last_raw_idx)

    @property
    def last_final_state(self) -> torch.Tensor | None:
        """Latest final concept vector state cache."""
        last_state = self.last_state
        if last_state is not None:
            return last_state[:, 0]
        return self.semantic_state_cache(self.last_final_name, self.last_final_idx)


@dataclass
class ConceptChunkStateUpdateResult:
    """Fixed-shape result of one decode chunk-source state update."""

    concept_input_states: torch.Tensor
    concept_update_mask: torch.Tensor


@dataclass
class ConceptDecodeMetadata:
    """Fixed-layout decode metadata derived once from engine inputs.

    Decode is always represented as the engine's fixed ``[1, batch]`` token
    layout at the model boundary and flattened to ``[batch]`` / ``[batch, H]``
    only inside ConceptLM helpers.
    """

    position_ids: torch.Tensor
    state_ids: torch.Tensor
    safe_state_ids: torch.Tensor
    valid_state_mask: torch.Tensor


@dataclass
class ConceptPrefillMetadata:
    """Packed prefill metadata derived once from token attention metadata.

    Field groups:
      - token stream: original engine-provided token request boundaries.
      - concept stream: compact chunk-token request boundaries and positions
        used by the concept predictor attention.
      - chunk merge: token -> concept ids and helper ids used to reduce encoder
        token states into concept states without a Python batch loop.
      - repeat/gather: concept -> token ids used to project compact concept
        states back to the packed token stream.
      - scalar bounds: eager compact sizes / upper bounds needed by metadata and
        attention launch parameters.
    """

    # Token stream metadata, shape [batch]. This is the original packed prefill
    # layout consumed by normal token attention.
    token_q_seqlens: torch.Tensor
    token_q_start_loc: torch.Tensor

    # Concept stream metadata, shape [batch] plus compact concept positions.
    # These describe the shorter chunk-token stream consumed by concept
    # predictor attention.
    concept_q_seqlens: torch.Tensor
    concept_q_start_loc: torch.Tensor
    concept_position_ids: torch.Tensor

    # Chunk merge metadata. ``merge_token_to_concept`` maps each packed token to
    # the compact concept row that owns it, or -1 when the token is dropped from
    # concept production. Counts/first/last ids implement mean/first/last merge.
    merge_token_to_concept: torch.Tensor
    merge_token_counts: torch.Tensor
    merge_first_token_ids: torch.Tensor
    merge_last_token_ids: torch.Tensor
    merge_short_concept_mask: torch.Tensor

    # Repeat/gather metadata. Maps each packed token row to the compact concept
    # row it should read after shift semantics are applied, or -1 for the
    # zero-concept row.
    token_to_concept: torch.Tensor

    # Scalar sizes/bounds. ``num_concepts_total`` is the exact compact size in
    # eager prefill; ``max_concepts_per_request`` is the per-request attention
    # launch bound.
    num_tokens_total: int
    num_concepts_total: int
    max_concepts_per_request: int


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
