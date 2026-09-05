# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig


@dataclass
class ConceptChunkInput:
    """Concept-stream input rows prepared from token-stream encoder states.

    ``source_states`` has a unified layout for prefill and decode:
    ``[concept_rows, num_sources, hidden]``. Source row 0 is the hidden state
    consumed by the concept predictor; rows 1: are encoder states consumed by
    ConceptRoute/SelfDD.
    """

    source_states: Tensor
    position_ids: Tensor
    attn_metadata: Any
    state_ids: Tensor | None = None
    update_mask: Tensor | None = None
    prefill_metadata: 'ConceptPrefillMetadata | None' = None
    decode_metadata: 'ConceptDecodeMetadata | None' = None

    @property
    def is_decoding(self) -> bool:
        """Whether this input came from the fixed-shape decode path."""
        return self.decode_metadata is not None


@dataclass
class ConceptForwardContext:
    """Temporary state needed around one concept-predictor forward."""

    saved_kv: list[tuple[Tensor, Tensor]] | None = None
    previous_final_state: Tensor | None = None
    previous_raw_states: Tensor | None = None


@dataclass
class ConceptDecoderInput:
    """Concept states consumed by the token decoder stack."""

    final_state: Tensor
    route_states: Tensor


@dataclass
class ConceptRuntimeCaches:
    """Backend-facing ConceptLM runtime cache views."""

    chunk_source_state: Tensor | None = None
    last_state: Tensor | None = None
    last_raw_states: Tensor | None = None
    last_final_state: Tensor | None = None
    concept_past_key_values: list[list[Tensor]] | None = None


@dataclass
class ConceptDecodeMetadata:
    """Fixed-layout decode metadata derived once from engine inputs."""

    position_ids: Tensor
    state_ids: Tensor
    safe_state_ids: Tensor
    valid_state_mask: Tensor


@dataclass
class ConceptPrefillMetadata:
    """Packed ConceptLM prefill runtime metadata.

    The model treats this as a backend-owned plan. Fields stay public for the current torch fallback path and tests, but
    model code should not rebuild or reinterpret this layout directly.
    """

    token_q_seqlens: Tensor
    token_q_start_loc: Tensor
    concept_q_seqlens: Tensor
    concept_q_start_loc: Tensor
    concept_position_ids: Tensor
    merge_token_to_concept: Tensor
    merge_token_start_ids: Tensor
    merge_token_counts: Tensor
    merge_first_token_ids: Tensor
    merge_last_token_ids: Tensor
    merge_short_concept_mask: Tensor
    token_to_concept: Tensor
    num_tokens_total: int
    num_concepts_total: int
    max_concepts_per_request: int


class ConceptLMRuntimeOpsImpl(ABC):
    """Backend contract for ConceptLM runtime/cache operations."""

    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.chunk_size = int(config.concept_chunk_size)
        self.merge_method = getattr(config, 'concept_chunk_merge_method', 'meanpooling')
        self.shift_feature = bool(getattr(config, 'concept_shift_feature', True))

    @abstractmethod
    def flatten_decode_position_ids(self, position_ids: Tensor, batch_size: int, device: torch.device) -> Tensor:
        """Normalize decode position ids to one absolute position per batch
        row."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def build_concept_chunk_input(
        self,
        source_states: Tensor,
        token_attn_metadata: Any,
        position_ids: Tensor,
        state_ids: Tensor | None = None,
        chunk_source_state_cache: Tensor | None = None,
    ) -> ConceptChunkInput:
        """Build concept-predictor source rows for prefill or decode."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def begin_concept_forward(self, chunk_input: ConceptChunkInput,
                              runtime_caches: ConceptRuntimeCaches) -> ConceptForwardContext:
        """Prepare transient state before the concept predictor forward."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def end_concept_forward(
        self,
        chunk_input: ConceptChunkInput,
        runtime_caches: ConceptRuntimeCaches,
        forward_context: ConceptForwardContext,
        source_states: Tensor,
        predicted_vectors: Tensor,
        concept_raw_states: list[Tensor],
    ) -> None:
        """Commit concept-predictor side effects for prefill or decode."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def build_decoder_concept_input(
        self,
        chunk_input: ConceptChunkInput,
        runtime_caches: ConceptRuntimeCaches,
        forward_context: ConceptForwardContext,
        predicted_vectors: Tensor,
        concept_raw_states: list[Tensor],
    ) -> ConceptDecoderInput:
        """Build token-decoder concept inputs for prefill or decode."""
        raise NotImplementedError('Not implemented.')


class ConceptLMRuntimeOpsBuilder(ABC):
    """ConceptLM runtime operation builder."""

    @staticmethod
    @abstractmethod
    def build(config: PretrainedConfig) -> ConceptLMRuntimeOpsImpl:
        """Build layer implementation."""
        raise NotImplementedError('Not implemented.')
