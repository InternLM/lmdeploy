# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import Tensor


class ConceptLMRuntimeOpsImpl(ABC):
    """ConceptLM runtime operation implementation.

    Model-specific runtime/cache operations live behind this single backend interface. That keeps model code free from
    direct kernel calls while avoiding one OpType/nn module per small ConceptLM state operation.
    """

    @abstractmethod
    def decode_chunk_state_update(
        self,
        chunk_source_state_cache: Tensor,
        current_source_states: Tensor,
        state_ids: Tensor,
        position_ids: Tensor,
        chunk_size: int,
        merge_method: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update state cache and return concept inputs, next rows, and
        mask."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def decode_kv_cache_snapshot(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        block_offsets: Tensor,
        kv_seqlens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Snapshot one decode KV slot per batch row."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def decode_kv_cache_restore(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        saved_k: Tensor,
        saved_v: Tensor,
        block_offsets: Tensor,
        kv_seqlens: Tensor,
        restore_mask: Tensor,
    ) -> None:
        """Restore one decode KV slot for masked batch rows."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def decode_concept_state_update(
        self,
        last_raw_state_cache: Tensor,
        last_final_state_cache: Tensor,
        predicted_vectors: Tensor,
        raw_states: Tensor,
        state_ids: Tensor,
        update_mask: Tensor,
    ) -> None:
        """Write final/raw concept states for masked decode rows."""
        raise NotImplementedError('Not implemented.')


class ConceptLMRuntimeOpsBuilder(ABC):
    """ConceptLM runtime operation builder."""

    @staticmethod
    @abstractmethod
    def build() -> ConceptLMRuntimeOpsImpl:
        """Build layer implementation."""
        raise NotImplementedError('Not implemented.')
