# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda.conceptlm import (
    decode_chunk_state_update,
    decode_concept_state_update,
    decode_kv_cache_restore,
    decode_kv_cache_snapshot,
)

from ..conceptlm import ConceptLMRuntimeOpsBuilder, ConceptLMRuntimeOpsImpl


class TritonConceptLMRuntimeOpsImpl(ConceptLMRuntimeOpsImpl):
    """Triton implementation of ConceptLM runtime operations."""

    def decode_chunk_state_update(
        self,
        chunk_source_state_cache: Tensor,
        current_source_states: Tensor,
        state_ids: Tensor,
        position_ids: Tensor,
        chunk_size: int,
        merge_method: str,
    ) -> tuple[Tensor, Tensor]:
        """Update state cache and return fixed-shape concept inputs."""
        return decode_chunk_state_update(
            chunk_source_state_cache,
            current_source_states,
            state_ids,
            position_ids,
            chunk_size,
            merge_method,
        )

    def decode_kv_cache_snapshot(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        block_offsets: Tensor,
        kv_seqlens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Snapshot one decode KV slot per batch row."""
        return decode_kv_cache_snapshot(k_cache, v_cache, block_offsets, kv_seqlens)

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
        return decode_kv_cache_restore(k_cache, v_cache, saved_k, saved_v, block_offsets, kv_seqlens, restore_mask)

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
        return decode_concept_state_update(
            last_raw_state_cache,
            last_final_state_cache,
            predicted_vectors,
            raw_states,
            state_ids,
            update_mask,
        )


class TritonConceptLMRuntimeOpsBuilder(ConceptLMRuntimeOpsBuilder):
    """Triton ConceptLM runtime operation builder."""

    @staticmethod
    def build(config) -> ConceptLMRuntimeOpsImpl:
        """Build layer implementation."""
        return TritonConceptLMRuntimeOpsImpl(config)
