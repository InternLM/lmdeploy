# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda.conceptlm import (
    decode_chunk_state_update,
    decode_concept_state_update,
    decode_kv_cache_restore,
    decode_kv_cache_snapshot,
    prefill_chunk_state_update,
    prefill_state_cache_update,
)

from ..conceptlm import ConceptLMRuntimeOpsBuilder, ConceptLMRuntimeOpsImpl
from ..default.conceptlm import DefaultConceptLMRuntimeOpsImpl


class TritonConceptLMRuntimeOpsImpl(DefaultConceptLMRuntimeOpsImpl):
    """Triton implementation of ConceptLM runtime operations."""

    def merge_chunks_packed(self, hidden_states: Tensor, prefill_metadata) -> Tensor:
        """Merge packed token states into packed concept rows."""
        if not hidden_states.is_cuda:
            return super().merge_chunks_packed(hidden_states, prefill_metadata)
        source_states = hidden_states.unsqueeze(1)
        return self.prefill_chunk_state_update(source_states, prefill_metadata)[:, 0]

    def prefill_chunk_state_update(self, source_states: Tensor, prefill_metadata) -> Tensor:
        """Merge prefill source states to compact concept rows."""
        if not source_states.is_cuda:
            return super().prefill_chunk_state_update(source_states, prefill_metadata)
        if source_states.dim() == 2:
            source_states = source_states.unsqueeze(1)
            return prefill_chunk_state_update(
                source_states,
                prefill_metadata.merge_token_start_ids,
                prefill_metadata.merge_token_counts,
                prefill_metadata.num_concepts_total,
                self.chunk_size,
                self.merge_method,
            )[:, 0]
        return prefill_chunk_state_update(
            source_states,
            prefill_metadata.merge_token_start_ids,
            prefill_metadata.merge_token_counts,
            prefill_metadata.num_concepts_total,
            self.chunk_size,
            self.merge_method,
        )

    def _write_prefill_state_caches_impl(
        self,
        chunk_source_state: Tensor | None,
        last_raw_states: Tensor | None,
        last_final_state: Tensor | None,
        state_ids: Tensor | None,
        prefill_metadata,
        source_states: Tensor,
        predicted_vectors: Tensor,
        concept_raw_states: list[Tensor],
    ) -> None:
        """Seed decode state caches from a completed CUDA prefill forward."""
        if state_ids is None:
            return
        if chunk_source_state is None or last_raw_states is None or last_final_state is None:
            return
        if not (source_states.is_cuda and chunk_source_state.is_cuda and last_raw_states.is_cuda
                and last_final_state.is_cuda):
            return super()._write_prefill_state_caches_impl(
                chunk_source_state,
                last_raw_states,
                last_final_state,
                state_ids,
                prefill_metadata,
                source_states,
                predicted_vectors,
                concept_raw_states,
            )

        raw_rows = self.stack_concept_raw_states(concept_raw_states)
        return prefill_state_cache_update(
            chunk_source_state,
            last_raw_states,
            last_final_state,
            source_states,
            predicted_vectors,
            raw_rows,
            state_ids,
            prefill_metadata.token_q_start_loc,
            prefill_metadata.token_q_seqlens,
            prefill_metadata.concept_q_start_loc,
            prefill_metadata.concept_q_seqlens,
            self.chunk_size,
            self.merge_method,
        )

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
