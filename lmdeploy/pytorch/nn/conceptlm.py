# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from torch import Tensor, nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.conceptlm import ConceptDecodeMetadata, ConceptPrefillMetadata


class ConceptLMRuntimeOps(nn.Module):
    """ConceptLM model-specific runtime operation wrapper.

    The model calls this nn module only. Backend implementations own dispatch, and CUDA implementations own direct
    Triton kernel launchers.
    """

    def __init__(self, config):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.ConceptLMRuntimeOps)
        self.impl = builder.build(config)

    def flatten_decode_position_ids(self, position_ids: Tensor, batch_size: int, device: torch.device) -> Tensor:
        """Normalize decode position ids to one absolute position per batch
        row."""
        return self.impl.flatten_decode_position_ids(position_ids, batch_size, device)

    def build_decode_metadata(self,
                              position_ids: Tensor,
                              state_ids: Tensor | None,
                              batch_size: int,
                              device: torch.device) -> ConceptDecodeMetadata:
        """Build fixed-shape decode metadata from engine state ids."""
        return self.impl.build_decode_metadata(position_ids, state_ids, batch_size, device)

    def select_decode_last_state_rows(self,
                                      last_state: Tensor | None,
                                      last_final_state: Tensor,
                                      last_raw_states: Tensor,
                                      decode_metadata: ConceptDecodeMetadata) -> tuple[Tensor, Tensor]:
        """Gather latest concept state rows for decode."""
        return self.impl.select_decode_last_state_rows(
            last_state,
            last_final_state,
            last_raw_states,
            decode_metadata,
        )

    def decode_concept_read_mask(self, decode_metadata: ConceptDecodeMetadata) -> Tensor:
        """Return rows whose current decode token should read a cached
        concept."""
        return self.impl.decode_concept_read_mask(decode_metadata)

    def build_concept_decode_metadata_static(self, token_attn_metadata: Any,
                                             decode_metadata: ConceptDecodeMetadata):
        """Build fixed-shape concept-stream decode metadata."""
        return self.impl.build_concept_decode_metadata_static(token_attn_metadata, decode_metadata)

    def build_prefill_metadata(self, token_attn_metadata: Any, position_ids: Tensor) -> ConceptPrefillMetadata:
        """Build packed token-to-concept metadata for batched prefill."""
        return self.impl.build_prefill_metadata(token_attn_metadata, position_ids)

    def merge_chunks_packed(self, hidden_states: Tensor, prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Merge packed token states into compact concept rows."""
        return self.impl.merge_chunks_packed(hidden_states, prefill_metadata)

    def repeat_shift_packed(self, concept_states: Tensor, prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Gather compact concept states back to packed token rows."""
        return self.impl.repeat_shift_packed(concept_states, prefill_metadata)

    def repeat_shift_source_states_packed(self, concept_states_with_zero: Tensor,
                                          prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Gather zero-prefixed compact concept route states to token rows."""
        return self.impl.repeat_shift_source_states_packed(concept_states_with_zero, prefill_metadata)

    def build_concept_prefill_metadata(self, token_attn_metadata: Any, prefill_metadata: ConceptPrefillMetadata):
        """Build chunk-stream attention metadata for packed prefill."""
        return self.impl.build_concept_prefill_metadata(token_attn_metadata, prefill_metadata)

    def write_prefill_state_caches_eager(
        self,
        chunk_source_state: Tensor | None,
        last_raw_states: Tensor | None,
        last_final_state: Tensor | None,
        state_ids: Tensor | None,
        prefill_metadata: ConceptPrefillMetadata,
        source_states: Tensor,
        predicted_vectors: Tensor,
        concept_raw_states: list[Tensor],
    ) -> None:
        """Seed decode state caches from a completed prefill forward."""
        return self.impl.write_prefill_state_caches_eager(
            chunk_source_state,
            last_raw_states,
            last_final_state,
            state_ids,
            prefill_metadata,
            source_states,
            predicted_vectors,
            concept_raw_states,
        )

    def stack_concept_raw_states(self, concept_raw_states: list[Tensor]) -> Tensor:
        """Stack raw concept-layer states."""
        return self.impl.stack_concept_raw_states(concept_raw_states)

    def snapshot_decode_concept_kv(self, concept_past_key_values: list[list[Tensor]],
                                   concept_attn_metadata: Any) -> list[tuple[Tensor, Tensor]]:
        """Snapshot concept KV slots that dummy rows may overwrite."""
        return self.impl.snapshot_decode_concept_kv(concept_past_key_values, concept_attn_metadata)

    def restore_decode_concept_kv(self, concept_past_key_values: list[list[Tensor]], concept_attn_metadata: Any,
                                  saved_kv: list[tuple[Tensor, Tensor]], restore_mask: Tensor) -> None:
        """Restore concept KV slots for dummy rows."""
        return self.impl.restore_decode_concept_kv(
            concept_past_key_values,
            concept_attn_metadata,
            saved_kv,
            restore_mask,
        )

    def write_decode_concept_states(
        self,
        last_raw_state_cache: Tensor,
        last_final_state_cache: Tensor,
        predicted_vectors: Tensor,
        raw_states: list[Tensor],
        state_ids: Tensor,
        update_mask: Tensor,
    ) -> None:
        """Write newly emitted decode concept states."""
        return self.impl.write_decode_concept_states(
            last_raw_state_cache,
            last_final_state_cache,
            predicted_vectors,
            raw_states,
            state_ids,
            update_mask,
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
        """Update state cache and return concept inputs plus update mask."""
        return self.impl.decode_chunk_state_update(
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
        return self.impl.decode_kv_cache_snapshot(
            k_cache,
            v_cache,
            block_offsets,
            kv_seqlens,
        )

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
        return self.impl.decode_kv_cache_restore(
            k_cache,
            v_cache,
            saved_k,
            saved_v,
            block_offsets,
            kv_seqlens,
            restore_mask,
        )

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
        return self.impl.decode_concept_state_update(
            last_raw_state_cache,
            last_final_state_cache,
            predicted_vectors,
            raw_states,
            state_ids,
            update_mask,
        )
