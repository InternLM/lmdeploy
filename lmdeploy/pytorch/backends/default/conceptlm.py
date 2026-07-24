# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from ..conceptlm import ConceptLMRuntimeOpsBuilder, ConceptLMRuntimeOpsImpl


def _flatten_decode_position_ids(position_ids: Tensor, batch_size: int, device: torch.device) -> Tensor:
    """Normalize decode position ids to one absolute position per batch row."""
    if position_ids.dim() == 0:
        position_ids = position_ids.view(1)
    if position_ids.dim() == 1:
        return position_ids.to(device=device, dtype=torch.long)
    position_ids = position_ids.reshape(-1)
    if position_ids.numel() == batch_size:
        return position_ids.to(device=device, dtype=torch.long)
    assert position_ids.numel() % batch_size == 0, (
        f'Cannot map position_ids with {position_ids.numel()} elements to batch size {batch_size}.')
    return position_ids.reshape(-1, batch_size)[-1].to(device=device, dtype=torch.long)


class DefaultConceptLMRuntimeOpsImpl(ConceptLMRuntimeOpsImpl):
    """Torch fallback implementation of ConceptLM runtime operations."""

    def decode_chunk_state_update(
        self,
        chunk_source_state_cache: Tensor,
        current_source_states: Tensor,
        state_ids: Tensor,
        position_ids: Tensor,
        chunk_size: int,
        merge_method: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update state cache and return fixed-shape decode rows."""
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
        position_ids = _flatten_decode_position_ids(position_ids, batch_size, current_source_states.device)
        assert position_ids.numel() == batch_size, (
            f'Expected {batch_size} decode position ids, got {position_ids.numel()}.')

        valid_state_mask = state_ids >= 0
        safe_state_ids = state_ids.clamp(min=0)
        previous_rows = chunk_source_state_cache.index_select(0, safe_state_ids)

        chunk_size = int(chunk_size)
        chunk_pos = torch.remainder(position_ids, chunk_size)
        update_mask = valid_state_mask & (torch.remainder(position_ids + 1, chunk_size) == 0)
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
        next_rows = torch.where(update_mask.view(batch_size, 1, 1), zero_rows, update_rows)
        next_rows = torch.where(valid_state_mask.view(batch_size, 1, 1), next_rows, previous_rows)
        concept_input_states = torch.where(update_mask.view(batch_size, 1, 1), concept_input_states, zero_rows)

        for batch_idx in range(batch_size):
            state_id = int(state_ids[batch_idx])
            if state_id >= 0:
                chunk_source_state_cache[state_id].copy_(next_rows[batch_idx])
        return concept_input_states, next_rows, update_mask


class DefaultConceptLMRuntimeOpsBuilder(ConceptLMRuntimeOpsBuilder):
    """Torch fallback ConceptLM runtime operation builder."""

    @staticmethod
    def build() -> ConceptLMRuntimeOpsImpl:
        """Build layer implementation."""
        return DefaultConceptLMRuntimeOpsImpl()
