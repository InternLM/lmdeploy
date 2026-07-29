# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..conceptlm import (
    ConceptChunkInput,
    ConceptDecodeMetadata,
    ConceptDecoderInput,
    ConceptForwardContext,
    ConceptLMRuntimeOpsBuilder,
    ConceptLMRuntimeOpsImpl,
    ConceptPrefillMetadata,
    ConceptRuntimeCaches,
)


@dataclass
class _PrefillTokenLayout:
    """Packed token-stream layout derived from prefill attention metadata."""

    q_seqlens: Tensor
    q_start_loc: Tensor
    q_seqlens_long: Tensor
    q_start_loc_long: Tensor
    token_seq: Tensor
    token_pos: Tensor
    total_tokens: int


@dataclass
class _PrefillConceptLayout:
    """Compact chunk-token stream layout used by concept predictor prefill."""

    q_seqlens: Tensor
    q_seqlens_long: Tensor
    q_start_loc: Tensor
    q_start_loc_long: Tensor
    seq: Tensor
    local_ids: Tensor
    position_ids: Tensor
    num_total: int
    max_per_request: int


@dataclass
class _PrefillMergeLayout:
    """Token-to-concept merge metadata for compact prefill."""

    token_to_concept: Tensor
    token_start_ids: Tensor
    token_counts: Tensor
    first_token_ids: Tensor
    last_token_ids: Tensor
    short_concept_mask: Tensor


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

    @staticmethod
    def concept_count_from_seq_len(seq_len: int, chunk_size: int) -> int:
        """Return reference ConceptLM chunk count for one request length."""
        seq_len = int(seq_len or 0)
        if seq_len <= 0:
            return 0
        if seq_len < chunk_size:
            return 1
        return seq_len // chunk_size

    @staticmethod
    def concept_counts_from_q_seqlens(q_seqlens: Tensor, chunk_size: int) -> Tensor:
        """Vectorized version of ``concept_count_from_seq_len``."""
        counts = torch.div(q_seqlens, chunk_size, rounding_mode='floor').clamp(min=1)
        return torch.where(q_seqlens > 0, counts, torch.zeros_like(q_seqlens))

    @staticmethod
    def repeat_slot_ids(token_pos: Tensor, chunk_size: int, shift_feature: bool) -> Tensor:
        """Return local concept slot read by each token after shift
        semantics."""
        if shift_feature:
            return torch.div(token_pos + 1, chunk_size, rounding_mode='floor') - 1
        return torch.div(token_pos, chunk_size, rounding_mode='floor') - 1

    def flatten_decode_position_ids(self, position_ids: Tensor, batch_size: int, device: torch.device) -> Tensor:
        """Normalize decode position ids to one absolute position per batch
        row."""
        return _flatten_decode_position_ids(position_ids, batch_size, device)

    def build_decode_metadata(self,
                              position_ids: Tensor,
                              state_ids: Tensor | None,
                              batch_size: int,
                              device: torch.device) -> ConceptDecodeMetadata:
        """Build fixed-shape decode metadata from engine state ids."""
        if state_ids is None:
            raise RuntimeError('ConceptLM decode requires state_ids.')
        state_ids = state_ids.to(device=device, dtype=torch.long).reshape(-1)
        if state_ids.numel() != batch_size:
            raise ValueError(f'Expected {batch_size} decode state ids, got {state_ids.numel()}.')
        valid_state_mask = state_ids >= 0
        return ConceptDecodeMetadata(
            position_ids=position_ids,
            state_ids=state_ids,
            safe_state_ids=state_ids.clamp(min=0),
            valid_state_mask=valid_state_mask,
        )

    @staticmethod
    def select_decode_state_rows(state_cache: Tensor, decode_metadata: ConceptDecodeMetadata) -> Tensor:
        """Gather state-cache rows and zero out padded decode rows."""
        rows = state_cache.index_select(0, decode_metadata.safe_state_ids)
        mask_shape = (decode_metadata.valid_state_mask.size(0), ) + (1, ) * (rows.dim() - 1)
        valid_mask = decode_metadata.valid_state_mask.view(mask_shape)
        return torch.where(valid_mask, rows, torch.zeros_like(rows))

    def select_decode_last_state_rows(self,
                                      last_state: Tensor | None,
                                      last_final_state: Tensor | None,
                                      last_raw_states: Tensor | None,
                                      decode_metadata: ConceptDecodeMetadata) -> tuple[Tensor, Tensor]:
        """Gather packed last-concept state rows once and return final/raw
        views."""
        if last_state is not None:
            rows = self.select_decode_state_rows(last_state, decode_metadata)
            return rows[:, 0], rows[:, 1:]
        if last_final_state is None or last_raw_states is None:
            raise RuntimeError('ConceptLM decode requires cached last concept states.')

        return (
            self.select_decode_state_rows(last_final_state, decode_metadata),
            self.select_decode_state_rows(last_raw_states, decode_metadata),
        )

    def decode_concept_read_mask(self, decode_metadata: ConceptDecodeMetadata) -> Tensor:
        """Return rows whose current decode token should read a cached
        concept."""
        repeat_slots = self.repeat_slot_ids(
            decode_metadata.position_ids,
            self.chunk_size,
            self.shift_feature,
        )
        return decode_metadata.valid_state_mask & (repeat_slots >= 0)

    def build_concept_decode_metadata_static(self, token_attn_metadata: Any,
                                             decode_metadata: ConceptDecodeMetadata):
        """Build fixed-shape concept-stream decode metadata."""
        device = decode_metadata.position_ids.device
        batch_size = decode_metadata.position_ids.numel()
        q_seqlens = token_attn_metadata.q_seqlens
        q_start_loc = token_attn_metadata.q_start_loc
        kv_seqlens = token_attn_metadata.kv_seqlens
        q_dtype = q_seqlens.dtype
        q_start_dtype = q_start_loc.dtype
        kv_dtype = kv_seqlens.dtype

        concept_q_seqlens = torch.ones((batch_size, ), dtype=q_dtype, device=device)
        concept_q_start_loc = torch.arange(batch_size, dtype=q_start_dtype, device=device)
        concept_cu_seqlens = F.pad(torch.cumsum(concept_q_seqlens, dim=0, dtype=torch.int32), (1, 0))
        concept_kv_seqlens = torch.div(
            decode_metadata.position_ids + 1,
            self.chunk_size,
            rounding_mode='floor',
        ).clamp(min=1).to(dtype=kv_dtype)

        updates = dict(
            is_decoding=True,
            block_offsets=token_attn_metadata.block_offsets,
            q_start_loc=concept_q_start_loc,
            q_seqlens=concept_q_seqlens,
            kv_seqlens=concept_kv_seqlens,
            cu_seqlens_q=concept_cu_seqlens,
            cu_seqlens_k=concept_cu_seqlens,
        )
        if hasattr(token_attn_metadata, 'kv_start_loc'):
            updates['kv_start_loc'] = concept_kv_seqlens - concept_q_seqlens.to(dtype=concept_kv_seqlens.dtype)
        if hasattr(token_attn_metadata, 'kv_flatten_size'):
            updates['kv_flatten_size'] = batch_size
        if hasattr(token_attn_metadata, 'max_q_seqlen'):
            updates['max_q_seqlen'] = 1
        if hasattr(token_attn_metadata, 'max_kv_seqlen'):
            updates['max_kv_seqlen'] = getattr(token_attn_metadata, 'max_kv_seqlen')
        if hasattr(token_attn_metadata, 'scheduler_metadata'):
            updates['scheduler_metadata'] = None
        if hasattr(token_attn_metadata, 'tile_scheduler_metadata'):
            updates['tile_scheduler_metadata'] = None
        if hasattr(token_attn_metadata, 'num_splits'):
            updates['num_splits'] = None
        if hasattr(token_attn_metadata, 'fill_seqlens'):
            updates['fill_seqlens'] = concept_q_seqlens
        return replace(token_attn_metadata, **updates)

    def build_concept_chunk_input(
        self,
        source_states: Tensor,
        token_attn_metadata: Any,
        position_ids: Tensor,
        state_ids: Tensor | None = None,
        chunk_source_state_cache: Tensor | None = None,
    ) -> ConceptChunkInput:
        """Build concept-predictor source rows for prefill or decode.

        The caller provides source states in the same layout for both phases:
        ``[token_or_batch, num_sources, hidden]``. Runtime metadata chooses the
        compact prefill merge or fixed-shape decode accumulator update.
        """
        if getattr(token_attn_metadata, 'is_decoding', False):
            if chunk_source_state_cache is None:
                raise RuntimeError('ConceptLM decode requires chunk source state cache.')
            batch_size = source_states.size(0)
            position_ids = self.flatten_decode_position_ids(position_ids, batch_size, source_states.device)
            decode_metadata = self.build_decode_metadata(
                position_ids,
                state_ids,
                batch_size,
                source_states.device,
            )
            concept_states, update_mask = self.decode_chunk_state_update(
                chunk_source_state_cache,
                source_states,
                decode_metadata.state_ids,
                decode_metadata.position_ids,
                self.chunk_size,
                self.merge_method,
            )
            concept_attn_metadata = self.build_concept_decode_metadata_static(token_attn_metadata, decode_metadata)
            concept_position_ids = self.decode_concept_position_ids(decode_metadata.position_ids)
            return ConceptChunkInput(
                source_states=concept_states,
                position_ids=concept_position_ids,
                attn_metadata=concept_attn_metadata,
                state_ids=decode_metadata.state_ids,
                update_mask=update_mask,
                decode_metadata=decode_metadata,
            )

        prefill_metadata = self.build_prefill_metadata(token_attn_metadata, position_ids)
        concept_states = self.prefill_chunk_state_update(source_states, prefill_metadata)
        concept_attn_metadata = self.build_concept_prefill_metadata(token_attn_metadata, prefill_metadata)
        return ConceptChunkInput(
            source_states=concept_states,
            position_ids=prefill_metadata.concept_position_ids,
            attn_metadata=concept_attn_metadata,
            state_ids=state_ids,
            prefill_metadata=prefill_metadata,
        )

    def decode_concept_position_ids(self, position_ids: Tensor) -> Tensor:
        """Return compressed-timeline RoPE positions for decode concept rows."""
        concept_index = torch.div(
            position_ids + 1,
            self.chunk_size,
            rounding_mode='floor',
        ) - 1
        return concept_index.clamp(min=0)

    def _get_max_concepts_per_request(self, token_attn_metadata: Any, concept_q_seqlens: Tensor) -> int:
        """Return per-request concept attention bound without hidden context
        access."""
        max_q_seqlen = getattr(token_attn_metadata, 'max_q_seqlen', None)
        if max_q_seqlen is not None:
            return self.concept_count_from_seq_len(int(max_q_seqlen), self.chunk_size)

        # Test/direct-call fallback. Serving should use the scheduler-provided
        # Python max_q_seqlen above, as in DSV4 metadata construction.
        return int(concept_q_seqlens.max().item()) if concept_q_seqlens.numel() > 0 else 0

    @staticmethod
    def _build_prefill_token_layout(token_attn_metadata: Any, position_ids: Tensor) -> _PrefillTokenLayout:
        """Build packed token-stream positions from prefill attention
        metadata."""
        q_seqlens = token_attn_metadata.q_seqlens
        q_start_loc = getattr(token_attn_metadata, 'q_start_loc', None)
        if q_start_loc is None:
            q_start_loc = F.pad(torch.cumsum(q_seqlens, dim=0, dtype=torch.int32), (1, 0))[:-1]

        total_tokens = int(position_ids.numel())

        q_seqlens_long = q_seqlens.to(dtype=torch.long, device=position_ids.device)
        q_start_loc_long = q_start_loc.to(dtype=torch.long, device=position_ids.device)
        cu_q_seqlens = getattr(token_attn_metadata, 'cu_seqlens_q', None)
        if cu_q_seqlens is None:
            cu_q_seqlens = F.pad(torch.cumsum(q_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_q_seqlens_long = cu_q_seqlens.to(dtype=torch.long, device=position_ids.device)

        token_ids = torch.arange(total_tokens, dtype=torch.long, device=position_ids.device)
        token_seq = torch.searchsorted(cu_q_seqlens_long[1:], token_ids, right=True)
        token_pos = token_ids - cu_q_seqlens_long[token_seq]

        return _PrefillTokenLayout(
            q_seqlens=q_seqlens,
            q_start_loc=q_start_loc,
            q_seqlens_long=q_seqlens_long,
            q_start_loc_long=q_start_loc_long,
            token_seq=token_seq,
            token_pos=token_pos,
            total_tokens=total_tokens,
        )

    def _build_prefill_concept_layout(self, token_attn_metadata: Any, position_ids: Tensor,
                                      token_layout: _PrefillTokenLayout) -> _PrefillConceptLayout:
        """Build compact chunk-token positions used by concept attention."""
        concept_q_seqlens_long = self.concept_counts_from_q_seqlens(token_layout.q_seqlens_long, self.chunk_size)
        concept_q_seqlens = concept_q_seqlens_long.to(dtype=token_layout.q_seqlens.dtype,
                                                      device=token_layout.q_seqlens.device)
        concept_q_start_loc = F.pad(torch.cumsum(concept_q_seqlens, dim=0, dtype=torch.int32), (1, 0))[:-1]
        concept_q_start_loc_long = concept_q_start_loc.to(dtype=torch.long, device=position_ids.device)
        concept_cu_seqlens_long = F.pad(torch.cumsum(concept_q_seqlens_long, dim=0), (1, 0))

        # TODO: remove this eager compact-size scalar read by moving ConceptLM
        # concept-stream allocation/compaction into the engine/backend contract,
        # like DSV4's precomputed metadata and Qwen3.5's state-cache metadata.
        num_concepts_total = int(concept_cu_seqlens_long[-1].item())
        concept_ids = torch.arange(num_concepts_total, dtype=torch.long, device=position_ids.device)
        concept_seq = torch.searchsorted(concept_cu_seqlens_long[1:], concept_ids, right=True)
        local_concept_ids = concept_ids - concept_cu_seqlens_long[concept_seq]
        concept_token_start = token_layout.q_start_loc_long[concept_seq] + local_concept_ids * self.chunk_size
        # DCP/Megatron V21 builds HLM rotary embeddings on the compressed
        # concept timeline, not on the original token timeline.  For ordinary
        # full-prompt prefill this yields 0, 1, 2, ...; deriving it from token
        # position ids keeps non-zero-offset chunks on the same absolute
        # concept index.
        concept_position_ids = torch.div(
            position_ids[concept_token_start].to(dtype=torch.long),
            self.chunk_size,
            rounding_mode='floor',
        )
        max_concepts_per_request = self._get_max_concepts_per_request(token_attn_metadata, concept_q_seqlens_long)

        return _PrefillConceptLayout(
            q_seqlens=concept_q_seqlens,
            q_seqlens_long=concept_q_seqlens_long,
            q_start_loc=concept_q_start_loc,
            q_start_loc_long=concept_q_start_loc_long,
            seq=concept_seq,
            local_ids=local_concept_ids,
            position_ids=concept_position_ids,
            num_total=num_concepts_total,
            max_per_request=max_concepts_per_request,
        )

    def _build_prefill_repeat_ids(self, token_layout: _PrefillTokenLayout,
                                  concept_layout: _PrefillConceptLayout) -> Tensor:
        """Map packed token rows to the concept row visible after shift."""
        seq_concept_start = concept_layout.q_start_loc_long[token_layout.token_seq]
        seq_concept_count = concept_layout.q_seqlens_long[token_layout.token_seq]
        repeat_slots = self.repeat_slot_ids(token_layout.token_pos, self.chunk_size, self.shift_feature)
        valid_repeat = (repeat_slots >= 0) & (repeat_slots < seq_concept_count)
        return torch.where(
            valid_repeat,
            seq_concept_start + repeat_slots,
            torch.full_like(repeat_slots, -1),
        )

    def _build_prefill_merge_layout(self, token_layout: _PrefillTokenLayout,
                                    concept_layout: _PrefillConceptLayout) -> _PrefillMergeLayout:
        """Build token-to-concept merge metadata for mean/first/last modes."""
        seq_concept_start = concept_layout.q_start_loc_long[token_layout.token_seq]
        seq_concept_count = concept_layout.q_seqlens_long[token_layout.token_seq]
        merge_slots = torch.div(token_layout.token_pos, self.chunk_size, rounding_mode='floor')
        valid_merge = (merge_slots >= 0) & (merge_slots < seq_concept_count)
        merge_token_to_concept = torch.where(
            valid_merge,
            seq_concept_start + merge_slots,
            torch.full_like(merge_slots, -1),
        )

        concept_seq_len = token_layout.q_seqlens_long[concept_layout.seq]
        merge_start_pos = concept_layout.local_ids * self.chunk_size
        merge_counts_long = (concept_seq_len - merge_start_pos).clamp(min=0)
        merge_counts_long = torch.minimum(
            merge_counts_long,
            torch.full_like(merge_counts_long, self.chunk_size),
        )
        merge_token_counts = merge_counts_long.to(dtype=torch.int32)
        merge_token_start_ids = token_layout.q_start_loc_long[concept_layout.seq] + merge_start_pos
        merge_first_token_ids = merge_token_start_ids
        merge_last_token_ids = merge_token_start_ids + merge_counts_long.clamp(min=1) - 1
        merge_short_concept_mask = merge_counts_long < self.chunk_size

        return _PrefillMergeLayout(
            token_to_concept=merge_token_to_concept,
            token_start_ids=merge_token_start_ids,
            token_counts=merge_token_counts,
            first_token_ids=merge_first_token_ids,
            last_token_ids=merge_last_token_ids,
            short_concept_mask=merge_short_concept_mask,
        )

    def build_prefill_metadata(self, token_attn_metadata: Any, position_ids: Tensor) -> ConceptPrefillMetadata:
        """Build packed token-to-concept metadata for batched prefill."""
        token_layout = self._build_prefill_token_layout(token_attn_metadata, position_ids)
        concept_layout = self._build_prefill_concept_layout(token_attn_metadata, position_ids, token_layout)
        token_to_concept = self._build_prefill_repeat_ids(token_layout, concept_layout)
        merge_layout = self._build_prefill_merge_layout(token_layout, concept_layout)

        return ConceptPrefillMetadata(
            token_q_seqlens=token_layout.q_seqlens,
            token_q_start_loc=token_layout.q_start_loc,
            concept_q_seqlens=concept_layout.q_seqlens,
            concept_q_start_loc=concept_layout.q_start_loc,
            concept_position_ids=concept_layout.position_ids,
            merge_token_to_concept=merge_layout.token_to_concept,
            merge_token_start_ids=merge_layout.token_start_ids,
            merge_token_counts=merge_layout.token_counts,
            merge_first_token_ids=merge_layout.first_token_ids,
            merge_last_token_ids=merge_layout.last_token_ids,
            merge_short_concept_mask=merge_layout.short_concept_mask,
            token_to_concept=token_to_concept,
            num_tokens_total=token_layout.total_tokens,
            num_concepts_total=concept_layout.num_total,
            max_concepts_per_request=concept_layout.max_per_request,
        )

    @staticmethod
    def _merge_chunks_mean_packed(hidden_states: Tensor, prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Mean-pool packed token states by precomputed concept ids."""
        merge_token_to_concept = prefill_metadata.merge_token_to_concept.to(device=hidden_states.device)
        valid_merge = (merge_token_to_concept >= 0).to(dtype=hidden_states.dtype).unsqueeze(-1)
        safe_merge_ids = merge_token_to_concept.clamp(min=0)
        merged = hidden_states.new_zeros((prefill_metadata.num_concepts_total, hidden_states.size(-1)))
        merged.index_add_(0, safe_merge_ids, hidden_states * valid_merge)
        counts = prefill_metadata.merge_token_counts.clamp(min=1).to(device=hidden_states.device,
                                                                      dtype=hidden_states.dtype)
        return merged / counts.unsqueeze(-1)

    def merge_chunks_packed(self, hidden_states: Tensor, prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Merge packed token states into packed per-request concept states."""
        if self.merge_method == 'first':
            merged = hidden_states[prefill_metadata.merge_first_token_ids]
            short_mean = self._merge_chunks_mean_packed(hidden_states, prefill_metadata)
            return torch.where(prefill_metadata.merge_short_concept_mask[:, None], short_mean, merged)
        if self.merge_method == 'last':
            merged = hidden_states[prefill_metadata.merge_last_token_ids]
            short_mean = self._merge_chunks_mean_packed(hidden_states, prefill_metadata)
            return torch.where(prefill_metadata.merge_short_concept_mask[:, None], short_mean, merged)

        return self._merge_chunks_mean_packed(hidden_states, prefill_metadata)

    def prefill_chunk_state_update(self, source_states: Tensor, prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Merge prefill source states to the unified concept-source layout."""
        if source_states.dim() == 2:
            return self.merge_chunks_packed(source_states, prefill_metadata)
        if source_states.dim() != 3:
            raise ValueError(f'ConceptLM prefill source states must be 2-D or 3-D, got {tuple(source_states.shape)}.')
        chunks = [self.merge_chunks_packed(source_states[:, source_id], prefill_metadata)
                  for source_id in range(source_states.size(1))]
        return torch.stack(tuple(chunks), dim=1)

    @staticmethod
    def _gather_zero_prefixed_concepts(concept_states_with_zero: Tensor,
                                       prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Gather zero-prefixed concept rows to packed token rows."""
        token_to_concept = prefill_metadata.token_to_concept.to(device=concept_states_with_zero.device)
        gather_ids = torch.clamp(token_to_concept + 1, min=0)
        return concept_states_with_zero[gather_ids]

    def repeat_shift_packed(self, concept_states: Tensor, prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Gather packed concept states back to packed token states."""
        concept_states_with_zero = torch.cat((torch.zeros_like(concept_states[:1]), concept_states), dim=0)
        return self._gather_zero_prefixed_concepts(concept_states_with_zero, prefill_metadata)

    def repeat_shift_source_states_packed(self, concept_states_with_zero: Tensor,
                                          prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Gather zero-prefixed packed concept source states to token rows."""
        return self._gather_zero_prefixed_concepts(concept_states_with_zero, prefill_metadata)

    def begin_concept_forward(self, chunk_input: ConceptChunkInput,
                              runtime_caches: ConceptRuntimeCaches) -> ConceptForwardContext:
        """Prepare transient state before running the concept predictor."""
        if not chunk_input.is_decoding:
            return ConceptForwardContext()

        concept_past_key_values = runtime_caches.concept_past_key_values
        if concept_past_key_values is None:
            raise RuntimeError('ConceptLM decode requires concept KV caches.')
        decode_metadata = chunk_input.decode_metadata
        if decode_metadata is None:
            raise RuntimeError('ConceptLM decode input is missing decode metadata.')
        saved_kv = self.snapshot_decode_concept_kv(concept_past_key_values, chunk_input.attn_metadata)

        previous_final_state = None
        previous_raw_states = None
        if not self.shift_feature:
            previous_final_state, previous_raw_states = self.select_decode_last_state_rows(
                runtime_caches.last_state,
                runtime_caches.last_final_state,
                runtime_caches.last_raw_states,
                decode_metadata,
            )

        return ConceptForwardContext(
            saved_kv=saved_kv,
            previous_final_state=previous_final_state,
            previous_raw_states=previous_raw_states,
        )

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
        if not chunk_input.is_decoding:
            prefill_metadata = chunk_input.prefill_metadata
            if prefill_metadata is None:
                raise RuntimeError('ConceptLM prefill input is missing prefill metadata.')
            self.write_prefill_state_caches(
                runtime_caches.chunk_source_state,
                runtime_caches.last_raw_states,
                runtime_caches.last_final_state,
                chunk_input.state_ids,
                prefill_metadata,
                source_states,
                predicted_vectors,
                concept_raw_states,
            )
            return

        concept_past_key_values = runtime_caches.concept_past_key_values
        if concept_past_key_values is None:
            raise RuntimeError('ConceptLM decode requires concept KV caches.')
        if forward_context.saved_kv is None:
            raise RuntimeError('ConceptLM decode forward context is missing KV snapshot.')
        if chunk_input.decode_metadata is None or chunk_input.update_mask is None:
            raise RuntimeError('ConceptLM decode input is missing metadata or update mask.')
        if runtime_caches.last_raw_states is None or runtime_caches.last_final_state is None:
            raise RuntimeError('ConceptLM decode requires cached last concept states.')
        self.restore_decode_concept_kv(
            concept_past_key_values,
            chunk_input.attn_metadata,
            forward_context.saved_kv,
            ~chunk_input.update_mask,
        )
        self.write_decode_concept_states(
            runtime_caches.last_raw_states,
            runtime_caches.last_final_state,
            predicted_vectors,
            concept_raw_states,
            chunk_input.decode_metadata.state_ids,
            chunk_input.update_mask,
        )

    def build_decoder_concept_input(
        self,
        chunk_input: ConceptChunkInput,
        runtime_caches: ConceptRuntimeCaches,
        forward_context: ConceptForwardContext,
        predicted_vectors: Tensor,
        concept_raw_states: list[Tensor],
    ) -> ConceptDecoderInput:
        """Build token-decoder concept inputs from committed concept state."""
        if not chunk_input.is_decoding:
            prefill_metadata = chunk_input.prefill_metadata
            if prefill_metadata is None:
                raise RuntimeError('ConceptLM prefill input is missing prefill metadata.')
            final_state = self.repeat_shift_packed(predicted_vectors, prefill_metadata)
            raw_states = self.stack_concept_raw_states(concept_raw_states)
            zero_chunk = torch.zeros_like(raw_states[:1])
            route_states = self.repeat_shift_source_states_packed(
                torch.cat((zero_chunk, raw_states), dim=0),
                prefill_metadata,
            )
            return ConceptDecoderInput(final_state=final_state, route_states=route_states)

        decode_metadata = chunk_input.decode_metadata
        if decode_metadata is None:
            raise RuntimeError('ConceptLM decode input is missing decode metadata.')
        if self.shift_feature:
            final_state, route_states = self.select_decode_last_state_rows(
                runtime_caches.last_state,
                runtime_caches.last_final_state,
                runtime_caches.last_raw_states,
                decode_metadata,
            )
        else:
            final_state = forward_context.previous_final_state
            route_states = forward_context.previous_raw_states
            if final_state is None or route_states is None:
                raise RuntimeError('ConceptLM decode forward context is missing previous concept states.')

        concept_read_mask = self.decode_concept_read_mask(decode_metadata)
        final_state = torch.where(concept_read_mask.view(-1, 1), final_state, torch.zeros_like(final_state))
        route_states = torch.where(concept_read_mask.view(-1, 1, 1), route_states, torch.zeros_like(route_states))
        return ConceptDecoderInput(final_state=final_state, route_states=route_states)

    def build_concept_prefill_metadata(self, token_attn_metadata: Any, prefill_metadata: ConceptPrefillMetadata):
        """Build chunk-stream attention metadata for packed prefill."""
        concept_q_seqlens = prefill_metadata.concept_q_seqlens
        concept_q_start_loc = prefill_metadata.concept_q_start_loc
        concept_cu_seqlens = F.pad(
            torch.cumsum(concept_q_seqlens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        max_concept_seqlen = int(prefill_metadata.max_concepts_per_request)

        updates = dict(
            is_decoding=False,
            q_start_loc=concept_q_start_loc,
            q_seqlens=concept_q_seqlens,
            kv_seqlens=concept_q_seqlens,
            cu_seqlens_q=concept_cu_seqlens,
            cu_seqlens_k=concept_cu_seqlens,
        )
        if hasattr(token_attn_metadata, 'kv_start_loc'):
            updates['kv_start_loc'] = concept_q_start_loc
        if hasattr(token_attn_metadata, 'kv_flatten_size'):
            updates['kv_flatten_size'] = int(prefill_metadata.num_concepts_total)
        if hasattr(token_attn_metadata, 'max_q_seqlen'):
            updates['max_q_seqlen'] = max_concept_seqlen
        if hasattr(token_attn_metadata, 'max_kv_seqlen'):
            updates['max_kv_seqlen'] = max_concept_seqlen
        return replace(token_attn_metadata, **updates)

    def merge_prefill_tail_chunk_states(self, source_states: Tensor,
                                        prefill_metadata: ConceptPrefillMetadata) -> Tensor:
        """Build per-request partial chunk accumulator rows after prefill."""
        device = source_states.device
        q_seqlens = prefill_metadata.token_q_seqlens.to(device=device, dtype=torch.long)
        q_start_loc = prefill_metadata.token_q_start_loc.to(device=device, dtype=torch.long)
        batch_size = q_seqlens.size(0)
        tail_lens = torch.remainder(q_seqlens, self.chunk_size)
        tail_lens = torch.where(q_seqlens < self.chunk_size, q_seqlens, tail_lens)
        tail_lens = torch.where(q_seqlens > 0, tail_lens, torch.zeros_like(tail_lens))
        has_tail = tail_lens > 0

        tail_rows = source_states.new_zeros((batch_size, source_states.size(1), source_states.size(2)),
                                            dtype=torch.float32)
        if source_states.size(0) == 0:
            return tail_rows
        if self.merge_method == 'first':
            token_ids = q_start_loc + q_seqlens - tail_lens
            token_ids = token_ids.clamp(min=0, max=max(source_states.size(0) - 1, 0))
            rows = source_states[token_ids]
            return torch.where(has_tail.view(batch_size, 1, 1), rows, tail_rows)
        if self.merge_method == 'last':
            token_ids = q_start_loc + q_seqlens - 1
            token_ids = token_ids.clamp(min=0, max=max(source_states.size(0) - 1, 0))
            rows = source_states[token_ids]
            return torch.where(has_tail.view(batch_size, 1, 1), rows, tail_rows)

        token_ids = torch.arange(prefill_metadata.num_tokens_total, dtype=torch.long, device=device)
        cu_q_seqlens = F.pad(torch.cumsum(q_seqlens, dim=0), (1, 0))
        token_seq = torch.searchsorted(cu_q_seqlens[1:], token_ids, right=True)
        token_pos = token_ids - cu_q_seqlens[token_seq]
        token_tail_start = q_seqlens[token_seq] - tail_lens[token_seq]
        valid_tail = (tail_lens[token_seq] > 0) & (token_pos >= token_tail_start)
        weighted_source = source_states.float() * valid_tail.to(dtype=torch.float32).view(-1, 1, 1)
        tail_rows.index_add_(0, token_seq, weighted_source)
        return tail_rows

    @staticmethod
    def stack_concept_raw_states(concept_raw_states: list[Tensor]) -> Tensor:
        """Stack raw concept-layer states to ``[rows, concept_layers,
        hidden]``."""
        return torch.stack(tuple(concept_raw_states), dim=1)

    def write_prefill_state_caches(
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
        return self._write_prefill_state_caches_impl(
            chunk_source_state,
            last_raw_states,
            last_final_state,
            state_ids,
            prefill_metadata,
            source_states,
            predicted_vectors,
            concept_raw_states,
        )

    def _write_prefill_state_caches_impl(
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
        if state_ids is None:
            return
        if chunk_source_state is None or last_raw_states is None or last_final_state is None:
            return

        state_ids = state_ids.to(device=source_states.device, dtype=torch.long).reshape(-1)
        valid_indices = torch.nonzero(state_ids >= 0, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return

        valid_state_ids = state_ids.index_select(0, valid_indices)
        tail_rows = self.merge_prefill_tail_chunk_states(source_states, prefill_metadata)
        chunk_source_state.index_copy_(0, valid_state_ids,
                                       tail_rows.index_select(0, valid_indices).to(dtype=chunk_source_state.dtype))

        concept_counts = prefill_metadata.concept_q_seqlens.to(device=source_states.device, dtype=torch.long)
        concept_start = prefill_metadata.concept_q_start_loc.to(device=source_states.device, dtype=torch.long)
        concept_valid_mask = (state_ids >= 0) & (concept_counts > 0)
        concept_indices = torch.nonzero(concept_valid_mask, as_tuple=False).flatten()
        if concept_indices.numel() == 0:
            return

        concept_state_ids = state_ids.index_select(0, concept_indices)
        last_concept_ids = concept_start.index_select(0, concept_indices) + concept_counts.index_select(
            0, concept_indices) - 1
        last_final_rows = predicted_vectors.index_select(0, last_concept_ids)
        last_final_state.index_copy_(0, concept_state_ids, last_final_rows.to(dtype=last_final_state.dtype))
        raw_rows = self.stack_concept_raw_states(concept_raw_states).index_select(0, last_concept_ids)
        last_raw_states.index_copy_(0, concept_state_ids, raw_rows.to(dtype=last_raw_states.dtype))

    def snapshot_decode_concept_kv(self, concept_past_key_values: list[list[Tensor]],
                                   concept_attn_metadata: Any) -> list[tuple[Tensor, Tensor]]:
        """Snapshot concept KV slots that dummy non-boundary rows may
        overwrite."""
        return [
            self.decode_kv_cache_snapshot(
                k_cache,
                v_cache,
                concept_attn_metadata.block_offsets,
                concept_attn_metadata.kv_seqlens,
            )
            for k_cache, v_cache in concept_past_key_values
        ]

    def restore_decode_concept_kv(self, concept_past_key_values: list[list[Tensor]], concept_attn_metadata: Any,
                                  saved_kv: list[tuple[Tensor, Tensor]], restore_mask: Tensor) -> None:
        """Restore concept KV slots for non-boundary and padded rows."""
        for (k_cache, v_cache), (saved_k, saved_v) in zip(concept_past_key_values, saved_kv):
            self.decode_kv_cache_restore(
                k_cache,
                v_cache,
                saved_k,
                saved_v,
                concept_attn_metadata.block_offsets,
                concept_attn_metadata.kv_seqlens,
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
        """Write newly emitted concept states to persistent decode caches."""
        raw_rows = self.stack_concept_raw_states(raw_states)
        self.decode_concept_state_update(
            last_raw_state_cache,
            last_final_state_cache,
            predicted_vectors,
            raw_rows,
            state_ids,
            update_mask,
        )

    @staticmethod
    def _decode_kv_cache_rows(k_cache: Tensor,
                              block_offsets: Tensor,
                              kv_seqlens: Tensor) -> tuple[Tensor, Tensor]:
        """Return cache block ids and page offsets for one slot per row."""
        block_size = k_cache.size(1)
        kv_seqlens = kv_seqlens.to(device=block_offsets.device, dtype=torch.long).clamp(min=1)
        slot_ids = kv_seqlens - 1
        block_idx = torch.div(slot_ids, block_size, rounding_mode='floor')
        page_offsets = torch.remainder(slot_ids, block_size)
        block_ids = block_offsets.to(dtype=torch.long).gather(1, block_idx.view(-1, 1)).view(-1)
        return block_ids, page_offsets

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
        position_ids = self.flatten_decode_position_ids(position_ids, batch_size, current_source_states.device)
        assert position_ids.numel() == batch_size, (
            f'Expected {batch_size} decode position ids, got {position_ids.numel()}.')

        valid_state_mask = state_ids >= 0
        safe_state_ids = state_ids.clamp(min=0)
        accumulator_dtype = chunk_source_state_cache.dtype
        previous_rows = chunk_source_state_cache.index_select(0, safe_state_ids)
        current_rows = current_source_states.to(dtype=accumulator_dtype)

        chunk_size = int(chunk_size)
        chunk_pos = torch.remainder(position_ids, chunk_size)
        update_mask = valid_state_mask & (torch.remainder(position_ids + 1, chunk_size) == 0)
        first_token_mask = valid_state_mask & (chunk_pos == 0)
        merge_method = str(merge_method)

        if merge_method == 'first':
            update_rows = torch.where(first_token_mask.view(batch_size, 1, 1), current_rows, previous_rows)
            concept_input_states = update_rows
        elif merge_method == 'last':
            update_rows = current_rows
            concept_input_states = current_rows
        else:
            update_rows = previous_rows + current_rows
            concept_input_states = update_rows / chunk_size

        zero_rows = torch.zeros_like(update_rows)
        next_rows = torch.where(update_mask.view(batch_size, 1, 1), zero_rows, update_rows)
        next_rows = torch.where(valid_state_mask.view(batch_size, 1, 1), next_rows, previous_rows)
        concept_zero_rows = torch.zeros_like(current_source_states)
        concept_input_states = torch.where(
            update_mask.view(batch_size, 1, 1),
            concept_input_states.to(dtype=current_source_states.dtype),
            concept_zero_rows,
        )

        for batch_idx in range(batch_size):
            state_id = int(state_ids[batch_idx])
            if state_id >= 0:
                chunk_source_state_cache[state_id].copy_(next_rows[batch_idx])
        return concept_input_states, update_mask

    def decode_kv_cache_snapshot(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        block_offsets: Tensor,
        kv_seqlens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Snapshot one decode KV slot per batch row."""
        block_ids, page_offsets = self._decode_kv_cache_rows(k_cache, block_offsets, kv_seqlens)
        return k_cache[block_ids, page_offsets].clone(), v_cache[block_ids, page_offsets].clone()

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
        block_ids, page_offsets = self._decode_kv_cache_rows(k_cache, block_offsets, kv_seqlens)
        restore_mask = restore_mask.to(device=k_cache.device, dtype=torch.bool).view(-1, 1, 1)
        current_k = k_cache[block_ids, page_offsets]
        current_v = v_cache[block_ids, page_offsets]
        k_cache[block_ids, page_offsets] = torch.where(restore_mask, saved_k, current_k)
        v_cache[block_ids, page_offsets] = torch.where(restore_mask, saved_v, current_v)

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
        state_ids = state_ids.to(device=predicted_vectors.device, dtype=torch.long).reshape(-1)
        update_mask = update_mask.to(device=predicted_vectors.device, dtype=torch.bool).reshape(-1)
        for batch_idx in range(state_ids.numel()):
            state_id = int(state_ids[batch_idx])
            if state_id < 0 or not bool(update_mask[batch_idx]):
                continue
            last_final_state_cache[state_id].copy_(predicted_vectors[batch_idx].to(last_final_state_cache.dtype))
            last_raw_state_cache[state_id].copy_(raw_states[batch_idx].to(last_raw_state_cache.dtype))


class DefaultConceptLMRuntimeOpsBuilder(ConceptLMRuntimeOpsBuilder):
    """Torch fallback ConceptLM runtime operation builder."""

    @staticmethod
    def build(config) -> ConceptLMRuntimeOpsImpl:
        """Build layer implementation."""
        return DefaultConceptLMRuntimeOpsImpl(config)
