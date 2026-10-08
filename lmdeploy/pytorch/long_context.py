# Copyright (c) OpenMMLab. All rights reserved.
"""Shared planning helpers for long-context prefill chunks."""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.multimodal.data_type import MultiModalInputs


@dataclass
class LongContextChunkPlan:
    """Boundary and payload decision for one long-context chunk.

    The planner is shared by the scheduler and input maker so both sides make
    the same chunk-boundary decision.  Scheduler code mostly consumes the
    absolute ``chunk_end`` as the temporary KV limit for a non-final chunk,
    while input construction consumes ``chunk_size`` and ``multimodals`` to
    build the next eager chunk forward.

    Args:
        chunk_limit: Effective per-chunk token budget.  This starts from
            ``max_prefill_token_num`` and may be raised to fit an indivisible
            multimodal span.
        chunk_start: Absolute prompt position where the next suffix chunk
            starts.  This is normally ``seq.num_history_ids`` after any
            accepted prefix-cache hit has advanced the sequence.
        chunk_end: Exclusive absolute prompt position where the planned chunk
            ends.  If a multimodal span would cross the budget boundary, this
            is clamped back to the span start, including the preceding token
            when the draft consumes shifted inputs.
        chunk_size: Number of tokens to send in the next forward,
            ``chunk_end - chunk_start``.
        is_last_chunk: Whether this chunk reaches the end of the prefill.
            Last chunks are handled as normal prefill so they can merge into
            persistent decode state.
        multimodals: Remaining multimodal payloads wholly contained in this
            chunk, or ``None`` when the caller does not need payloads or no
            multimodal data is emitted for this chunk.
    """

    chunk_limit: int
    chunk_start: int
    chunk_end: int
    chunk_size: int
    is_last_chunk: bool
    multimodals: 'MultiModalInputs|None'


def sort_long_context_multimodals(multimodals: 'MultiModalInputs') -> 'MultiModalInputs':
    """Return multimodals sorted by prompt start within each modality."""
    output = defaultdict(list)
    for modal_type, modal_datas in multimodals.items():
        output[modal_type] = sorted(modal_datas, key=lambda data: data.start)
    return output


def has_long_context_multimodal(multimodals: 'MultiModalInputs') -> bool:
    """Return whether at least one multimodal span remains."""
    return any(len(modal_datas) > 0 for modal_datas in multimodals.values())


def _iter_sorted_multimodals(multimodals: 'MultiModalInputs'):
    multimodal_data = []
    for modal_type, modal_datas in multimodals.items():
        if len(modal_datas) == 0:
            continue
        multimodal_data += [(modal_type, data) for data in modal_datas]

    yield from sorted(multimodal_data, key=lambda item: item[1].start)


def _iter_multimodal_chunk_spans(sorted_multimodals, input_shift: int):
    """Merge spans that share a shifted draft-input dependency."""
    group_start = group_end = 0
    for _, data in sorted_multimodals:
        span_start = max(0, data.start - input_shift)
        if span_start >= group_end:
            if group_end > group_start:
                yield group_start, group_end
            group_start = span_start
        group_end = max(group_end, data.end)
    if group_end > group_start:
        yield group_start, group_end


def get_long_context_chunk_limit(seq: 'SchedulerSequence', max_prefill_token_num: int) -> int:
    """Return the token budget for one long-context chunk."""
    mm_for_chunk_limit = seq.get_chunk_limit_multimodals()
    spans = _iter_multimodal_chunk_spans(_iter_sorted_multimodals(mm_for_chunk_limit), seq.prefill_input_shift)
    max_span = max((end - start for start, end in spans), default=0)
    return max(max_prefill_token_num, max_span)


def plan_long_context_chunk(seq: 'SchedulerSequence',
                            chunk_limit: int,
                            multimodals: 'MultiModalInputs|None' = None,
                            include_multimodals: bool = True,
                            save_alignment: int = 0) -> LongContextChunkPlan:
    """Plan a model-safe chunk, optionally stopping at its last save boundary.

    Alignment is a constraint, not a sampling interval: earlier aligned positions in this chunk are skipped. If no safe
    aligned position advances the request, use the ordinary chunk and keep only its running state.
    """
    chunk_size = min(seq.num_token_ids, chunk_limit)
    start = seq.num_history_ids
    end = start + chunk_size

    if multimodals is None:
        multimodals = seq.get_input_multimodals()
    spans = list(_iter_sorted_multimodals(multimodals))
    chunk_spans = []
    # The preceding MTP row needs the first multimodal embedding. Compute
    # them in the same chunk, after the target has encoded the whole span.
    for span_start, span_end in _iter_multimodal_chunk_spans(spans, seq.prefill_input_shift):
        if span_start >= end:
            break
        chunk_spans.append((span_start, span_end))
        if span_end > end:
            end = span_start
            break

    # Embedding-only inputs do not yet have a Store content identity.
    if save_alignment and len(seq.history_embeddings) == 0:
        save_end = end // save_alignment * save_alignment
        for span_start, span_end in reversed(chunk_spans):
            if span_start < save_end < span_end:
                save_end = span_start // save_alignment * save_alignment
        if save_end > start:
            end = save_end

    chunk_size = end - start
    out_multimodals = None
    if include_multimodals and spans:
        out_multimodals = defaultdict(list)
        for modal_type, data in spans:
            if data.end <= end:
                out_multimodals[modal_type].append(data)
    return LongContextChunkPlan(chunk_limit=chunk_limit,
                                chunk_start=start,
                                chunk_end=end,
                                chunk_size=chunk_size,
                                is_last_chunk=chunk_size == seq.num_token_ids,
                                multimodals=out_multimodals)
