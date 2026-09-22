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
            is clamped back to the span start.
        chunk_size: Number of tokens to send in the next forward,
            ``chunk_end - chunk_start``.
        is_last_chunk: Whether the remaining suffix fits in ``chunk_limit``.
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


def _max_multimodal_span(multimodals: 'MultiModalInputs') -> int:
    return max([data.end - data.start for modal_datas in multimodals.values() for data in modal_datas], default=0)


def get_long_context_chunk_limit(seq: 'SchedulerSequence', max_prefill_token_num: int) -> int:
    """Return the token budget for one long-context chunk."""
    mm_for_chunk_limit = seq.get_chunk_limit_multimodals()
    return max(max_prefill_token_num, _max_multimodal_span(mm_for_chunk_limit))


def plan_long_context_chunk(seq: 'SchedulerSequence',
                            chunk_limit: int,
                            multimodals: 'MultiModalInputs|None' = None,
                            include_multimodals: bool = True) -> LongContextChunkPlan:
    """Plan the next chunk without splitting multimodal spans."""
    chunk_size = min(seq.num_token_ids, chunk_limit)
    start = seq.num_history_ids
    end = start + chunk_size

    if multimodals is None:
        multimodals = seq.get_input_multimodals()
    if len(multimodals) == 0:
        return LongContextChunkPlan(chunk_limit=chunk_limit,
                                    chunk_start=start,
                                    chunk_end=end,
                                    chunk_size=chunk_size,
                                    is_last_chunk=seq.num_token_ids <= chunk_limit,
                                    multimodals=None)

    out_multimodals = defaultdict(list)
    for modal_type, data in _iter_sorted_multimodals(multimodals):
        assert data.start >= start, 'multimodal data should be sorted by start'
        if data.start >= end:
            break
        if data.end > end:
            # Do not split a multimodal span; recompute from its start in the
            # next chunk instead.
            end = data.start
            break
        if include_multimodals:
            out_multimodals[modal_type].append(data)

    chunk_size = end - start
    if not include_multimodals:
        out_multimodals = None
    return LongContextChunkPlan(chunk_limit=chunk_limit,
                                chunk_start=start,
                                chunk_end=end,
                                chunk_size=chunk_size,
                                is_last_chunk=seq.num_token_ids <= chunk_limit,
                                multimodals=out_multimodals)
