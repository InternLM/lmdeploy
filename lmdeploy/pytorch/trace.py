# Copyright (c) OpenMMLab. All rights reserved.
"""Bounded, host-only TTFT accounting.

No tensor allocation, SDK calls or GPU syncs.
"""

import logging
import time
from dataclasses import dataclass

from lmdeploy.trace import PrefillMetrics

logger = logging.getLogger('lmdeploy')
_observation_errors = 0


def observation_error_count():
    """Process-local count of input summaries omitted after observation
    errors."""
    return _observation_errors


@dataclass(slots=True)
class PrefillTrace:
    """One accumulator per input turn, frozen after accepted prefill output."""

    summary: PrefillMetrics | None = None
    invalid: bool = False
    chunks: int = 0
    tokens: int = 0
    input_tokens: int = 0
    recomputed: int = 0
    min_tokens: int = 0
    max_tokens: int = 0
    local_cached: int = 0
    external_cached: int = 0
    external_start: int = 0
    external_end: int = 0
    wait: float = 0.0
    max_wait: float = 0.0
    preemptions: int = 0
    reschedules: int = 0
    pending: int = 0
    dispatched: bool = False
    preempted: bool = False
    high_water: int = 0
    completed_at: float | None = None

    def invalidate(self, operation, error_type):
        """Omit this input's incomplete summary without disabling other
        inputs."""
        if self.invalid or self.summary is not None:
            return
        self.invalid = True
        self.pending = 0
        global _observation_errors
        _observation_errors += 1
        # Bound repeated diagnostics; never log exception text or input data.
        if _observation_errors & (_observation_errors - 1) == 0:
            try:
                logger.warning('Prefill trace summary omitted: operation=%s error_type=%s count=%d',
                               operation, error_type, _observation_errors)
            except Exception:
                pass  # Diagnostic handlers must not fail model execution.

    def preempt(self):
        if not self.invalid and self.dispatched and self.summary is None and not self.preempted:
            self.preemptions += 1
            self.preempted = True

    def dispatch(self, seq, start, count, now):
        if self.invalid or self.summary is not None:
            return
        if not self.dispatched:
            # Observe committed ownership only. Tentative matches, lookup hits,
            # failed loads and retained history outside this input are excluded.
            self.external_cached = max(0, min(start, self.external_end, seq.input_end_pos)
                                       - max(self.external_start, seq.input_start_pos))
            self.external_cached = min(self.external_cached, seq.cached_tokens)
            self.local_cached = max(0, seq.cached_tokens - self.external_cached)
        if self.pending == 0 and self.completed_at is not None:
            gap = max(0.0, now - self.completed_at)
            self.wait += gap
            self.max_wait = max(self.max_wait, gap)
        if self.preempted:
            self.reschedules += 1
            self.preempted = False
        self.dispatched = True
        self.pending += 1
        return (self, start, start + count, seq.input_start_pos, seq.input_end_pos)

    def complete(self, start, end, input_start, input_end, now):
        if self.invalid:
            return
        self.pending -= 1
        # An advancing start can reflect accepted KV reuse on re-admission.
        # Recompute counts execution of positions already crossed, whether
        # previously computed here or supplied by cache.
        self.recomputed += max(0, min(end, self.high_water) - start)
        self.high_water = max(self.high_water, end)
        count = end - start
        self.chunks += 1
        self.tokens += count
        self.input_tokens += max(0, min(end, input_end) - max(start, input_start))
        self.min_tokens = min(self.min_tokens, count) if self.chunks > 1 else count
        self.max_tokens = max(self.max_tokens, count)
        self.completed_at = now

    def accept(self):
        """Freeze only when the engine accepts the final prefill output."""
        if self.invalid or self.summary is not None or self.pending or not self.chunks:
            return
        try:
            self.summary = PrefillMetrics(
                chunk_count=self.chunks, computed_tokens=self.tokens,
                computed_input_tokens=self.input_tokens, recomputed_tokens=self.recomputed,
                min_chunk_tokens=self.min_tokens, max_chunk_tokens=self.max_tokens,
                local_cached_tokens=self.local_cached, external_cached_tokens=self.external_cached,
                inter_chunk_wait_seconds=self.wait, max_inter_chunk_wait_seconds=self.max_wait,
                preemption_count=self.preemptions, reschedule_count=self.reschedules)
        except Exception as exc:
            self.invalidate('accept', type(exc).__name__)


def begin_prefill(running, inputs):
    """Capture immutable token ranges before async prefetch mutates sequences.

    Inputs must still be on the host. Never transfer a tensor just for metrics. The returned batch-sized sidecar stays
    in the engine loop, not the workers.
    """
    if inputs is None or inputs.is_decoding:
        return None
    try:
        if inputs.seq_length.device.type != 'cpu' or inputs.history_lengths.device.type != 'cpu':
            raise ValueError('prefill trace inputs must be on the host')
        lengths = inputs.seq_length.tolist()
        starts = inputs.history_lengths.tolist()
        if len(running) != len(lengths) or len(running) != len(starts):
            raise ValueError('prefill trace batch mismatch')
        now = time.monotonic()
    except Exception as exc:
        for seq in running:
            if seq.prefill_trace is not None:
                seq.prefill_trace.invalidate('begin', type(exc).__name__)
        return None
    records = []
    for seq, start, count in zip(running, starts, lengths):
        if seq.prefill_trace is not None:
            try:
                if count <= 0:
                    continue
                record = seq.prefill_trace.dispatch(seq, start, count, now)
                if record is not None:
                    records.append(record)
            except Exception as exc:
                seq.prefill_trace.invalidate('dispatch', type(exc).__name__)
    return records or None


def complete_prefill(batch):
    """Called only after the existing executor output wait has completed."""
    if batch is None:
        return
    now = time.monotonic()
    for state, start, end, input_start, input_end in batch:
        try:
            state.complete(start, end, input_start, input_end, now)
        except Exception as exc:
            state.invalidate('complete', type(exc).__name__)


def discard_prefill(batch, operation, error_type):
    """Release accounting for forwards whose execution cannot be
    established."""
    if batch is not None:
        for state, *_ in batch:
            state.invalidate(operation, error_type)
