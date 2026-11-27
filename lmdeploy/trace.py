# Copyright (c) OpenMMLab. All rights reserved.
"""Process-start trace flags shared by engine and service code; no service
imports."""
import os
from dataclasses import dataclass

ENABLED = bool(os.getenv('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT')
               or os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'))

@dataclass(frozen=True, slots=True)
class PrefillMetrics:
    """Completed host-side prefill summary; no per-chunk history or payloads.

    Token counters describe executed target-model prefill positions, including replay. Cached counters cover only the
    current input at initial dispatch. Waits cover completion observed by the host through the next dispatch, excluding
    time already queued in the executor. They are not GPU timings.
    """

    chunk_count: int
    computed_tokens: int
    computed_input_tokens: int
    recomputed_tokens: int
    min_chunk_tokens: int
    max_chunk_tokens: int
    local_cached_tokens: int
    external_cached_tokens: int
    inter_chunk_wait_seconds: float
    max_inter_chunk_wait_seconds: float
    preemption_count: int
    reschedule_count: int
