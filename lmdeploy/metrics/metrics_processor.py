# Copyright (c) OpenMMLab. All rights reserved.
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from .stats import IterationStats, RequestState, SchedulerStats

if TYPE_CHECKING:
    from lmdeploy.messages import EngineCoreEvent, ResponseType
    from lmdeploy.pytorch.paging.scheduler import Scheduler


@dataclass
class MetricsContext:
    req_state: RequestState = RequestState()
    scheduler_stats: SchedulerStats = SchedulerStats()
    iteration_stats: IterationStats = IterationStats()
    engine_core_timestamp: float = 0.0
    engine_core_events: List['EngineCoreEvent'] = field(default_factory=list)


class MetricsProcessor:

    def __init__(self):
        """Initialize metrics processor."""
        self._current_ctx = MetricsContext()

    def set_context(self, ctx: MetricsContext):
        """Set metrics context."""
        self._current_ctx = ctx

    def get_context(self):
        """Get current context."""
        return self._current_ctx

    @contextmanager
    def context(self, ctx: MetricsContext):
        """Context manager."""
        old_ctx = self.get_context()
        self.set_context(ctx)
        try:
            yield ctx
        finally:
            self.set_context(old_ctx)


_METRICS_PROCESSOR = None


def get_metrics_processor():
    global _METRICS_PROCESSOR
    if _METRICS_PROCESSOR is None:
        _METRICS_PROCESSOR = MetricsProcessor()

    return _METRICS_PROCESSOR


# Metrics getters
def get_current_metrics_context():
    return get_metrics_processor().get_context()


def get_current_request_state():
    return get_metrics_processor().get_context().req_state


def get_current_scheduler_stats():
    return get_metrics_processor().get_context().scheduler_stats


def get_current_iteration_stats():
    return get_metrics_processor().get_context().iteration_stats


def get_current_engine_core_timestamp():
    return get_metrics_processor().get_context().engine_core_timestamp


def get_current_engine_core_events():
    return get_metrics_processor().get_context().engine_core_events


# Metrics setters
def init_async_engine_request_state(prompt_len: int):
    """Initialize request state in async engine."""
    from .stats import RequestStateStats
    req_state = get_current_request_state()
    req_state.arrival_time = time.perf_counter()
    req_state.prompt_len = prompt_len
    req_state.is_prefilling = True  # new request starts as prefill
    req_state.stats = RequestStateStats(arrival_time=req_state.arrival_time)


def init_async_engine_iteration_stats():
    """Initialize iteration stats in async engine."""
    ctx = get_current_metrics_context()
    ctx.iteration_stats = IterationStats()


def set_async_engine_request_state(is_prefilling: bool = True):
    """Set request state in async engine."""
    get_current_request_state().is_prefilling = is_prefilling


def increment_async_engine_scheduler_stats_total_req():
    """Set scheduler stats in async engine."""
    get_current_scheduler_stats().num_total_reqs += 1


def increment_async_engine_scheduler_stats_finished_req():
    """Set scheduler stats in async engine."""
    get_current_scheduler_stats().num_finished_reqs += 1


def set_pt_engine_scheduler_stats(scheduler: 'Scheduler'):
    """Set scheduler stats in PyTorch engine."""
    scheduler_stats = get_current_scheduler_stats()
    scheduler_stats.num_running_reqs = scheduler.num_running()
    scheduler_stats.num_waiting_reqs = scheduler.num_waiting()
    scheduler_stats.gpu_cache_usage = scheduler.usage


def set_async_engine_iteration_stats(num_prompt_tokens: int = 0, num_generation_tokens: int = 0):
    """Set iteration stats in async engine."""
    iteration_stats = get_current_iteration_stats()
    iteration_stats.num_prompt_tokens = num_prompt_tokens
    iteration_stats.num_generation_tokens = num_generation_tokens


def set_pt_engine_core_newtoken_timestamp():
    """Set engine core new token generation timestamp in PyTorch engine."""
    ctx = get_current_metrics_context()
    ctx.engine_core_timestamp = time.perf_counter()


def set_pt_engine_core_event_queued():
    """Set engine core event in PyTorch engine."""
    from lmdeploy.messages import EngineCoreEvent, EngineCoreEventType

    engine_core_events = get_current_engine_core_events()
    engine_core_events.append(EngineCoreEvent.new_event(EngineCoreEventType.QUEUED))


def set_pt_engine_core_event_scheduled():
    """Set engine core event in PyTorch engine."""
    from lmdeploy.messages import EngineCoreEvent, EngineCoreEventType

    engine_core_events = get_current_engine_core_events()
    engine_core_events.append(EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED))


# Metrics updaters
def update_iteration_stats(reps_status: 'ResponseType', ctx: MetricsContext):
    """Update iteration stats."""
    iteration_stats = get_current_iteration_stats()

    # perform computations with metrics ctx
    iteration_stats.update_from_ctx(reps_status, ctx)


# Async Engine interface for metrics processor
class AsyncEngineMetricsProcessorInterface:
    """Provides simple interface to metrics functions used by async engine."""

    def get_context(self) -> MetricsContext:
        return get_current_metrics_context()

    def get_scheduler_stats(self) -> SchedulerStats:
        return get_current_scheduler_stats()

    def init_stats(self, prompt_len: int):
        """Initialize metrics for a new request."""
        init_async_engine_request_state(prompt_len)
        init_async_engine_iteration_stats()
        increment_async_engine_scheduler_stats_total_req()

    def set_stats(self, prev_len: int, input_len: int, output_len: int):
        """Set metrics values for a request."""
        is_prefilling = (prev_len == 0)
        num_prompt_tokens = input_len if is_prefilling else 0
        num_new_generation_tokens = output_len - prev_len

        set_async_engine_request_state(is_prefilling)
        set_async_engine_iteration_stats(num_prompt_tokens=num_prompt_tokens,
                                         num_generation_tokens=num_new_generation_tokens)

    def increment_finished_requests(self):
        increment_async_engine_scheduler_stats_finished_req()

    def update_global_iteration_stats(self, reps_status: 'ResponseType', ctx: MetricsContext):
        """Update global iteration stats."""
        update_iteration_stats(reps_status, ctx)


async_engine_metrics_api = AsyncEngineMetricsProcessorInterface()
