# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from lmdeploy.utils import get_logger

from .loggers import StatLoggerBase
from .stats import IterationStats, RequestState, SchedulerStats

if TYPE_CHECKING:
    from lmdeploy.messages import EngineCoreEvent
    from lmdeploy.pytorch.paging.scheduler import Scheduler

logger = get_logger('lmdeploy')


@dataclass
class MetricsContext:
    enable_metrics: bool = False
    req_state: RequestState = RequestState()
    scheduler_stats: SchedulerStats = SchedulerStats()
    engine_core_timestamp: float = 0.0
    engine_core_events: List['EngineCoreEvent'] = field(default_factory=list)


class MetricsManager:

    def __init__(self):
        """Initialize metrics manager."""
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


_METRICS_MANAGER = None


def get_metrics_manager():
    global _METRICS_MANAGER
    if _METRICS_MANAGER is None:
        _METRICS_MANAGER = MetricsManager()

    return _METRICS_MANAGER


# Metrics getters
def is_metrics_enabled():
    return get_metrics_manager().get_context().enable_metrics


def get_current_metrics_context():
    return get_metrics_manager().get_context()


def get_current_request_state():
    return get_metrics_manager().get_context().req_state


def get_current_scheduler_stats():
    return get_metrics_manager().get_context().scheduler_stats


def get_current_engine_core_timestamp():
    return get_metrics_manager().get_context().engine_core_timestamp


def get_current_engine_core_events():
    return get_metrics_manager().get_context().engine_core_events


# Metrics setters
def set_metrics_enabled_flag(enable_metrics: bool):
    """Set metrics enabled flag."""
    ctx = get_current_metrics_context()
    ctx.enable_metrics = enable_metrics

    if enable_metrics:
        logger.info('Metrics are enabled.')


def init_async_engine_request_state(prompt_len: int):
    """Initialize request state in async engine."""
    from .stats import RequestStateStats
    req_state = get_current_request_state()
    req_state.arrival_time = time.perf_counter()
    req_state.prompt_len = prompt_len
    req_state.is_prefilling = True  # new request starts as prefill
    req_state.stats = RequestStateStats(arrival_time=req_state.arrival_time)


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
    # actually running requests
    scheduler_stats.num_running_reqs = scheduler.num_locked()
    # waiting to be scheduled requests + scheduled but not yet started requests
    scheduler_stats.num_waiting_reqs = scheduler.num_waiting() + scheduler.num_running()
    scheduler_stats.gpu_cache_usage = scheduler.usage


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


# Metrics processor
class MetricsProcessor:
    """Metrics processor."""

    def __init__(self):
        self.metrics_queue: asyncio.Queue = None
        self.consumer_task: asyncio.Task = None

    def start_metrics_handler(self, enable_metrics: bool):
        set_metrics_enabled_flag(enable_metrics)

        if self.consumer_task is None:
            self.metrics_queue = asyncio.Queue()
            self.consumer_task = asyncio.create_task(self._run_metrics_handler())
            logger.info('Metrics handler task started.')

    async def stop_metrics_handler(self):
        if self.consumer_task is not None:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass  # Expected cancellation
            finally:
                self.consumer_task = None
                logger.info('Metrics handler task stopped.')

    async def _run_metrics_handler(self):
        """A background task that consumes and processes metrics data."""
        while True:
            try:
                task_type, data = await self.metrics_queue.get()

                if task_type == 'update':
                    prev_len, input_len, output_len, iteration_stats = data
                    self._update_stats(prev_len, input_len, output_len, iteration_stats)
                elif task_type == 'record':
                    stat_loggers, iteration_stats = data
                    self._record_stats(stat_loggers, iteration_stats)

                self.metrics_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.info(f'Error in metrics handler: {e}')

    def queue_update(self, prev_len: int, input_len: int, output_len: int, iteration_stats: IterationStats):
        if self.metrics_queue is not None:
            update_data = (prev_len, input_len, output_len, iteration_stats)
            self.metrics_queue.put_nowait(('update', update_data))

    def queue_record(self, stat_loggers: List[StatLoggerBase], iteration_stats: IterationStats):
        if self.metrics_queue is not None:
            record_data = (stat_loggers, iteration_stats)
            self.metrics_queue.put_nowait(('record', record_data))

    def increment_total_requests(self):
        increment_async_engine_scheduler_stats_total_req()

    def increment_finished_requests(self):
        increment_async_engine_scheduler_stats_finished_req()

    def init_stats(self, prompt_len: int):
        init_async_engine_request_state(prompt_len)

    def _update_stats(self, prev_len: int, input_len: int, output_len: int, iteration_stats: IterationStats):
        is_prefilling = (prev_len == 0)
        num_prompt_tokens = input_len
        num_new_generation_tokens = output_len - prev_len

        set_async_engine_request_state(is_prefilling)
        iteration_stats.update_from_output(engine_core_timestamp=get_current_engine_core_timestamp(),
                                           engine_core_events=get_current_engine_core_events(),
                                           num_prompt_tokens=num_prompt_tokens,
                                           num_new_generation_tokens=num_new_generation_tokens,
                                           is_prefilling=is_prefilling,
                                           req_stats=get_current_request_state().stats)

    def _record_stats(self, stat_loggers: List[StatLoggerBase], iteration_stats: IterationStats):
        for stat_logger in stat_loggers:
            stat_logger.record(scheduler_stats=get_current_scheduler_stats(), iteration_stats=iteration_stats)


metrics_processor = MetricsProcessor()
