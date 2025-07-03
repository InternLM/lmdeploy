# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lmdeploy.utils import get_logger

from .stats import IterationStats, RequestState, SchedulerStats

if TYPE_CHECKING:
    from lmdeploy.messages import EngineOutput

logger = get_logger('lmdeploy')


@dataclass
class MetricsContext:
    enable_metrics: bool = False
    scheduler_stats: SchedulerStats = SchedulerStats()


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


def get_current_scheduler_stats():
    return get_metrics_manager().get_context().scheduler_stats


# Metrics setters
def set_metrics_enabled_flag(enable_metrics: bool):
    """Set metrics enabled flag."""
    ctx = get_current_metrics_context()
    ctx.enable_metrics = enable_metrics

    if enable_metrics:
        logger.info('Metrics are enabled.')


def increment_async_engine_scheduler_stats_total_req():
    """Set scheduler stats in async engine."""
    get_current_scheduler_stats().num_total_reqs += 1


def increment_async_engine_scheduler_stats_finished_req():
    """Set scheduler stats in async engine."""
    get_current_scheduler_stats().num_finished_reqs += 1


# Metrics processor
class MetricsProcessor():
    """Metrics processor."""

    def __init__(self):
        self.metrics_queue: asyncio.Queue = None
        self.metrics_handler: asyncio.Task = None

    def start_metrics_handler(self, enable_metrics: bool):
        set_metrics_enabled_flag(enable_metrics)

        if enable_metrics and self.metrics_handler is None:
            self.metrics_queue = asyncio.Queue()
            self.metrics_handler = asyncio.create_task(self._run_metrics_handler())
            logger.info('Metrics handler task started.')

    async def stop_metrics_handler(self):
        if self.metrics_handler is not None:
            self.metrics_handler.cancel()
            try:
                await self.metrics_handler
            except asyncio.CancelledError:
                pass  # Expected cancellation
            finally:
                self.metrics_handler = None
                logger.info('Metrics handler task stopped.')

    async def _run_metrics_handler(self):
        """A background task that consumes and processes metrics data."""
        while True:
            try:
                # fetch
                update_data = await self.metrics_queue.get()
                input_len, prev_len, outputs, req_state, iteration_stats = update_data

                # compute
                self._update_stats(input_len, prev_len, outputs, req_state, iteration_stats)

                # record
                scheduler_stats = get_current_scheduler_stats()
                for stat_logger in self.stat_loggers:
                    stat_logger.record(scheduler_stats=scheduler_stats, iteration_stats=iteration_stats)

                self.metrics_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f'Metrics handler background task failed: {e}')

    def queue_update(self, update_data: tuple):
        if not is_metrics_enabled() or self.metrics_queue is None:
            return

        self.metrics_queue.put_nowait(update_data)

    def increment_total_requests(self):
        increment_async_engine_scheduler_stats_total_req()

    def increment_finished_requests(self):
        increment_async_engine_scheduler_stats_finished_req()

    def _update_stats(self, input_len: int, prev_len: int, outputs: 'EngineOutput', req_state: RequestState,
                      iteration_stats: IterationStats):
        from lmdeploy.messages import ResponseType

        status = outputs.status
        metrics_info = outputs.metrics_info
        scheduler_raw_info = metrics_info.scheduler_raw_info

        # update scheduler stats
        scheduler_stats = get_current_scheduler_stats()
        scheduler_stats.num_running_reqs = scheduler_raw_info['locked']
        scheduler_stats.num_waiting_reqs = scheduler_raw_info['waiting'] + scheduler_raw_info['running']
        scheduler_stats.gpu_cache_usage = 1.0 - (scheduler_raw_info['free_gpu_blocks'] /
                                                 scheduler_raw_info['total_gpu_blocks'])

        # update from per-iteration outputs
        iteration_stats.update_from_output(engine_core_timestamp=metrics_info.engine_core_timestamp,
                                           engine_core_events=metrics_info.engine_core_events,
                                           num_prompt_tokens=input_len,
                                           num_new_generation_tokens=(outputs.num_token - prev_len),
                                           is_prefilling=(prev_len == 0),
                                           req_stats=req_state.stats)

        # update from finished request
        if status is ResponseType.FINISH:
            iteration_stats.update_from_finished_request(finish_reason=status,
                                                         num_prompt_tokens=input_len,
                                                         req_stats=req_state.stats)

        req_state.is_prefilling = False  # change to decode after first update


metrics_processor = MetricsProcessor()
