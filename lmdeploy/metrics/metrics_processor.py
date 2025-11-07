# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field

from lmdeploy.messages import ResponseType, ScheduleMetrics
from lmdeploy.utils import get_logger

from .stats import SchedulerStats

logger = get_logger('lmdeploy')


@dataclass
class MetricsContext:
    enable_metrics: bool = False
    scheduler_stats: SchedulerStats = field(default_factory=SchedulerStats)


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
                outputs, req_state, iteration_stats = update_data

                # update request state according the engine events
                if outputs and outputs.req_metrics:
                    # when users visit "/abort_request" endpoint, `req_metrics` might be None
                    req_state.update_from_events(outputs.req_metrics.engine_events)

                # update iteration stats based on outputs and request state.
                # some attributes of req_state will also be updated, e.g., lastest_token_time
                iteration_stats.update_from_output(outputs, req_state)

                # record iteration stats
                for stat_logger in self.stat_loggers:
                    stat_logger.record_iteration(iteration_stats)

                if outputs.status == ResponseType.FINISH:
                    # record finished request stats
                    for stat_logger in self.stat_loggers:
                        stat_logger.record_finish(req_state.finish_stats)

                self.metrics_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f'Metrics handler background task failed: {e}')

    async def udpate_schedule_stats(self, schedule_metrics: ScheduleMetrics):
        stats = get_current_scheduler_stats()
        stats.update_from_schedule_metrics(schedule_metrics)
        # record schedule stats
        for stat_logger in self.stat_loggers:
            stat_logger.record_schedule(stats)

    def queue_update(self, update_data: tuple):
        if not is_metrics_enabled() or self.metrics_queue is None:
            return

        self.metrics_queue.put_nowait(update_data)

    def increment_total_requests(self):
        increment_async_engine_scheduler_stats_total_req()

    def increment_finished_requests(self):
        increment_async_engine_scheduler_stats_finished_req()


metrics_processor = MetricsProcessor()
