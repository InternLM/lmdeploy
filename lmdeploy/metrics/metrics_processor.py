# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

from lmdeploy.messages import ResponseType, ScheduleMetrics
from lmdeploy.pytorch.utils import singleton
from lmdeploy.utils import get_logger

from .stats import SchedulerStats

logger = get_logger('lmdeploy')


@singleton
class MetricsProcessor():
    """Metrics processor."""

    def __init__(self):
        """Init metrics processor."""
        self.enable_metrics: bool = False
        self.scheduler_stats = SchedulerStats()
        self.stat_loggers = []
        self.metrics_queue: asyncio.Queue = None
        self.metrics_handler: asyncio.Task = None

    def start_metrics_handler(self, enable_metrics: bool):
        """Start metrics handler."""
        self.enable_metrics = enable_metrics
        if enable_metrics and self.metrics_handler is None:
            self.metrics_queue = asyncio.Queue()
            self.metrics_handler = asyncio.create_task(self._run_metrics_handler())
            logger.info('Metrics handler task started.')

    async def stop_metrics_handler(self):
        """Stop metrics handler."""
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
                # fetch data from the queue
                update_data = await self.metrics_queue.get()
                outputs, req_stats, iteration_stats, specdecode_stats = update_data

                # update request stats
                if outputs and outputs.req_metrics:
                    # when users visit "/abort_request" endpoint, `req_metrics` might be None
                    req_stats.update_from_events(outputs.req_metrics.engine_events)

                # update iteration stats
                # some attributes of req_stats will also be updated, e.g., lastest_token_time
                iteration_stats.update_from_output(outputs, req_stats)

                # update spec decode stats
                if specdecode_stats is not None:
                    specdecode_stats.update_from_output(outputs)

                # record iteration stats
                for stat_logger in self.stat_loggers:
                    stat_logger.record_iteration(iteration_stats)
                    if specdecode_stats is not None:
                        stat_logger.record_specdecode(specdecode_stats)

                # record finished request stats
                if outputs.status == ResponseType.FINISH:
                    for stat_logger in self.stat_loggers:
                        stat_logger.record_finish(req_stats)

                self.metrics_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f'Metrics handler background task failed: {e}')

    async def update_schedule_stats(self, schedule_metrics: ScheduleMetrics):
        """Update schedule stats."""
        self.scheduler_stats.update_from_schedule_metrics(schedule_metrics)
        # record schedule stats
        for stat_logger in self.stat_loggers:
            stat_logger.record_schedule(self.scheduler_stats)

    def queue_update(self, update_data: tuple):
        """Queue update."""
        if not self.enable_metrics or self.metrics_queue is None:
            return
        self.metrics_queue.put_nowait(update_data)

    def increment_total_requests(self):
        """Increment total requests."""
        self.scheduler_stats.num_total_reqs += 1

    def increment_finished_requests(self):
        """Increment finished requests."""
        self.scheduler_stats.num_finished_reqs += 1


metrics_processor = MetricsProcessor()
