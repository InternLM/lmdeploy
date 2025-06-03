# Copyright (c) OpenMMLab. All rights reserved.
# adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import numpy as np
import prometheus_client

from lmdeploy.metrics.stats import IterationStats, SchedulerStats
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

prometheus_client.disable_created_metrics()


class StatLoggerBase(ABC):

    @abstractmethod
    def record(self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]):
        ...

    def log(self):  # noqa
        pass


class LoggingStatLogger(StatLoggerBase):

    def __init__(self, dp_rank: int = 0):
        self.dp_rank = dp_rank
        self._reset(time.perf_counter())
        self.last_scheduler_stats = SchedulerStats()

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(iteration_stats.num_prompt_tokens)
        self.num_generation_tokens.append(iteration_stats.num_generation_tokens)

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        # Compute summary metrics for tracked stats
        return float(np.sum(tracked_stats) / (now - self.last_log_time))

    def record(self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]):
        """Log Stats to standard output."""

        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        self.last_scheduler_stats = scheduler_stats

    def log(self):
        now = time.perf_counter()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(self.num_generation_tokens, now)

        self._reset(now)

        scheduler_stats = self.last_scheduler_stats

        # Format and print output.
        log_msg = (f"[{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} "
                   f'DP{self.dp_rank}] '
                   f'Avg prompt throughput: {prompt_throughput:.1f} tokens/s, '
                   f'Avg generation throughput: {generation_throughput:.1f} tokens/s, '
                   f'Running: {scheduler_stats.num_running_reqs} reqs, '
                   f'Waiting: {scheduler_stats.num_waiting_reqs} reqs, '
                   f'GPU KV cache usage: {scheduler_stats.gpu_cache_usage * 100 :.1f}%')
        print(log_msg)


class PrometheusStatLogger(StatLoggerBase):

    def __init__(self, model_name: str, max_model_len: int, dp_rank: int = 0):
        self.dp_rank = dp_rank

        # unregister any existing lmdeploy collectors
        for collector in list(prometheus_client.REGISTRY._collector_to_names):
            if hasattr(collector, '_name') and 'lmdeploy' in collector._name:
                prometheus_client.REGISTRY.unregister(collector)

        # config information
        self.info_backend_config = prometheus_client.Info(name='lmdeploy:backend_config',
                                                          documentation='information of backend_config')

        labelnames = ['model_name', 'engine']
        labelvalues = [model_name, str(dp_rank)]

        #
        # Scheduler state
        #
        self.gauge_scheduler_running = prometheus_client.Gauge(
            name='lmdeploy:num_requests_running',
            documentation='Number of requests in model execution batches.',
            labelnames=labelnames).labels(*labelvalues)

        self.gauge_scheduler_waiting = prometheus_client.Gauge(
            name='lmdeploy:num_requests_waiting',
            documentation='Number of requests waiting to be processed.',
            labelnames=labelnames).labels(*labelvalues)

        #
        # GPU cache
        #
        self.gauge_gpu_cache_usage = prometheus_client.Gauge(
            name='lmdeploy:gpu_cache_usage_perc',
            documentation='GPU KV-cache usage. 1 means 100 percent usage.',
            labelnames=labelnames).labels(*labelvalues)

        #
        # Counters
        #
        self.counter_prompt_tokens = prometheus_client.Counter(name='lmdeploy:prompt_tokens_total',
                                                               documentation='Number of prefill tokens processed.',
                                                               labelnames=labelnames).labels(*labelvalues)

        self.counter_generation_tokens = prometheus_client.Counter(
            name='lmdeploy:generation_tokens_total',
            documentation='Number of generation tokens processed.',
            labelnames=labelnames).labels(*labelvalues)

        from lmdeploy.messages import ResponseType
        self.counter_request_success: dict[ResponseType, prometheus_client.Counter] = {}
        counter_request_success_base = prometheus_client.Counter(
            name='lmdeploy:request_success_total',
            documentation='Count of successfully processed requests.',
            labelnames=labelnames + ['finished_reason'])
        for reason in ResponseType:
            self.counter_request_success[reason] = counter_request_success_base.labels(*(labelvalues + [str(reason)]))

        #
        # Histograms of counts
        #
        self.histogram_num_prompt_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_prompt_tokens',
                documentation='Number of prefill tokens processed.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_num_generation_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_generation_tokens',
                documentation='Number of generation tokens processed.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_iteration_tokens = \
            prometheus_client.Histogram(
                name='lmdeploy:iteration_tokens_total',
                documentation='Histogram of number of tokens per engine_step.',
                buckets=[
                    1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                    16384
                ],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_max_num_generation_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_max_num_generation_tokens',
                documentation='Histogram of maximum number of requested generation tokens.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_n_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_params_n',
                documentation='Histogram of the n request parameter.',
                buckets=[1, 2, 5, 10, 20],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_max_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_params_max_tokens',
                documentation='Histogram of the max_tokens request parameter.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        #
        # Histogram of timing intervals
        #
        self.histogram_time_to_first_token = \
            prometheus_client.Histogram(
                name='lmdeploy:time_to_first_token_seconds',
                documentation='Histogram of time to first token in seconds.',
                buckets=[
                    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0,
                    640.0, 2560.0
                ],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_time_per_output_token = \
            prometheus_client.Histogram(
                name='lmdeploy:time_per_output_token_seconds',
                documentation='Histogram of time per output token in seconds.',
                buckets=[
                    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
                ],
                labelnames=labelnames).labels(*labelvalues)

        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0,
            960.0, 1920.0, 7680.0
        ]
        self.histogram_e2e_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:e2e_request_latency_seconds',
                documentation='Histogram of e2e request latency in seconds.',
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_queue_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_queue_time_seconds',
                documentation='Histogram of time spent in WAITING phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_inference_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_inference_time_seconds',
                documentation='Histogram of time spent in RUNNING phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_prefill_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_prefill_time_seconds',
                documentation='Histogram of time spent in PREFILL phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_decode_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_decode_time_seconds',
                documentation='Histogram of time spent in DECODE phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)

    def record(self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]):
        """Log to prometheus."""

        self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
        self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)

        self.gauge_gpu_cache_usage.set(scheduler_stats.gpu_cache_usage)

        if iteration_stats is None:
            return

        self.counter_prompt_tokens.inc(iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens.inc(iteration_stats.num_generation_tokens)
        self.histogram_iteration_tokens.observe(iteration_stats.num_prompt_tokens +
                                                iteration_stats.num_generation_tokens)

        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token.observe(ttft)

        for tpot in iteration_stats.time_per_output_tokens_iter:
            self.histogram_time_per_output_token.observe(tpot)

        for finished_request in iteration_stats.finished_requests:
            self.counter_request_success[finished_request.finish_reason].inc()
            self.histogram_e2e_time_request.observe(finished_request.e2e_latency)
            self.histogram_queue_time_request.observe(finished_request.queued_time)
            self.histogram_prefill_time_request.observe(finished_request.prefill_time)
            self.histogram_inference_time_request.observe(finished_request.inference_time)
            self.histogram_decode_time_request.observe(finished_request.decode_time)
            self.histogram_num_prompt_tokens_request.observe(finished_request.num_prompt_tokens)
            self.histogram_num_generation_tokens_request.observe(finished_request.num_generation_tokens)


def build_buckets(mantissa_lst: List[int], max_value: int) -> List[int]:
    """Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values until the value exceeds the specified maximum."""
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    return build_buckets([1, 2, 5], max_value)


def setup_loggers(model_name: str, max_model_len: int, engine_num: int):
    """Setup loggers."""
    stat_loggers: List[List[StatLoggerBase]] = []
    for dp_rank in range(engine_num):
        stat_loggers.append([
            LoggingStatLogger(dp_rank=dp_rank),
            PrometheusStatLogger(model_name=model_name, max_model_len=max_model_len, dp_rank=dp_rank)
        ])

    return stat_loggers
