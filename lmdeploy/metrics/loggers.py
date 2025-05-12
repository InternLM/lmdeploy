# Copyright (c) OpenMMLab. All rights reserved.
# adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py

import time
from abc import ABC, abstractmethod
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

    def __init__(self, engine_index: int = 0):
        self.engine_index = engine_index
        self._reset(time.monotonic())
        self.last_scheduler_stats = SchedulerStats()

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: list[int] = []
        self.num_generation_tokens: list[int] = []

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(iteration_stats.num_prompt_tokens)
        self.num_generation_tokens.append(iteration_stats.num_generation_tokens)

    def _get_throughput(self, tracked_stats: list[int], now: float) -> float:
        # Compute summary metrics for tracked stats
        return float(np.sum(tracked_stats) / (now - self.last_log_time))

    def record(self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]):
        """Log Stats to standard output."""

        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        self.last_scheduler_stats = scheduler_stats

    def log(self):
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(self.num_generation_tokens, now)

        self._reset(now)

        scheduler_stats = self.last_scheduler_stats

        # Format and print output.
        logger.info(
            'Avg prompt throughput: %.1f tokens/s, '
            'Avg generation throughput: %.1f tokens/s, '
            'Running: %d reqs, Waiting: %d reqs, '
            'GPU KV cache usage: %.1f%%, ',
            prompt_throughput,
            generation_throughput,
            scheduler_stats.num_running_reqs,
            scheduler_stats.num_waiting_reqs,
            scheduler_stats.gpu_cache_usage * 100,
        )


class PrometheusStatLogger(StatLoggerBase):

    def __init__(self, labelnames: Optional[List[str]] = []):

        # unregister any existing lmdeploy collectors
        for collector in list(prometheus_client.REGISTRY._collector_to_names):
            if hasattr(collector, '_name') and 'lmdeploy' in collector._name:
                prometheus_client.REGISTRY.unregister(collector)

        max_model_len = 4096  # FIXME, hard code now, get from model config

        # config Information
        self.info_backend_config = prometheus_client.Info(name='lmdeploy:backend_config',
                                                          documentation='information of backend_config')

        #
        # Scheduler state
        #
        self.gauge_scheduler_running = prometheus_client.Gauge(
            name='lmdeploy:num_requests_running',
            documentation='Number of requests in model execution batches.',
            labelnames=labelnames)

        self.gauge_scheduler_waiting = prometheus_client.Gauge(
            name='lmdeploy:num_requests_waiting',
            documentation='Number of requests waiting to be processed.',
            labelnames=labelnames)

        #
        # Counters
        #
        self.counter_prompt_tokens = prometheus_client.Counter(name='lmdeploy:prompt_tokens_total',
                                                               documentation='Number of prefill tokens processed.',
                                                               labelnames=labelnames)

        self.counter_generation_tokens = prometheus_client.Counter(
            name='lmdeploy:generation_tokens_total',
            documentation='Number of generation tokens processed.',
            labelnames=labelnames)

        # from lmdeploy.messages import ResponseType
        # self.counter_request_success: dict[ResponseType,
        #                                    prometheus_client.Counter] = {}
        # counter_request_success_base = prometheus_client.Counter(
        #     name="lmdeploy:request_success_total",
        #     documentation="Count of successfully processed requests.",
        #     labelnames=labelnames + ["finished_reason"])
        # for reason in FinishReason:
        #     self.counter_request_success[
        #         reason] = counter_request_success_base.labels(*(labelvalues +
        #                                                         [str(reason)]))

        #
        # Histograms of counts
        #
        self.histogram_num_prompt_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_prompt_tokens',
                documentation='Number of prefill tokens processed.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames)

        self.histogram_num_generation_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_generation_tokens',
                documentation='Number of generation tokens processed.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames)

        # FIXME, build_cudagraph_buckets
        # self.histogram_iteration_tokens = \
        #     prometheus_client.Histogram(
        #         name="lmdeploy:iteration_tokens_total",
        #         documentation="Histogram of number of tokens per engine_step.",
        #         buckets=build_cudagraph_buckets(vllm_config),
        #         labelnames=labelnames)

        self.histogram_max_num_generation_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_max_num_generation_tokens',
                documentation='Histogram of maximum number of requested generation tokens.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames)

        self.histogram_n_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_params_n',
                documentation='Histogram of the n request parameter.',
                buckets=[1, 2, 5, 10, 20],
                labelnames=labelnames)

        self.histogram_max_tokens_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_params_max_tokens',
                documentation='Histogram of the max_tokens request parameter.',
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames)

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
                labelnames=labelnames)

        self.histogram_time_per_output_token = \
            prometheus_client.Histogram(
                name='lmdeploy:time_per_output_token_seconds',
                documentation='Histogram of time per output token in seconds.',
                buckets=[
                    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
                ],
                labelnames=labelnames)

        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0,
            960.0, 1920.0, 7680.0
        ]
        self.histogram_e2e_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:e2e_request_latency_seconds',
                documentation='Histogram of e2e request latency in seconds.',
                buckets=request_latency_buckets,
                labelnames=labelnames)
        self.histogram_queue_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_queue_time_seconds',
                documentation='Histogram of time spent in WAITING phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames)
        self.histogram_inference_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_inference_time_seconds',
                documentation='Histogram of time spent in RUNNING phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames)
        self.histogram_prefill_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_prefill_time_seconds',
                documentation='Histogram of time spent in PREFILL phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames)
        self.histogram_decode_time_request = \
            prometheus_client.Histogram(
                name='lmdeploy:request_decode_time_seconds',
                documentation='Histogram of time spent in DECODE phase for request.',
                buckets=request_latency_buckets,
                labelnames=labelnames)

    # def log_metrics_info(self, type: str, config_obj: SupportsMetricsInfo):
    #     metrics_info = config_obj.metrics_info()

    #     name, documentation = None, None
    #     if type == "cache_config":
    #         name = "lmdeploy:cache_config_info"
    #         documentation = "Information of the LLMEngine CacheConfig"
    #     assert name is not None, f"Unknown metrics info type {type}"

    #     # Info type metrics are syntactic sugar for a gauge permanently set to 1
    #     # Since prometheus multiprocessing mode does not support Info, emulate
    #     # info here with a gauge.
    #     info_gauge = prometheus_client.Gauge(
    #         name=name,
    #         documentation=documentation,
    #         labelnames=metrics_info.keys()).labels(**metrics_info)
    #     info_gauge.set(1)

    def record(self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]):
        """Log to prometheus."""

        self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
        self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)
        # self.gauge_gpu_cache_usage.set(scheduler_stats.gpu_cache_usage) # TODO

        if iteration_stats is None:
            return
        else:
            print(iteration_stats.__dict__)

        self.counter_prompt_tokens.inc(iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens.inc(iteration_stats.num_generation_tokens)
        self.counter_generation_tokens.inc(iteration_stats.num_generation_tokens)
        # self.histogram_iteration_tokens.observe(
        #     iteration_stats.num_prompt_tokens + \
        #     iteration_stats.num_generation_tokens)

        # import pdb; pdb.set_trace()
        for ttft in iteration_stats.time_to_first_tokens_iter:
            print(f'ttft: {ttft}')
            self.histogram_time_to_first_token.observe(ttft)

        # import pdb; pdb.set_trace()
        for tpot in iteration_stats.time_per_output_tokens_iter:
            print(f'tpot: {tpot}')
            self.histogram_time_per_output_token.observe(tpot)

        # import pdb; pdb.set_trace()
        for finished_request in iteration_stats.finished_requests:
            # self.counter_request_success[finished_request.finish_reason].inc()
            self.histogram_e2e_time_request.observe(finished_request.e2e_latency)
            self.histogram_queue_time_request.observe(finished_request.queued_time)
            self.histogram_prefill_time_request.observe(finished_request.prefill_time)
            self.histogram_inference_time_request.observe(finished_request.inference_time)
            self.histogram_decode_time_request.observe(finished_request.decode_time)
            self.histogram_num_prompt_tokens_request.observe(finished_request.num_prompt_tokens)
            self.histogram_num_generation_tokens_request.observe(finished_request.num_generation_tokens)
            # self.histogram_max_tokens_request.observe(
            #     finished_request.max_tokens_param)


def build_buckets(mantissa_lst: list[int], max_value: int) -> list[int]:
    """Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values until the value exceeds the specified maximum."""
    exponent = 0
    buckets: list[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> list[int]:
    """
    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    return build_buckets([1, 2, 5], max_value)
