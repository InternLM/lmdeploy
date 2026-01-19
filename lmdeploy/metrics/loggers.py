# Copyright (c) OpenMMLab. All rights reserved.
# adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import numpy as np

from lmdeploy.metrics.stats import IterationStats, RequestStats, SchedulerStats, SpeculativeDecodingStats
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class StatLoggerBase(ABC):

    @abstractmethod
    def record_schedule(self, stats: SchedulerStats) -> None:
        ...

    @abstractmethod
    def record_iteration(self, stats: IterationStats) -> None:
        ...

    @abstractmethod
    def record_specdecode(self, stats: SpeculativeDecodingStats) -> None:
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
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        # spec decode
        self.num_drafts: int = 0
        self.num_draft_tokens: int = 0
        self.num_accepted_tokens: int = 0
        self.num_accepted_tokens_per_pos: np.ndarray = None

    def record_schedule(self, stats: SchedulerStats):
        self.last_scheduler_stats = stats

    def record_iteration(self, stats: IterationStats):
        # In the first iteration of a sequence, stats.prompt_tokens is the
        # prompt token number of a sequence. In subsequent iterations,
        # the value is 0. This enables cumulative counting in `total_prompt_tokens`
        self.total_prompt_tokens += stats.prompt_tokens
        self.total_generation_tokens += stats.new_generation_tokens

    def record_specdecode(self, stats: SpeculativeDecodingStats):
        """Record spec decoding stats."""
        if stats.num_drafts <= 0:
            return
        if self.num_accepted_tokens_per_pos is None:
            self.num_accepted_tokens_per_pos = np.zeros(stats.num_spec_tokens)
        self.num_drafts += stats.num_drafts
        self.num_draft_tokens += stats.num_draft_tokens
        self.num_accepted_tokens += stats.num_accepted_tokens
        self.num_accepted_tokens_per_pos += stats.num_accepted_tokens_per_pos

    def record_finish(self, stats: RequestStats):
        pass

    def log_spec_msg(self):
        """Get spec decoding logging msg."""
        if self.num_drafts == 0:
            return

        draft_acceptance_rate = (self.num_accepted_tokens / self.num_draft_tokens *
                                 100 if self.num_draft_tokens > 0 else float('nan'))

        # conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (self.num_accepted_tokens / self.num_drafts)

        acceptance_rates = self.num_accepted_tokens_per_pos / self.num_drafts
        rates_str = ', '.join(f'{p:.3f}' for p in acceptance_rates)

        log_msg = ('SpecDecoding metrics: '
                   f'Draft acceptance rate: {draft_acceptance_rate:.2f}%, '
                   f'Mean acceptance length: {mean_acceptance_length:.2f}, '
                   f'Accepted: {self.num_accepted_tokens} tokens, '
                   f'Drafted: {self.num_draft_tokens} tokens, '
                   f'Per-position acceptance rate: {rates_str}')
        return log_msg

    def log(self):
        now = time.perf_counter()
        spec_msg = self.log_spec_msg()

        # skip logging if no tokens were processed
        if self.total_prompt_tokens == 0 and self.total_generation_tokens == 0:
            self._reset(now)
            return

        # derive log information
        prompt_throughput = self.total_prompt_tokens / (now - self.last_log_time)
        generation_throughput = self.total_generation_tokens / (now - self.last_log_time)
        scheduler_stats = self.last_scheduler_stats
        self._reset(now)

        # format and print
        log_msg = (f"[{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} "
                   f'DP{self.dp_rank}] '
                   f'Avg prompt throughput: {prompt_throughput:.1f} tokens/s, '
                   f'Avg generation throughput: {generation_throughput:.1f} tokens/s, '
                   f'Finished: {scheduler_stats.num_finished_reqs} reqs, '
                   f'Unfinished: {scheduler_stats.num_total_reqs-scheduler_stats.num_finished_reqs} reqs, '
                   f'Running: {scheduler_stats.num_running_reqs} reqs, '
                   f'Waiting: {scheduler_stats.num_waiting_reqs} reqs, '
                   f'GPU KV cache usage: {scheduler_stats.gpu_cache_usage * 100 :.1f}%, '
                   f'Prefix cache hit rate: {scheduler_stats.prefix_cache_hit_rate * 100 :.1f}%')

        if spec_msg is not None:
            log_msg += ', ' + spec_msg
        print(log_msg, flush=True)


class PrometheusStatLogger(StatLoggerBase):

    def __init__(self, model_name: str, max_model_len: int, dp_rank: int = 0):
        try:
            import prometheus_client
            prometheus_client.disable_created_metrics()  # disable noisy creation timestamp gauge in prometheus
        except ImportError:
            raise ImportError(
                'To use metrics system , please install prometheus_client by `pip install prometheus_client`')

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
        # Scheduler stats
        #
        self.gauge_scheduler_finished = prometheus_client.Gauge(name='lmdeploy:num_requests_finished',
                                                                documentation='Number of current finished requests.',
                                                                labelnames=labelnames).labels(*labelvalues)

        self.gauge_scheduler_unfinished = prometheus_client.Gauge(
            name='lmdeploy:num_requests_unfinished',
            documentation='Number of current unfinished requests.',
            labelnames=labelnames).labels(*labelvalues)

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
                    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
                ],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_iter_token_latency = \
            prometheus_client.Histogram(
                name='lmdeploy:iter_token_latency',
                documentation='Histogram of inter-token latency',
                buckets=[
                    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
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

    def record_schedule(self, stats: SchedulerStats) -> None:
        """Report schedule metrics to prometheus."""
        self.gauge_scheduler_finished.set(stats.num_finished_reqs)
        self.gauge_scheduler_unfinished.set(stats.num_total_reqs - stats.num_finished_reqs)
        self.gauge_scheduler_running.set(stats.num_running_reqs)
        self.gauge_scheduler_waiting.set(stats.num_waiting_reqs)
        self.gauge_gpu_cache_usage.set(stats.gpu_cache_usage)

    def record_iteration(self, stats: IterationStats) -> None:
        """Report token-related metrics to prometheus."""

        self.counter_prompt_tokens.inc(stats.prompt_tokens)
        self.counter_generation_tokens.inc(stats.new_generation_tokens)
        self.histogram_iteration_tokens.observe(stats.prompt_tokens + stats.new_generation_tokens)

        if stats.ttft:
            self.histogram_time_to_first_token.observe(stats.ttft)

        if stats.tpot:
            self.histogram_time_per_output_token.observe(stats.tpot)

        if stats.itl:
            self.histogram_iter_token_latency.observe(stats.itl)

    def record_finish(self, stats: RequestStats) -> None:
        self.counter_request_success[stats.finish_reason].inc()
        self.histogram_e2e_time_request.observe(stats.e2e_latency)
        self.histogram_queue_time_request.observe(stats.queued_time_interval)
        self.histogram_prefill_time_request.observe(stats.prefill_time_interval)
        self.histogram_inference_time_request.observe(stats.inference_time_interval)
        self.histogram_decode_time_request.observe(stats.decode_time_interval)
        self.histogram_num_prompt_tokens_request.observe(stats.prompt_tokens)
        self.histogram_num_generation_tokens_request.observe(stats.generation_tokens)

    def record_specdecode(self, stats: SpeculativeDecodingStats) -> None:
        pass


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
