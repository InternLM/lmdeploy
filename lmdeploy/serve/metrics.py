# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
import pynvml
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info, disable_created_metrics

disable_created_metrics()


class IterTimer:
    """"The timer to count all the time of iteration."""

    def __init__(self, iterable):
        self._iterable = iterable
        self._duration = 0

    def __iter__(self):
        return self

    def __next__(self):
        start = time.perf_counter()
        item = next(iter(self._iterable))
        self._duration += (time.perf_counter() - start)
        return item

    def get_duration(self):
        """Get the whole duration of iteration.

        Known as model forwarding latency.
        """
        return self._duration

    def __aiter__(self):
        return self

    async def __anext__(self):
        start = time.perf_counter()
        item = await self._iterable.__anext__()
        self._duration += (time.perf_counter() - start)
        return item


@dataclass
class Stats:
    """Log system information."""
    # system status
    cpu_utilization: Optional[float] = None
    cpu_memory_used_bytes: Optional[float] = None
    gpu_utilization: Optional[Dict] = None
    gpu_memory_used_bytes: Optional[Dict] = None

    def refresh(self):
        """Fresh system status."""
        p = psutil.Process()
        self.cpu_utilization = psutil.cpu_percent()
        self.cpu_memory_used_bytes = p.memory_info().rss
        pynvml.nvmlInit()
        self.gpu_memory_used_bytes = {}
        self.gpu_utilization = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(i))
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_memory_used_bytes[str(i)] = str(mem_info.used)
            self.gpu_utilization[str(i)] = str(utilization.gpu)


def refresh_system(metrics):
    """A thread life long function to get hardware information."""
    while True:
        time.sleep(1)
        # Log to prometheus.
        stats = metrics.stats
        stats.refresh()
        # Info gpu stats
        metrics.info_gpu_utilization.info(stats.gpu_utilization)
        metrics.info_gpu_memory_used_bytes.info(stats.gpu_memory_used_bytes)
        # Set system stat gauges.
        metrics.gauge_cpu_utilization.set(stats.cpu_utilization)
        metrics.gauge_cpu_memory_used_bytes.set(stats.cpu_memory_used_bytes)


class Metrics:
    """The metrics for serving."""

    def __init__(self, applied: bool = False, labelnames: Optional[List[str]] = []):
        self.applied = applied
        # Unregister any existing lmdeploy collectors
        for collector in list(REGISTRY._collector_to_names):
            if hasattr(collector, '_name') and 'lmdeploy' in collector._name:
                REGISTRY.unregister(collector)

        # Config Information
        self.info_backend_config = Info(name='lmdeploy:backend_config', documentation='information of backend_config')

        # System stats
        self.info_gpu_utilization = Info(name='lmdeploy:gpu_utilization',
                                         documentation='GPU utilization. 1 means 100 percent usage.')
        self.info_gpu_memory_used_bytes = Info(name='lmdeploy:gpu_memory_used_bytes',
                                               documentation='GPU memory used bytes.')
        self.gauge_cpu_utilization = Gauge(name='lmdeploy:cpu_utilization',
                                           documentation='CPU utilization. 1 means 100 percent usage.',
                                           labelnames=labelnames)
        self.gauge_cpu_memory_used_bytes = Gauge(name='lmdeploy:cpu_memory_used_bytes',
                                                 documentation='CPU memory used bytes.',
                                                 labelnames=labelnames)

        # requests
        self.counter_request_success = Counter(name='lmdeploy:request_success',
                                               documentation='Number of successful requests.',
                                               labelnames=labelnames)
        self.counter_request_failure = Counter(name='lmdeploy:request_failure',
                                               documentation='Number of failed requests.',
                                               labelnames=labelnames)
        self.counter_request_total = Counter(name='lmdeploy:request_total',
                                             documentation='Number of total requests.',
                                             labelnames=labelnames)

        # latency metrics
        self.histogram_duration_queue = Histogram(
            name='lmdeploy:duration_queue_seconds',
            documentation=  # noqa
            'Avarate duration waiting in the queue of requests in s.',
            labelnames=labelnames,
        )
        self.histogram_duration_infer = Histogram(
            name='lmdeploy:duration_infer_seconds',
            documentation='Average inference time in s.',
            labelnames=labelnames,
        )
        self.histogram_duration_preprocess = Histogram(
            name='lmdeploy:duration_preprocess_seconds',
            documentation='Average duration of processing inputs in s.',
            labelnames=labelnames,
        )
        self.histogram_first_token_latency = Histogram(
            name='lmdeploy:first_token_latency_seconds',
            documentation='Average first token latency in s.',
            labelnames=labelnames,
        )
        self.stats = Stats()
        self.refresh_thread = threading.Thread(target=refresh_system, args=(self, ), daemon=True)
        self.refresh_thread.start()

    def info(self, backend_config: object) -> None:
        if self.applied:
            config_dict = {key: str(value) for key, value in dataclasses.asdict(backend_config).items()}
            self.info_backend_config.info(config_dict)

    def failure_frame(self):
        """log the failaure frame."""
        if self.applied:
            self.counter_request_failure.inc()
            self.counter_request_total.inc()

    def last_token_frame(self, iterator):
        """log the last token frame."""
        if self.applied:
            self.histogram_duration_infer.observe(iterator.get_duration())
            self.counter_request_success.inc()
            self.counter_request_total.inc()

    def insert_frame(self):
        """Insert a frame."""
        if self.applied:
            return time.time()
        return None

    def update_preprocess(self, start_frame):
        """Update preprocess duration."""
        if self.applied:
            self.histogram_duration_preprocess.observe(time.time() - start_frame)

    def update_queue_waiting(self, start_frame):
        """Update queue waiting time."""
        if self.applied:
            self.histogram_duration_queue.observe(time.time() - start_frame)

    def update_FTL(self, start_frame):
        """Update first token latency."""
        if self.applied:
            self.histogram_first_token_latency.observe(time.time() - start_frame)
