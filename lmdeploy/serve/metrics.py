# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
import pynvml
from prometheus_client import REGISTRY, Gauge, Info, disable_created_metrics

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
    """Created by LLMEngine for use by StatLogger."""

    # request stats
    request_success: int = 0
    request_failure: int = 0
    request_total: int = 0
    request_responding: int = 0
    request_waiting: int = 0

    # latency stats
    duration_queue: float = 0
    duration_infer: float = 0
    duration_preprocess: float = 0
    duration_postprocess: float = 0
    first_token_latency: float = 0

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

    def __init__(self,
                 applied: bool = False,
                 labelnames: Optional[List[str]] = []):
        self.applied = applied
        # Unregister any existing lmdeploy collectors
        for collector in list(REGISTRY._collector_to_names):
            if hasattr(collector, '_name') and 'lmdeploy' in collector._name:
                REGISTRY.unregister(collector)

        # Config Information
        self.info_backend_config = Info(
            name='lmdeploy:backend_config',
            documentation='information of backend_config')

        # System stats
        self.info_gpu_utilization = Info(
            name='lmdeploy:gpu_utilization',
            documentation='GPU utilization. 1 means 100 percent usage.')
        self.info_gpu_memory_used_bytes = Info(
            name='lmdeploy:gpu_memory_used_bytes',
            documentation='GPU memory used bytes.')
        self.gauge_cpu_utilization = Gauge(
            name='lmdeploy:cpu_utilization',
            documentation='CPU utilization. 1 means 100 percent usage.',
            labelnames=labelnames)
        self.gauge_cpu_memory_used_bytes = Gauge(
            name='lmdeploy:cpu_memory_used_bytes',
            documentation='CPU memory used bytes.',
            labelnames=labelnames)

        # requests
        self.gauge_request_success = Gauge(
            name='lmdeploy:request_success',
            documentation='Number of successful requests.',
            labelnames=labelnames)
        self.gauge_request_failure = Gauge(
            name='lmdeploy:request_failure',
            documentation='Number of failed requests.',
            labelnames=labelnames)
        self.gauge_request_total = Gauge(
            name='lmdeploy:request_total',
            documentation='Number of total requests.',
            labelnames=labelnames)

        # latency metrics
        self.gauge_duration_queue = Gauge(
            name='lmdeploy:duration_queue',
            documentation=  # noqa
            'Avarate duration waiting in the queue of requests in s.',
            labelnames=labelnames,
        )
        self.gauge_duration_infer = Gauge(
            name='lmdeploy:duration_infer',
            documentation='Average inference time in s.',
            labelnames=labelnames,
        )
        self.gauge_duration_preprocess = Gauge(
            name='lmdeploy:duration_preprocess',
            documentation='Average duration of processing inputs in s.',
            labelnames=labelnames,
        )
        self.gauge_duration_postprocess = Gauge(
            name='lmdeploy:duration_postprocess',
            documentation='Average duration of processing outputs in s.',
            labelnames=labelnames,
        )
        self.gauge_first_token_latency = Gauge(
            name='lmdeploy:first_token_latency',
            documentation='Average first token latency in s.',
            labelnames=labelnames,
        )
        self.stats = Stats()
        self.refresh_thread = threading.Thread(target=refresh_system,
                                               args=(self, ),
                                               daemon=True)
        self.refresh_thread.start()

    def info(self, backend_config: object) -> None:
        if self.applied:
            config_dict = {
                key: str(value)
                for key, value in dataclasses.asdict(backend_config).items()
            }
            self.info_backend_config.info(config_dict)

    def failure_frame(self):
        """log the failaure frame."""
        if self.applied:
            self.stats.request_failure += 1
            self.stats.request_total += 1

    def last_token_frame(self, iterator):
        """log the last token frame."""
        if self.applied:
            self.stats.duration_infer += iterator.get_duration()
            self.stats.request_success += 1
            self.stats.request_total += 1
            self.log()

    def insert_frame(self):
        """Insert a frame."""
        if self.applied:
            return time.time()
        return None

    def update_postprocess(self, start_frame):
        """Update postprocess duration."""
        if self.applied:
            self.stats.duration_postprocess += time.time() - start_frame

    def update_preprocess(self, start_frame):
        """Update preprocess duration."""
        if self.applied:
            self.stats.duration_preprocess += time.time() - start_frame

    def update_queue_waiting(self, start_frame):
        """Update queue waiting time."""
        if self.applied:
            self.stats.duration_queue += time.time() - start_frame

    def update_FTL(self, start_frame):
        """Update first token latency."""
        if self.applied:
            self.stats.first_token_latency += time.time() - start_frame

    def log(self) -> None:
        """Called by LLMEngine.

        Logs to prometheus and tracked stats every iteration. Logs to Stdout
        every self.local_interval seconds.
        """
        stats = self.stats
        # Add to request counters.
        self.gauge_request_total.set(stats.request_total)
        self.gauge_request_success.set(stats.request_success)
        self.gauge_request_failure.set(stats.request_failure)

        # duration gauges
        self.gauge_duration_infer.set(stats.duration_infer /
                                      stats.request_total)
        self.gauge_duration_queue.set(stats.duration_queue /
                                      stats.request_total)
        self.gauge_duration_preprocess.set(stats.duration_preprocess /
                                           stats.request_total)
        self.gauge_duration_postprocess.set(stats.duration_postprocess /
                                            stats.request_total)
        self.gauge_first_token_latency.set(stats.first_token_latency /
                                           stats.request_total)
