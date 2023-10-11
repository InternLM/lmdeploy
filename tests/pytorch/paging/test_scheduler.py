import pytest

from lmdeploy.pytorch_poc.config import CacheConfig, SchedulerConfig
# from lmdeploy.pytorch_poc.messages import SchedulerSession
from lmdeploy.pytorch_poc.paging.scheduler import Scheduler

# import torch


class TestScheduler:

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 4

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks):
        yield CacheConfig(block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks)

    @pytest.fixture
    def scheduler_config(self):
        yield SchedulerConfig(max_batches=4,
                              max_session_len=128,
                              max_request_output_len=64)

    @pytest.fixture
    def scheduler(self, cache_config, scheduler_config):
        yield Scheduler(scheduler_config=scheduler_config,
                        cache_config=cache_config)
