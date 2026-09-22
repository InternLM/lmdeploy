import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


@pytest.fixture
def block_size():
    return 16


@pytest.fixture
def num_cpu_blocks():
    return 4


@pytest.fixture
def num_gpu_blocks():
    return 16


@pytest.fixture
def max_batch_size():
    return 4


@pytest.fixture
def cache_config(block_size, num_cpu_blocks, num_gpu_blocks, max_batch_size):
    return CacheConfig(max_batches=max_batch_size,
                       block_size=block_size,
                       num_cpu_blocks=num_cpu_blocks,
                       num_gpu_blocks=num_gpu_blocks,
                       enable_prefix_caching=True)


@pytest.fixture
def scheduler_config(max_batch_size):
    return SchedulerConfig(max_batches=max_batch_size,
                           max_session_len=128,
                           max_request_output_len=64,
                           eviction_type='recompute')


@pytest.fixture
def seq_meta(block_size):
    return SequenceMeta(block_size, strategy=ARSequenceStrategy())


@pytest.fixture
def scheduler(cache_config, scheduler_config, seq_meta):
    return Scheduler(scheduler_config=scheduler_config,
                     cache_config=cache_config,
                     seq_meta=seq_meta)


@pytest.fixture
def ssm_cache_config(block_size, num_cpu_blocks, num_gpu_blocks, max_batch_size):
    return CacheConfig(max_batches=max_batch_size,
                       block_size=block_size,
                       num_cpu_blocks=num_cpu_blocks,
                       num_gpu_blocks=num_gpu_blocks,
                       enable_prefix_caching=True,
                       num_state_caches=max_batch_size + 1 + 8,
                       prefix_cache_state_budget=8,
                       states_shapes=[((1, ), torch.float32)])


@pytest.fixture
def ssm_scheduler(ssm_cache_config, scheduler_config, seq_meta):
    return Scheduler(scheduler_config=scheduler_config,
                     cache_config=ssm_cache_config,
                     seq_meta=seq_meta)


@pytest.fixture
def block_mgr(scheduler):
    return scheduler.block_manager


@pytest.fixture
def block_trie(scheduler):
    return scheduler.block_trie
