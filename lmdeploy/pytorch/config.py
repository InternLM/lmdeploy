# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """PyTorch Engine Config.

    Args:
        model_name (str): name of the given model.
        tp (int): Tensor Parallelism. default 1.
        session_len (int): Max session length. Default 4096.
        max_batch_szie: (int): Max batch size. Default 128.
        eviction_type (str): What action to perform when kv cache
            is full, ['recompute', 'copy'], Default 'recompute'.
        prefill_interval (int): Interval to perform prefill,
            Default 16.
        block_size (int): paging cache block size, default 64.
        num_cpu_blocks (int): Num cpu blocks. If num is 0, cache
            would be allocate according to current environment.
        num_gpu_blocks (int): Num gpu blocks. If num is 0, cache
            would be allocate according to current environment.
    """
    model_name: str = ''
    tp: int = 1
    session_len: int = None
    max_batch_size: int = 128
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    block_size: int = 64
    num_cpu_blocks: int = 0
    num_gpu_blocks: int = 0


@dataclass
class SchedulerConfig:
    """Config of scheduler."""

    max_batches: int
    max_session_len: int
    max_request_output_len: int = 512
    eviction_type: str = 'recompute'
    prefill_interval: int = 16


@dataclass
class CacheConfig:
    """Config of key value cache."""

    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int


@dataclass
class ModelConfig:
    """Config of model."""

    hidden_size: int
    num_layers: int
    num_heads: int
    bos_token_id: int
    eos_token_id: int
    dtype: str
    multi_query_attention: bool = False
    json_config: dict = field(default_factory=dict)

    def get_head_size(self):
        """get head size."""
        return self.hidden_size // self.num_heads
