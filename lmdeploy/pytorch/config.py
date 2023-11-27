# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    """Config of scheduler."""

    max_batches: int
    max_session_len: int
    max_request_output_len: int
    recompute: bool = False


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
