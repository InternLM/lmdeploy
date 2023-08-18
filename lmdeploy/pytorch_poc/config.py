from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_batches: int
    max_session_len: int
    recompute: bool = False


@dataclass
class CacheConfig:
    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int


@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    num_heads: int

    def get_head_size(self):
        return self.hidden_size // self.num_heads
