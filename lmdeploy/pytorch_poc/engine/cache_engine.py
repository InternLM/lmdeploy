# modify from: https://github.com/vllm-project/vllm
import torch
from lmdeploy.pytorch_poc.config import CacheConfig, ModelConfig


class CacheEngine:

    def __init__(self, cache_config: CacheConfig,
                 model_config: ModelConfig) -> None:
        self.cache_config = cache_config
        self.model_config = model_config

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.num_layers
        self.num_heads = model_config.num_heads

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]


def allocate_gpu_cache(self):
    pass


def allocate_cpu_cache(self):
    pass
