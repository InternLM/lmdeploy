# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Dict, List, Tuple

import torch
from torch.distributed._tensor import DeviceMesh

from lmdeploy.pytorch_poc.config import CacheConfig, ModelConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CacheEngine:
    """Host and Device memory maintainer.

    Args:
        cache_config (CacheConfig): config of the cache information.
        model_config (ModelConfig): config of the model.
        rank (int): distribution rank, 0 on non-distributed environment.
        world_size (int): distribution world size, 1 on non-distributed
            environment.
        device_mesh (DeviceMesh): distribution device mesh.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        rank: int = 0,
        world_size: int = 1,
        device_mesh: DeviceMesh = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        if device_mesh is None and self.world_size > 1:
            device_mesh = DeviceMesh('cuda', list(range(self.world_size)))
        self.device_mesh = device_mesh

        self.cache_config = cache_config
        self.model_config = model_config

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.num_layers
        self.num_heads = model_config.num_heads
        self.dtype = model_config.dtype

        # Initialize the cache.
        self.local_gpu_cache = self.allocate_gpu_cache()
        self.local_cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    @property
    def gpu_cache(self):
        """gpu cache."""
        return self.local_gpu_cache

    def get_key_block_shape(self, local: bool = False) -> Tuple[int, int, int]:
        """get shape of key block."""
        num_heads = self.num_heads
        if local and not self.model_config.multi_query_attention:
            assert self.num_heads % self.world_size == 0, \
                f'num_heads: {self.num_heads}, world_size: {self.world_size}'
            num_heads = self.num_heads // self.world_size
        return (
            self.block_size,
            num_heads,
            self.head_size,
        )

    def get_value_block_shape(self,
                              local: bool = False) -> Tuple[int, int, int]:
        """get shape of value block."""
        num_heads = self.num_heads
        if local and not self.model_config.multi_query_attention:
            assert self.num_heads % self.world_size == 0, \
                f'num_heads: {self.num_heads}, world_size: {self.world_size}'
            num_heads = self.num_heads // self.world_size
        return (
            self.block_size,
            num_heads,
            self.head_size,
        )

    def allocate_gpu_cache(self):
        """allocate caches on GPU."""
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape(local=True)
        value_block_shape = self.get_value_block_shape(local=True)

        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device='cuda',
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device='cuda',
            )
            gpu_cache.append((key_blocks, value_blocks))

        return gpu_cache

    def allocate_cpu_cache(self):
        """allocate caches on Host."""
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()

        # TODO: pin memory might need be banned on wsl
        pin_memory = True

        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(self, src: List[KVCache], dst: List[KVCache],
              src_to_dst: Dict[int, int]):
        """Move caches from src memory to dst memory.

        Args:
            src (List[KVCache]): Source cache.
            dst (List[KVCache]): Destination cache.
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]

                for src_id, dst_id in src_to_dst.items():
                    dst_key_cache[dst_id].copy_(src_key_cache[src_id])
                    dst_value_cache[dst_id].copy_(src_value_cache[src_id])

                    event = self.events[i]
                    event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        """Move cache from Host to Device.

        Args:
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        self._swap(self.local_cpu_cache, self.local_gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        """Move cache from Device to Host.

        Args:
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        self._swap(self.local_gpu_cache, self.local_cpu_cache, src_to_dst)

    @staticmethod
    def get_cache_block_size(block_size: int,
                             model_config: ModelConfig) -> int:
        """Get the required cache size of the model.

        Args:
            block_size (int): The token numbers of the block.
            model_config (ModelConfig): The config of the model.

        Return:
            int: Required memory size in bytes.
        """
        head_size = model_config.get_head_size()
        num_layers = model_config.num_layers
        num_heads = model_config.num_heads

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)

        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    """get size of the given dtype.

    Args:
        dtype (torch.dtype): Data type.

    Return:
        int: size in bytes.
    """
    return torch.tensor([], dtype=dtype).element_size()
