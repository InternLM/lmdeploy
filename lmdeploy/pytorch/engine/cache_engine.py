# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Dict, List, Literal, Tuple

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.utils import get_logger

from ..config import CacheConfig, ModelConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]

logger = get_logger('lmdeploy')


class CacheEngine:
    """Host and Device memory maintainer.

    Args:
        cache_config (CacheConfig): config of the cache information.
        model_config (ModelConfig): config of the model.
        rank (int): distribution rank, 0 on non-distributed environment.
        world_size (int): distribution world size, 1 on non-distributed
            environment.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if rank == 0:
            logger.info(f'build CacheEngine with config:{cache_config}')
        self.rank = rank
        self.world_size = world_size

        self.cache_config = cache_config
        self.model_config = model_config

        self.block_size = cache_config.block_size
        self.num_layers = model_config.num_layers
        self.kv_cache_dtype = model_config.dtype
        if cache_config.quant_policy > 0:
            self.kv_cache_dtype = torch.uint8

        # Initialize the cache.
        self.local_gpu_cache = self.allocate_gpu_cache()
        self.local_cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

        logger.debug(
            f'Initialize cache engine with {cache_config.num_gpu_blocks}'
            f' gpu blocks and {cache_config.num_cpu_blocks} cpu blocks.')

    @property
    def cpu_cache(self):
        """gpu cache."""
        return self.local_cpu_cache

    @property
    def gpu_cache(self):
        """gpu cache."""
        return self.local_gpu_cache

    @property
    def num_gpu_blocks(self):
        """num gpu blocks."""
        return self.cache_config.num_gpu_blocks

    @property
    def num_cpu_blocks(self):
        """num gpu blocks."""
        return self.cache_config.num_cpu_blocks

    @classmethod
    def _get_key_block_shape_impl(cls,
                                  model_config: ModelConfig,
                                  block_size: int,
                                  head_size: int,
                                  world_size: int = 1,
                                  quant_policy: Literal[0, 4, 8] = 0,
                                  local: bool = True):
        """get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads
        if local and not model_config.multi_query_attention:
            assert num_heads % world_size == 0, \
                f'num_heads: {num_heads}, world_size: {world_size}'
            num_heads = num_heads // world_size
        if quant_policy == 4:  # pack head_dim to uint8
            assert head_size % 2 == 0, \
                f'head_size: {head_size}, quant_policy: {quant_policy}'
            head_size = head_size // 2
        return attn_backend.get_k_block_shape(block_size, num_heads, head_size,
                                              dtype)

    @classmethod
    def _get_value_block_shape_impl(cls,
                                    model_config: ModelConfig,
                                    block_size: int,
                                    head_size: int,
                                    world_size: int = 1,
                                    quant_policy: Literal[0, 4, 8] = 0,
                                    local: bool = True):
        """get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads
        if local and not model_config.multi_query_attention:
            assert num_heads % world_size == 0, \
                f'num_heads: {num_heads}, world_size: {world_size}'
            num_heads = num_heads // world_size
        if quant_policy == 4:  # pack head_dim to uint8
            assert head_size % 2 == 0, \
                f'head_size: {head_size}, quant_policy: {quant_policy}'
            head_size = head_size // 2

        return attn_backend.get_v_block_shape(block_size, num_heads, head_size,
                                              dtype)

    def get_key_block_shape(self, local: bool = False) -> Tuple[int, int, int]:
        """get shape of key block."""
        head_size = self.model_config.k_head_dim
        if head_size is None:
            head_size = self.model_config.head_dim
        return self._get_key_block_shape_impl(
            self.model_config,
            block_size=self.block_size,
            head_size=head_size,
            world_size=self.world_size,
            quant_policy=self.cache_config.quant_policy,
            local=local,
        )

    def get_value_block_shape(self,
                              local: bool = False) -> Tuple[int, int, int]:
        """get shape of value block."""
        head_size = self.model_config.v_head_dim
        if head_size is None:
            head_size = self.model_config.head_dim
        return self._get_value_block_shape_impl(
            self.model_config,
            block_size=self.block_size,
            head_size=head_size,
            world_size=self.world_size,
            quant_policy=self.cache_config.quant_policy,
            local=local,
        )

    def allocate_gpu_cache(self):
        """allocate caches on GPU."""
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape(local=True)
        value_block_shape = self.get_value_block_shape(local=True)

        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.kv_cache_dtype,
                device='cuda',
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.kv_cache_dtype,
                device='cuda',
            )
            if self.cache_config.quant_policy in (4, 8):
                key_scales_zeros = torch.empty(
                    size=(self.num_gpu_blocks, *key_block_shape[:-1], 2),
                    dtype=self.model_config.dtype,
                    device='cuda',
                )
                value_scales_zeros = torch.empty(
                    size=(self.num_gpu_blocks, *value_block_shape[:-1], 2),
                    dtype=self.model_config.dtype,
                    device='cuda',
                )
                gpu_cache.append((key_blocks, value_blocks, key_scales_zeros,
                                  value_scales_zeros))
            else:
                gpu_cache.append((key_blocks, value_blocks))

        return gpu_cache

    def allocate_cpu_cache(self):
        """allocate caches on Host."""
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape(local=True)
        value_block_shape = self.get_value_block_shape(local=True)

        # TODO: pin memory might need be banned on wsl
        pin_memory = True

        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.kv_cache_dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.kv_cache_dtype,
                pin_memory=pin_memory,
            )
            if self.cache_config.quant_policy in (4, 8):
                key_scales_zeros = torch.empty(
                    size=(self.num_cpu_blocks, *key_block_shape[:-1], 2),
                    dtype=self.model_config.dtype,
                    pin_memory=pin_memory,
                )
                value_scales_zeros = torch.empty(
                    size=(self.num_cpu_blocks, *value_block_shape[:-1], 2),
                    dtype=self.model_config.dtype,
                    pin_memory=pin_memory,
                )
                cpu_cache.append((key_blocks, value_blocks, key_scales_zeros,
                                  value_scales_zeros))
            else:
                cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    @torch.inference_mode()
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
                    if isinstance(dst_key_cache[dst_id], torch.Tensor):
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

    @classmethod
    def get_cache_block_size(cls,
                             block_size: int,
                             model_config: ModelConfig,
                             world_size: int = 1,
                             quant_policy: int = 0) -> int:
        """Get the required cache size of the model.

        Args:
            block_size (int): The token numbers of the block.
            model_config (ModelConfig): The config of the model.

        Return:
            int: Required memory size in bytes.
        """
        num_layers = model_config.num_layers
        key_head_size = model_config.k_head_dim
        value_head_size = model_config.v_head_dim
        if key_head_size is None:
            key_head_size = model_config.head_dim
        if value_head_size is None:
            value_head_size = model_config.head_dim
        key_shape = cls._get_key_block_shape_impl(
            model_config,
            block_size=block_size,
            head_size=key_head_size,
            world_size=world_size,
            local=True,
            quant_policy=quant_policy,
        )
        value_shape = cls._get_value_block_shape_impl(
            model_config,
            block_size=block_size,
            head_size=value_head_size,
            world_size=world_size,
            quant_policy=quant_policy,
            local=True,
        )
        if quant_policy == 0:
            dtype = model_config.dtype
            key_block = torch.empty(key_shape, dtype=dtype, device='meta')
            value_block = torch.empty(value_shape, dtype=dtype, device='meta')
            mem_key_block = key_block.numel() * key_block.element_size()
            mem_value_block = value_block.numel() * value_block.element_size()
        elif quant_policy in (4, 8):
            key_block = torch.empty(key_shape,
                                    dtype=torch.uint8,
                                    device='meta')
            value_block = torch.empty(value_shape,
                                      dtype=torch.uint8,
                                      device='meta')
            key_scale_zero_block = torch.empty((*key_shape[:-1], 2),
                                               dtype=model_config.dtype,
                                               device='meta')
            value_scale_zero_block = torch.empty((*value_shape[:-1], 2),
                                                 dtype=model_config.dtype,
                                                 device='meta')
            mem_key_block = key_block.numel() * key_block.element_size(
            ) + key_scale_zero_block.numel(
            ) * key_scale_zero_block.element_size()
            mem_value_block = value_block.numel() * value_block.element_size(
            ) + value_scale_zero_block.numel(
            ) * value_scale_zero_block.element_size()
        else:
            raise ValueError(f'unsupported quant_policy {quant_policy}')

        total = num_layers * (mem_key_block + mem_value_block)
        return total
