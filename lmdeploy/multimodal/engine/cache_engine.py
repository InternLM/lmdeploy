# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List, Optional, Tuple

import torch

from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

FEATURE_BLOCK_SHAPE = (256, 4096)


class EncoderCacheEngine:
    """Manages the memory pool for image features.

    This engine allocates and manages a contiguous block of GPU memory
    to store image embeddings transferred from an encoder. It is adapted for
    an encoder-LLM separated architecture.
    Args:
        rank (int): Distributed rank.
        tp_rank (int): Tensor parallelism rank.
        world_size (int): Distributed world size.
    """

    def __init__(
        self,
        num_gpu_blocks: int = 128,
        rank: int = 0,
        tp_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.tp_rank = tp_rank

        self.feature_dtype = torch.bfloat16
        self._num_gpu_blocks = num_gpu_blocks

        self.encoder_gpu_cache = self._allocate_gpu_cache()

        self.migration_backend_impl: Optional[MigrationBackendImpl] = None

        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        self.events = torch.cuda.Event()

        # for memory block management
        self.free_blocks = list(range(num_gpu_blocks))
        logger.debug(f'Initialize feature cache engine with {self.num_gpu_blocks} gpu blocks.')

    @property
    def free_block_count(self) -> int:
        """Number of free blocks available in the cache."""
        return len(self.free_blocks)

    @property
    def gpu_cache(self) -> torch.Tensor:
        """The GPU feature pool tensor."""
        return self.encoder_gpu_cache

    @property
    def num_gpu_blocks(self) -> int:
        """Number of GPU blocks."""
        return self._num_gpu_blocks

    @staticmethod
    def get_feature_block_shape() -> Tuple[int, int]:
        """Get the shape of a single image feature block."""
        return FEATURE_BLOCK_SHAPE

    def _allocate_cache(self, num_blocks: int, device: torch.device) -> torch.Tensor:
        """Allocate the memory pool on the specified device."""
        block_shape = self.get_feature_block_shape()

        # allocate a large contiguous tensor as the encoder cache
        encoder_cache = torch.empty(
            size=(num_blocks, *block_shape),
            dtype=self.feature_dtype,
            device=device,
        )
        return encoder_cache

    def _allocate_gpu_cache(self) -> torch.Tensor:
        """Allocate the feature pool on the GPU."""
        return self._allocate_cache(self.num_gpu_blocks, 'cuda')

    @classmethod
    def get_cache_block_size(cls) -> int:
        """Get the memory size in bytes of a single feature block."""

        shape = cls.get_feature_block_shape()
        dtype = torch.bfloat16

        meta_tensor = torch.empty(shape, dtype=dtype, device='meta')
        return meta_tensor.numel() * meta_tensor.element_size()

    """ Methods for Disaggregation Begin. """

    def p2p_initialize(self, migration_init_request: DistServeInitRequest) -> List[DistServeKVTransferEndpointInfo]:
        if not self.migration_backend_impl:
            self.migration_backend_impl: MigrationBackendImpl = MIGRATION_BACKENDS.module_dict['DLSlime']()
        migration_init_request.rank = self.rank
        self.migration_backend_impl.p2p_initialize(migration_init_request)

        t = self.encoder_gpu_cache
        if t.numel() > 0:
            register_mr_request = DistServeRegisterMRMessage(
                protocol=migration_init_request.protocol,
                remote_engine_id=migration_init_request.remote_engine_id,
                mr_key='encoder_cache',  # fix memory registration key
                addr=t.data_ptr(),
                offset=t.storage_offset(),
                length=t.numel() * t.itemsize)
            self.migration_backend_impl.register_memory_region(register_mr_request)

        return [
            DistServeKVTransferEndpointInfo(protocol=migration_init_request.protocol,
                                            endpoint_info=json.dumps(
                                                self.migration_backend_impl.endpoint_info(
                                                    migration_init_request.remote_engine_id,
                                                    migration_init_request.protocol)))
        ]

    def p2p_connect(self, remote_engine_id: str, migration_conn_request: List[DistServeKVTransferEndpointInfo]):
        self.migration_backend_impl.p2p_connect(remote_engine_id, migration_conn_request[self.tp_rank])

    """ Methods for Disaggregation End. """
