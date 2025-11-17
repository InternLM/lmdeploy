# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.model_agent import build_model_agent
from lmdeploy.utils import get_logger

from .base import ExecutorBase

logger = get_logger('lmdeploy')


class UniExecutor(ExecutorBase):
    """Single node single device Executor."""

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 misc_config: MiscConfig,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda'):
        """Initialize Executor."""
        super().__init__(model_path=model_path,
                         model_config=model_config,
                         cache_config=cache_config,
                         backend_config=backend_config,
                         dist_config=DistConfig(),
                         misc_config=misc_config,
                         adapters=adapters,
                         device_type=device_type)

        self.device_ctx = DeviceContext(device_type=device_type)
        self.model_agent = build_model_agent(model_path=model_path,
                                             model_config=model_config,
                                             cache_config=cache_config,
                                             backend_config=backend_config,
                                             misc_config=misc_config,
                                             device_ctx=self.device_ctx,
                                             adapters=adapters)

    def download_models(self):
        """Download model."""
        raise NotImplementedError('Not Implemented.')

    def build_model(self):
        """Build model."""
        self.model_agent.build_model()

    def gather_free_mem(self):
        """Gather available memory."""
        return [self.model_agent.get_free_mem()]

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        self.model_agent.set_cache_config(cache_config)

    def set_model_config(self, model_config: ModelConfig):
        """Set all cache config."""
        self.model_agent.set_model_config(model_config)

    def build_graph_runner(self):
        """Build graph runner."""
        self.model_agent.build_graph_runner()

    def build_cache_engine(self):
        """Build cache engine."""
        self.model_agent.build_cache_engine()

    def warmup(self):
        self.model_agent.warmup()

    def start(self, forward_event: asyncio.Event):
        """Start engine loop."""
        self.model_agent.start(forward_event)

    def stop(self):
        """Stop engine loop."""
        self.model_agent.stop()

    def release(self):
        """Release resources."""
        self.model_agent.release()

    async def forward_async(self, inputs):
        """Start forward."""
        self.model_agent.set_forward_inputs(inputs)
        # switch to task: ModelAgent._async_loop_inputs_preprocess
        await asyncio.sleep(0)

    async def get_output_async(self, dp_rank: int = 0):
        """Get output async."""
        assert dp_rank == 0
        return await self.model_agent.get_output_async()

    def get_input_processor(self):
        """Get input processor."""
        return self.model_agent.get_input_processor()

    """ PD Disaggregation API Begin """

    def p2p_initialize(self, init_request: DistServeInitRequest):
        """Init rdma link.

        note: return list to be composible with multiprocess executor like ray.
        """
        return [self.model_agent.cache_engine.p2p_initialize(init_request)]

    def p2p_connect(self, remote_engine_id: str, conn_request: List[DistServeKVTransferEndpointInfo]):
        """rdma_connect."""
        self.model_agent.cache_engine.p2p_connect(remote_engine_id, conn_request)

    async def migrate(self, batch: MigrationExecutionBatch):
        """KV Cache Migration."""
        return await self.model_agent.cache_engine.migrate(batch)

    """ PD Disaggregation API End """
