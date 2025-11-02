# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import gc
from typing import Any, Dict, List, Optional

from lmdeploy.pytorch.backends.selector import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.model_agent import build_model_agent
from lmdeploy.utils import get_logger

from .dist_utils import init_process_group, setup_master_addr

logger = get_logger('lmdeploy')


class WorkerWrapperBase:
    """Worker wrapper."""

    def __init__(
        self,
        model_path: str,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        model_config: ModelConfig,
        dist_config: DistConfig,
        misc_config: MiscConfig,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
        log_level: int = 30,
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.dist_config = dist_config
        self.misc_config = misc_config
        self.adapters = adapters
        self.device_type = device_type
        self.log_level = log_level
        self.dp = dist_config.dp
        self.tp = dist_config.tp
        self.world_size = dist_config.world_size
        self.device_type = device_type

        logger.setLevel(log_level)
        self.out_que: asyncio.Queue = None
        self._output_loop: asyncio.Task = None

        # frequently gc would cause latency spike
        # default threshold (700, 10, 10)
        gc.set_threshold(10000, 100, 100)

    def init_process_group(self, rank: int, master_addr: str = None, master_port: str = None):
        """Initialize process group."""
        self.rank = rank
        if self.world_size > 1:
            if master_addr is not None and master_port is not None:
                setup_master_addr(master_addr, master_port)

            init_process_group(rank, self.world_size)

        ccl_backend = get_backend(self.device_type).ccl_backend()
        self.dist_ctx = DistContext.build(self.rank, self.dist_config, ccl_backend)

    def pack_output(self, output: Dict):
        """Pack output."""
        return output

    async def _get_outputs_loop(self):
        """Get outputs loop."""
        assert self.out_que is not None
        while True:
            ret = await self.get_output_async()
            ret = self.pack_output(ret)
            self.out_que.put_nowait(ret)

    async def get_outputs(self):
        """Get outputs."""
        assert self.out_que is not None
        qsize = self.out_que.qsize()
        if qsize > 0:
            outs = []
            for _ in range(qsize):
                outs.append(self.out_que.get_nowait())
            return outs
        else:
            return [await self.out_que.get()]

    def build_model(self):
        """Build model."""
        self.device_ctx = DeviceContext(device_type=self.device_type)

        self.model_agent = build_model_agent(model_path=self.model_path,
                                             model_config=self.model_config,
                                             cache_config=self.cache_config,
                                             backend_config=self.backend_config,
                                             misc_config=self.misc_config,
                                             device_ctx=self.device_ctx,
                                             dist_ctx=self.dist_ctx,
                                             adapters=self.adapters)
        self.model_agent.build_model()

    def get_free_mem(self):
        """Gather free mem."""
        return self.model_agent.get_free_mem()

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        self.model_agent.set_cache_config(cache_config)

    def set_model_config(self, model_config: ModelConfig):
        """Set all model config."""
        self.model_agent.set_model_config(model_config)

    def build_graph_runner(self):
        """Build graph runner."""
        self.model_agent.build_graph_runner()

    def build_cache_engine(self):
        """Build cache engine."""
        self.model_agent.build_cache_engine()

    def update_params(self, request: Any):
        """Update params."""
        self.model_agent.update_params(request)

    def warmup(self):
        """warmup."""
        self.model_agent.warmup()

    def sleep(self, level: int = 1):
        """Sleep."""
        self.model_agent.sleep(level)

    def wakeup(self, tags: Optional[List[str]] = None):
        """Wakeup."""
        self.model_agent.wakeup(tags)

    def get_input_processor(self):
        """Build cache engine."""
        return self.model_agent.get_input_processor()

    def start(self):
        """Start engine loop."""
        self.model_agent.start()
        event_loop = asyncio.get_event_loop()
        self.out_que = asyncio.Queue()
        self._output_loop = event_loop.create_task(self._get_outputs_loop(), name='GetOutputsLoop')

    def stop(self):
        """Stop engine loop."""
        self.model_agent.stop()
        if self._output_loop is not None:
            self._output_loop.cancel()

    async def stop_async(self):
        await self.model_agent.stop_async()
        if self._output_loop is not None:
            self._output_loop.cancel()
            try:
                await self._output_loop
            except asyncio.CancelledError:
                logger.debug('worker output loop cancelled.')

    async def forward_async(self, inputs):
        """Start forward."""
        self.model_agent.set_forward_inputs(inputs)

    async def get_output_async(self):
        """Get output async."""
        ret = await self.model_agent.get_output_async()
        return ret

    def release(self):
        """Stop engine loop."""
        self.model_agent.release()

    """ PD Disaggregation API Begin """

    def p2p_initialize(self, init_request: DistServeInitRequest):
        return self.model_agent.cache_engine.p2p_initialize(init_request)

    def p2p_connect(self, remote_engine_id: str, conn_request: List[DistServeKVTransferEndpointInfo]):
        return self.model_agent.cache_engine.p2p_connect(remote_engine_id, conn_request)

    async def migrate(self, inputs: MigrationExecutionBatch):
        return await self.model_agent.cache_engine.migrate(inputs)

    """ PD Disaggregation API End """
