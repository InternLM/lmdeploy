# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Any, Dict

from lmdeploy.pytorch.backends.selector import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.model_agent import build_model_agent
from lmdeploy.utils import get_logger

from .dist_utils import init_process_group, setup_master_addr

logger = get_logger('lmdeploy')


class WorkerWrapperBase:
    """worker wrapper."""

    def __init__(
        self,
        model_path: str,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        model_config: ModelConfig,
        dp: int,
        tp: int,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
        tokenizer: Any = None,
        log_level: int = 30,
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.tokenizer = tokenizer
        self.adapters = adapters
        self.device_type = device_type
        self.log_level = log_level
        self.dp = dp
        self.tp = tp
        self.world_size = tp * dp
        self.device_type = device_type

        logger.setLevel(log_level)
        self.out_que: asyncio.Queue = None
        self._output_loop: asyncio.Task = None

    def init_process_group(self, rank: int, master_addr: str = None, master_port: str = None):
        """initialize process group."""
        self.rank = rank
        if self.world_size > 1:
            if master_addr is not None and master_port is not None:
                setup_master_addr(master_addr, master_port)

            init_process_group(rank, self.world_size)

        ccl_backend = get_backend(self.device_type).ccl_backend()
        self.dist_ctx = DistContext.build(self.rank, self.tp, self.dp, ccl_backend)

    def pack_output(self, output: Dict):
        """pack output."""
        return output

    async def _get_outputs_loop(self):
        """get outputs loop."""
        assert self.out_que is not None
        while True:
            ret = await self.get_output_async()
            ret = self.pack_output(ret)
            self.out_que.put_nowait(ret)

    async def get_outputs(self):
        """get outputs."""
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
        """build model."""
        self.device_ctx = DeviceContext(device_type=self.device_type)

        self.model_agent = build_model_agent(model_path=self.model_path,
                                             model_config=self.model_config,
                                             cache_config=self.cache_config,
                                             backend_config=self.backend_config,
                                             tokenizer=self.tokenizer,
                                             device_ctx=self.device_ctx,
                                             dist_ctx=self.dist_ctx,
                                             adapters=self.adapters)
        self.model_agent.build_model()

    def get_free_mem(self):
        """gather free mem."""
        return self.model_agent.get_free_mem()

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.model_agent.set_cache_config(cache_config)

    def set_model_config(self, model_config: ModelConfig):
        """set all model config."""
        self.model_agent.set_model_config(model_config)

    def build_graph_runner(self):
        """build graph runner."""
        self.model_agent.build_graph_runner()

    def build_cache_engine(self):
        """build cache engine."""
        self.model_agent.build_cache_engine()

    def get_input_processor(self):
        """build cache engine."""
        return self.model_agent.get_input_processor()

    def start(self):
        """start engine loop."""
        self.model_agent.start()
        event_loop = asyncio.get_event_loop()
        self.out_que = asyncio.Queue()
        self._output_loop = event_loop.create_task(self._get_outputs_loop(), name='GetOutputsLoop')

    def stop(self):
        """stop engine loop."""
        self.model_agent.stop()
        if self._output_loop is not None:
            self._output_loop.cancel()

    async def forward_async(self, inputs):
        """start forward."""
        self.model_agent.set_forward_inputs(inputs)

    async def get_output_async(self):
        """get output async."""
        ret = await self.model_agent.get_output_async()
        return ret

    def release(self):
        """stop engine loop."""
        self.model_agent.release()
