# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Any, Dict

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext
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
                 tokenizer: Any,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda'):
        """initialize Executor."""
        super().__init__(model_path=model_path,
                         model_config=model_config,
                         cache_config=cache_config,
                         backend_config=backend_config,
                         tokenizer=tokenizer,
                         dp=1,
                         tp=1,
                         adapters=adapters,
                         device_type=device_type)

        self.device_ctx = DeviceContext(device_type=device_type)
        self.model_agent = build_model_agent(model_path=model_path,
                                             model_config=model_config,
                                             cache_config=cache_config,
                                             backend_config=backend_config,
                                             tokenizer=tokenizer,
                                             device_ctx=self.device_ctx,
                                             adapters=adapters)

    def download_models(self):
        """download model."""
        raise NotImplementedError('Not Implemented.')

    def build_model(self):
        """build model."""
        self.model_agent.build_model()

    def gather_free_mem(self):
        """gather available memory."""
        return [self.model_agent.get_free_mem()]

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.model_agent.set_cache_config(cache_config)

    def set_model_config(self, model_config: ModelConfig):
        """set all cache config."""
        self.model_agent.set_model_config(model_config)

    def build_graph_runner(self):
        """build graph runner."""
        self.model_agent.build_graph_runner()

    def build_cache_engine(self):
        """build cache engine."""
        self.model_agent.build_cache_engine()

    def start(self, forward_event: asyncio.Event):
        """start engine loop."""
        self.model_agent.start(forward_event)

    def stop(self):
        """stop engine loop."""
        self.model_agent.stop()

    def release(self):
        """release resources."""
        self.model_agent.release()

    async def forward_async(self, inputs):
        """start forward."""
        self.model_agent.set_forward_inputs(inputs)

    async def get_output_async(self):
        """get output async."""
        return await self.model_agent.get_output_async()

    def get_input_processor(self):
        """get input processor."""
        return self.model_agent.get_input_processor()
