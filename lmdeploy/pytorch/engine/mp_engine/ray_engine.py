# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.ray import RayContext, get_device_str, get_resource_kwargs
from lmdeploy.utils import get_logger

from .base import MPEngine
from .base_worker import EngineOutputGather, EngineWorkerBase

logger = get_logger('lmdeploy')


class RayEngineWorker(EngineWorkerBase):

    def __init__(self,
                 model_path: str,
                 engine_config: PytorchEngineConfig = None,
                 log_level: int = 30,
                 **kwargs) -> None:
        """Initialize Ray engine worker."""
        from lmdeploy.pytorch.engine.engine import Engine
        logger.setLevel(log_level)
        # create engine
        if engine_config is not None:
            engine_config.enable_mp_engine = False
        engine = Engine.from_pretrained(model_path, engine_config=engine_config, **kwargs)
        super().__init__(engine)

        self._stream_id = 0
        self._stream_aiter = dict()
        self._stream_task = dict()
        self._engine_output_gather = EngineOutputGather()

    async def _stream_task_wrapper(self, stream_id: int, func: str, *args, **kwargs):
        """Create a stream task."""
        method = getattr(self, func)
        event = self._stream_aiter[stream_id][0]
        async for result in method(*args, **kwargs):
            self._engine_output_gather.add(stream_id, result)
            self._stream_aiter[stream_id][1] = (result, False)
            event.set()
        self._stream_aiter[stream_id][1] = (result, True)
        event.set()

    def create_stream_task(self, func, *args, **kwargs):
        """Create a stream task."""
        stream_id = self._stream_id
        self._stream_id += 1
        event_loop = asyncio.get_event_loop()
        self._stream_aiter[stream_id] = [asyncio.Event(), None]
        task = event_loop.create_task(self._stream_task_wrapper(stream_id, func, *args, **kwargs))
        self._stream_task[stream_id] = task

        return stream_id

    async def get_stream_task_result(self, stream_id: int):
        """Get the result of a stream task."""
        assert stream_id in self._stream_aiter, f'Stream id {stream_id} not found.'
        stopped = False

        event = self._stream_aiter[stream_id][0]
        await event.wait()
        result, stopped = self._stream_aiter[stream_id][1]
        event.clear()

        result = self._engine_output_gather.pop(stream_id, result)

        if stopped:
            self._stream_aiter.pop(stream_id, None)
            self._stream_task.pop(stream_id, None)
        return result, stopped


def _update_runtime_envs(runtime_env: Dict):
    """Update runtime envs."""
    new_envs = _envs.get_all_envs()
    env_vars: Dict = runtime_env.get('env_vars', {})
    env_vars.update(new_envs)
    runtime_env['env_vars'] = env_vars
    return runtime_env


class RayMPEngine(MPEngine):

    def __init__(self, model_path: str, engine_config: PytorchEngineConfig = None, **kwargs) -> None:
        """Initialize mp engine."""
        self.ray_ctx = self._init_ray(engine_config)
        placement_group = self.ray_ctx.get_placement_group()
        self.placement_group = placement_group

        self.worker = self._create_worker(model_path, engine_config, log_level=logger.level, **kwargs)
        super().__init__()

    def _init_ray(self, engine_config: PytorchEngineConfig = None):
        """Initialize Ray."""
        if engine_config is None:
            engine_config = PytorchEngineConfig()

        device_type = engine_config.device_type if engine_config else 'cuda'
        dp = engine_config.dp if engine_config else 1
        world_size = engine_config.tp if dp <= 1 else 1

        ray_ctx = RayContext(world_size, dp=dp, device_type=device_type)
        return ray_ctx

    def _create_worker(self, model_path: str, engine_config: PytorchEngineConfig = None, **kwargs):
        """Create a Ray worker."""
        bundle_id = 0 if len(_envs.ray_external_pg_bundles) == 0 else _envs.ray_external_pg_bundles[0]
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=self.placement_group,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_id,
        )

        runtime_env = dict()
        _update_runtime_envs(runtime_env)
        device_str = get_device_str(engine_config.device_type)
        resource_kwargs = get_resource_kwargs(device_str=device_str, resource_used=0.01)
        worker = ray.remote(
            num_cpus=0,
            **resource_kwargs,
            scheduling_strategy=scheduling_strategy,
            runtime_env=runtime_env,
        )(RayEngineWorker).remote(model_path, engine_config, **kwargs)

        return worker

    def _collective_rpc(self, func, *args, **kwargs):
        """Collective rpc call."""
        method = getattr(self.worker, func)
        return ray.get(method.remote(*args, **kwargs))

    async def _collective_rpc_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        method = getattr(self.worker, func)
        return await method.remote(*args, **kwargs)

    async def _collective_rpc_streaming_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        # ray generator would try cache every result, which is too verbose.
        stream_id = await self._collective_rpc_async('create_stream_task', func, *args, **kwargs)

        stopped = False
        while not stopped:
            result, stopped = await self._collective_rpc_async('get_stream_task_result', stream_id)
            yield result

    def close(self) -> None:
        """Close mp engine."""
        logger.info('Closing mp engine.')
        self._collective_rpc('close')
        self.ray_ctx.shutdown()

    def start_loop(self) -> None:
        """Start mp engine loop."""
