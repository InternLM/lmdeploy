# Copyright (c) OpenMMLab. All rights reserved.
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import pickle
from typing import TYPE_CHECKING, Dict
from lmdeploy.pytorch import envs as _envs
import os

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.utils import get_logger
from lmdeploy.pytorch.ray import RayContext
from .base_worker import EngineWorkerBase

from .base import MPEngine

logger = get_logger('lmdeploy')

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine


class RayEngineWorker(EngineWorkerBase):

    def __init__(self, model_path: str, tokenizer: object, engine_config: PytorchEngineConfig = None, log_level:int = 30, **kwargs) -> None:
        """Initialize Ray engine worker."""
        from lmdeploy.pytorch.engine.engine import Engine
        from lmdeploy.tokenizer import Tokenizer
        logger.setLevel(log_level)
        # create engine
        if engine_config is not None:
            engine_config.enable_mp_engine = False
        if tokenizer is None:
            tokenizer = Tokenizer(model_path)
        engine = Engine.from_pretrained(model_path, tokenizer=tokenizer, engine_config=engine_config, **kwargs)
        super().__init__(engine)


def _update_runtime_envs(runtime_env: Dict):
    """Update runtime envs."""
    new_envs = _envs.get_all_envs()
    env_vars: Dict = runtime_env.get('env_vars', {})
    env_vars.update(new_envs)
    runtime_env['env_vars'] = env_vars
    return runtime_env


class RayMPEngine(MPEngine):

    def __init__(self, model_path: str, tokenizer: object, engine_config: PytorchEngineConfig = None, **kwargs) -> None:
        """Initialize mp engine."""
        self.ray_ctx = self._init_ray(engine_config)
        placement_group = self.ray_ctx.get_placement_group()
        self.placement_group = placement_group

        self.worker = self._create_worker(model_path, tokenizer, engine_config, log_level=logger.level, **kwargs)
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

    def _create_worker(self, model_path: str, tokenizer: object, engine_config: PytorchEngineConfig = None, **kwargs):
        """Create a Ray worker."""
        try:
            pickle.dumps(tokenizer)
        except Exception:
            logger.warning('Failed to pickle tokenizer. It would be created in subprocess.')
            tokenizer = None

        bundle_id = 0
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=self.placement_group,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_id,
        )

        runtime_env = dict()
        _update_runtime_envs(runtime_env)
        worker = ray.remote(
            num_cpus=0,
            num_gpus=0.01,
            scheduling_strategy=scheduling_strategy,
            runtime_env=runtime_env,
        )(RayEngineWorker).remote(model_path, tokenizer, engine_config, **kwargs)

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
        method = getattr(self.worker, func)
        async for result in method.remote(*args, **kwargs):
            yield await result

    def close(self) -> None:
        """Close mp engine."""
        logger.info('Closing mp engine.')
        self._collective_rpc('close')
        
        ray.util.remove_placement_group(self.placement_group)
        if ray.is_initialized():
            ray.shutdown()
            logger.debug('Shutdown Ray.')

    def start_loop(self) -> None:
        """Start mp engine loop."""
