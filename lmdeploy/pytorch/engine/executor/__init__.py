# Copyright (c) OpenMMLab. All rights reserved.
from logging import Logger
from typing import Dict

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.utils import get_logger

from .base import ExecutorBase


def get_distributed_executor_backend(world_size: int, dp: int, device_type: str, logger: Logger = None):
    """Get distributed executor backend."""
    from lmdeploy.pytorch.backends import get_backend

    def _log_info(message: str):
        if logger is not None:
            logger.info(message)

    def _log_and_set_backend(message: str, executor_backend: str):
        """Log and set backend."""
        message += f' distributed_executor_backend={executor_backend}.'
        _log_info(message)
        return executor_backend

    executor_backend = envs.executor_backend
    if executor_backend is not None:
        return _log_and_set_backend('found environment LMDEPLOY_EXECUTOR_BACKEND.', executor_backend)

    if world_size == 1:
        return 'uni'

    if dp > 1:
        executor_backend = 'ray'
        return _log_and_set_backend(f'dp={dp}.', 'ray')

    backend = get_backend(device_type)
    if not backend.support_ray():
        return _log_and_set_backend(f'device={device_type} does not support ray.', 'mp')
    else:
        return 'ray'

    # TODO: fix mp hanging, do not delete the comment.
    # device_count = backend.device_count()
    # if device_count is None:
    #     return _log_and_set_backend(f'device={device_type} can not get device_count.', 'mp')

    # if device_count < world_size:
    #     executor_backend = 'ray'
    #     return _log_and_set_backend(f'local device_count({device_count})<world_size({world_size}),', 'ray')
    # else:
    #     executor_backend = 'mp'
    #     return _log_and_set_backend(f'local device_count({device_count})>=world_size({world_size}),', 'mp')


def build_executor(
    model_path: str,
    cache_config: CacheConfig,
    backend_config: BackendConfig,
    dist_config: DistConfig,
    misc_config: MiscConfig,
    adapters: Dict[str, str] = None,
    device_type: str = 'cuda',
    distributed_executor_backend: str = None,
    dtype: str = 'auto',
    specdecode_config: SpecDecodeConfig = None,
) -> ExecutorBase:
    """Build model agent executor."""
    logger = get_logger('lmdeploy')
    dp = dist_config.dp
    world_size = dist_config.world_size

    model_config = ModelConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        hf_overrides=misc_config.hf_overrides,
        dist_config=dist_config,
        is_draft_model=False,
        spec_method=None if specdecode_config is None else specdecode_config.method,
    )

    if distributed_executor_backend is None:
        distributed_executor_backend = get_distributed_executor_backend(world_size, dp, device_type, logger)

    if dp > 1:
        assert distributed_executor_backend == 'ray', (
            'dp>1 requires distributed_executor_backend="ray", ',
            f'get distributed_executor_backend="{distributed_executor_backend}"')

    if misc_config.empty_init:
        assert distributed_executor_backend == 'ray', (
            'empty_init requires distributed_executor_backend="ray", ',
            f'get distributed_executor_backend="{distributed_executor_backend}"')

    if distributed_executor_backend is not None:
        logger.info(f'Build <{distributed_executor_backend}> executor.')
    if distributed_executor_backend == 'uni':
        assert world_size == 1, 'uni executor only support world_size==1.'
        from .uni_executor import UniExecutor
        return UniExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            specdecode_config=specdecode_config,
        )
    elif distributed_executor_backend == 'mp':
        from .mp_executor import MPExecutor
        logger.warning('MPExecutor will be deprecated in future releases, please use RayExecutor instead.')
        return MPExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            dist_config=dist_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            specdecode_config=specdecode_config,
        )
    elif distributed_executor_backend == 'ray':
        from .ray_executor import RayExecutor
        return RayExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            dist_config=dist_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            dtype=dtype,
            specdecode_config=specdecode_config,
        )
    else:
        raise RuntimeError(f'Unsupported distributed_executor_backend: {distributed_executor_backend}.')
