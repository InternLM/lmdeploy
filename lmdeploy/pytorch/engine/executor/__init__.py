# Copyright (c) OpenMMLab. All rights reserved.
from logging import Logger
from typing import Any, Dict

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from .base import ExecutorBase


def get_distributed_executor_backend(world_size: int, device_type: str, logger: Logger = None):
    """get distributed executor backend."""

    def _log_info(message):
        if logger is not None:
            logger.info(message)

    from lmdeploy.pytorch.backends import get_backend
    if world_size == 1:
        return None

    backend = get_backend(device_type)
    if not backend.support_ray():
        executor_backend = 'mp'
        _log_info(f'device={device_type} does not support ray. '
                  f'distributed_executor_backend={executor_backend}.')
        return executor_backend

    device_count = backend.device_count()
    if device_count is None:
        executor_backend = 'mp'
        _log_info(f'device={device_type} can not get device_count. '
                  f'distributed_executor_backend={executor_backend}.')
        return executor_backend

    if device_count < world_size:
        executor_backend = 'ray'
        _log_info(f'local device_count({device_count})<world_size({world_size}), '
                  f'distributed_executor_backend={executor_backend}.')
    else:
        executor_backend = 'mp'
        _log_info(f'local device_count({device_count})>=world_size({world_size}), '
                  f'distributed_executor_backend={executor_backend}.')
    return executor_backend


def build_executor(model_path: str,
                   cache_config: CacheConfig,
                   backend_config: BackendConfig,
                   tokenizer: Any,
                   dp: int = 1,
                   tp: int = 1,
                   adapters: Dict[str, str] = None,
                   device_type: str = 'cuda',
                   distributed_executor_backend: str = None,
                   dtype: str = 'auto') -> ExecutorBase:
    """build model agent executor."""
    logger = get_logger('lmdeploy')

    world_size = dp * tp
    model_config = ModelConfig.from_pretrained(model_path, trust_remote_code=True, dtype=dtype, tp=tp)

    if distributed_executor_backend is None:
        distributed_executor_backend = get_distributed_executor_backend(world_size, device_type, logger)

    if world_size == 1:
        from .uni_executor import UniExecutor
        return UniExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            tokenizer=tokenizer,
            adapters=adapters,
            device_type=device_type,
        )
    elif distributed_executor_backend == 'mp':
        from .mp_executor import MPExecutor
        return MPExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            tokenizer=tokenizer,
            dp=dp,
            tp=tp,
            adapters=adapters,
            device_type=device_type,
        )
    elif distributed_executor_backend == 'ray':
        from .ray_executor import RayExecutor
        return RayExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            tokenizer=tokenizer,
            dp=dp,
            tp=tp,
            adapters=adapters,
            device_type=device_type,
            dtype=dtype,
        )
    else:
        raise RuntimeError(f'Unsupported distributed_executor_backend: {distributed_executor_backend}.')
