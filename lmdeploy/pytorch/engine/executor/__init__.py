# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from .base import ExecutorBase


def get_distributed_executor_backend(world_size: int, device_type: str):
    """get distributed executor backend."""
    from lmdeploy.pytorch.backends import get_backend
    if world_size == 1:
        return None

    backend = get_backend(device_type)
    if not backend.support_ray():
        return 'mp'
    device_count = backend.device_count()
    if device_count < world_size:
        return 'ray'
    else:
        return 'mp'


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
        distributed_executor_backend = get_distributed_executor_backend(world_size, device_type)
        if distributed_executor_backend is not None:
            logger.info(f'Distributed Executor backend: {distributed_executor_backend}')

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
