# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig

from .base import ExecutorBase


def build_executor(model_path: str,
                   model_config: ModelConfig,
                   cache_config: CacheConfig,
                   backend_config: BackendConfig,
                   tokenizer: Any,
                   dp: int = 1,
                   tp: int = 1,
                   nproc_per_node: int = None,
                   adapters: Dict[str, str] = None,
                   device_type: str = 'cuda') -> ExecutorBase:
    """build model agent executor."""

    world_size = dp * tp
    if nproc_per_node is None:
        nproc_per_node = world_size

    nnodes = world_size // nproc_per_node

    if dp * tp == 1:
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
    elif nnodes == 1:
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
    else:
        from .ray_executor import RayExecutor
        return RayExecutor(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            tokenizer=tokenizer,
            dp=dp,
            tp=tp,
            nproc_per_node=nproc_per_node,
            adapters=adapters,
            device_type=device_type,
        )
