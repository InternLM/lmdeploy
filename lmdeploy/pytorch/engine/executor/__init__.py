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
                   adapters: Dict[str, str] = None,
                   device_type: str = 'cuda') -> ExecutorBase:
    """build model agent executor."""
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
    elif dp * tp > 1:
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
        raise RuntimeError('Failed to build executor.')
