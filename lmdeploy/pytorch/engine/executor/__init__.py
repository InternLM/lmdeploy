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
                   adapters: Dict[str, str] = None) -> ExecutorBase:
    """build model agent executor."""
    if dp * tp == 1:
        from .uni_executor import UniExecutor
        return UniExecutor(model_path=model_path,
                           model_config=model_config,
                           cache_config=cache_config,
                           backend_config=backend_config,
                           tokenizer=tokenizer,
                           adapters=adapters)
    else:
        raise RuntimeError('Failed to build executor.')
