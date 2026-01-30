# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig


def build_runner(model: nn.Module, model_config: ModelConfig, cache_config: CacheConfig, backend_config: BackendConfig,
                 device: torch.device):

    use_compile = False
    if envs.force_torch_compile:
        use_compile = True
    elif hasattr(model, 'use_torch_compile'):
        use_compile = model.use_torch_compile()  # type: ignore[attr-defined]

    if use_compile:
        from .compile_runner import TorchCompileRunner  # noqa: F401
        return TorchCompileRunner(model,
                                  model_config=model_config,
                                  cache_config=cache_config,
                                  backend_config=backend_config,
                                  device=device)
    else:
        from .cudagraph_runner import CUDAGraphRunner  # noqa: F401
        return CUDAGraphRunner(model,
                               model_config=model_config,
                               cache_config=cache_config,
                               backend_config=backend_config,
                               device=device)
