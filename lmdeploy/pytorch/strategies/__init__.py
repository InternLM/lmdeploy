# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import MiscConfig, ModelConfig


def build_strategy_factory(model_config: ModelConfig, misc_config: MiscConfig):
    """Build strategy factory."""
    model_paradigm = model_config.model_paradigm

    if model_paradigm == 'ar':
        from .ar import ARStrategyFactory
        return ARStrategyFactory(model_config=model_config)
    elif model_paradigm == 'dllm':
        from .dllm import DLLMStrategyFactory
        return DLLMStrategyFactory(model_config=model_config, dllm_config=misc_config.dllm_config)
    else:
        raise RuntimeError(f'Unsupported model paradigm: {model_paradigm}')
