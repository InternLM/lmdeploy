# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import MiscConfig, ModelConfig, SpecDecodeConfig


def build_strategy_factory(model_config: ModelConfig,
                           misc_config: MiscConfig,
                           specdecode_config: SpecDecodeConfig = None):
    """Build strategy factory."""
    model_paradigm = model_config.model_paradigm

    if model_paradigm == 'ar':
        from .ar import ARStrategyFactory
        return ARStrategyFactory(model_config=model_config)
    elif model_paradigm == 'dllm':
        from .dllm import DLLMStrategyFactory
        return DLLMStrategyFactory(model_config=model_config, dllm_config=misc_config.dllm_config)
    elif model_paradigm == 'ar_spec':
        from .ar_spec import ARSpecStrategyFactory
        assert specdecode_config is not None, 'specdecode_config must be provided for ar_spec model'
        return ARSpecStrategyFactory(model_config=model_config, specdecode_config=specdecode_config)
    else:
        raise RuntimeError(f'Unsupported model paradigm: {model_paradigm}')
