# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.strategies.base.sequence import SequenceStrategy

if TYPE_CHECKING:
    from lmdeploy.pytorch.strategies.base.cudagraph import CudagraphStrategy
    from lmdeploy.pytorch.strategies.base.model_inputs import ModelInputsStrategy
    from lmdeploy.pytorch.strategies.base.sampling import SamplingStrategy
    from lmdeploy.pytorch.strategies.base.model_agent import ModelAgentStrategy
    from lmdeploy.pytorch.strategies.base.engine import EngineStrategy
    from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

from ..base import StrategyFactoryBase


class ARStrategyFactory(StrategyFactoryBase):

    def __init__(self, model_config: ModelConfig):
        """config."""
        self.model_config = model_config

    def build_cudagraph_strategy(self) -> 'CudagraphStrategy':
        """Build cudagraph strategy."""
        from .cudagraph import ARCudagraphStrategy
        return ARCudagraphStrategy()

    def build_sampling_strategy(self) -> 'SamplingStrategy':
        """Build sampling strategy."""
        from .sampling import ARSamplingStrategy
        pad_token_id = self.model_config.bos_token_id
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        return ARSamplingStrategy(pad_token_id)

    def build_model_inputs_strategy(self) -> 'ModelInputsStrategy':
        """Build model inputs strategy."""
        from .model_inputs import ARModelInputsStrategy
        return ARModelInputsStrategy()

    def build_model_agent_strategy(self) -> 'ModelAgentStrategy':
        """Build model agent strategy."""
        from .model_agent import ARModelAgentStrategy
        return ARModelAgentStrategy()

    def build_engine_strategy(self, cache_config: 'CacheConfig',
                              scheduler_config: 'SchedulerConfig') -> 'EngineStrategy':
        """Build engine strategy."""
        from .engine import AREngineStrategy
        return AREngineStrategy(cache_config=cache_config, scheduler_config=scheduler_config)

    def build_sequence_strategy(self) -> SequenceStrategy:
        from .sequence import ARSequenceStrategy
        return ARSequenceStrategy()
