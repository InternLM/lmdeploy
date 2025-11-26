# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.pytorch.config import ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.strategies.base.sequence import SequenceStrategy

if TYPE_CHECKING:
    from lmdeploy.pytorch.strategies.base.cudagraph import CudagraphStrategy
    from lmdeploy.pytorch.strategies.base.model_inputs import ModelInputsStrategy
    from lmdeploy.pytorch.strategies.base.sampling import SamplingStrategy
    from lmdeploy.pytorch.strategies.base.model_agent import ModelAgentStrategy
    from lmdeploy.pytorch.strategies.base.engine import EngineStrategy
    from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

from ..base import StrategyFactoryBase


class ARSpecStrategyFactory(StrategyFactoryBase):

    def __init__(self, model_config: ModelConfig, specdecode_config: SpecDecodeConfig):
        """config."""
        self.model_config = model_config
        self.specdecode_config = specdecode_config
        self.pad_token_id = model_config.bos_token_id or 0

    def build_cudagraph_strategy(self) -> 'CudagraphStrategy':
        """Build cudagraph strategy."""
        from .cudagraph import ARSpecCudagraphStrategy
        return ARSpecCudagraphStrategy(self.specdecode_config.num_speculative_tokens)

    def build_sampling_strategy(self) -> 'SamplingStrategy':
        """Build sampling strategy."""
        from .sampling import ARSpecSamplingStrategy
        pad_token_id = self.model_config.bos_token_id
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        return ARSpecSamplingStrategy(pad_token_id)

    def build_model_inputs_strategy(self) -> 'ModelInputsStrategy':
        """Build model inputs strategy."""
        from .model_inputs import ARSpecModelInputsStrategy
        return ARSpecModelInputsStrategy(self.specdecode_config.num_speculative_tokens)

    def build_model_agent_strategy(self) -> 'ModelAgentStrategy':
        """Build model agent strategy."""
        from .model_agent import ARSpecModelAgentStrategy
        return ARSpecModelAgentStrategy(self.specdecode_config.num_speculative_tokens)

    def build_engine_strategy(self, cache_config: 'CacheConfig',
                              scheduler_config: 'SchedulerConfig') -> 'EngineStrategy':
        """Build engine strategy."""
        from .engine import ARSpecEngineStrategy
        return ARSpecEngineStrategy(cache_config=cache_config,
                                    scheduler_config=scheduler_config,
                                    num_spec_tokens=self.specdecode_config.num_speculative_tokens)

    def build_sequence_strategy(self) -> SequenceStrategy:
        from .sequence import ARSpecSequenceStrategy
        return ARSpecSequenceStrategy()
