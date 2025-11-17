# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

    from .cudagraph import CudagraphStrategy
    from .engine import EngineStrategy
    from .model_agent import ModelAgentStrategy
    from .model_inputs import ModelInputsStrategy
    from .sampling import SamplingStrategy
    from .sequence import SequenceStrategy


class StrategyFactoryBase(ABC):

    @abstractmethod
    def build_cudagraph_strategy(self) -> 'CudagraphStrategy':
        """Build cudagraph strategy."""
        pass

    @abstractmethod
    def build_sampling_strategy(self) -> 'SamplingStrategy':
        """Build sampling strategy."""
        pass

    @abstractmethod
    def build_model_inputs_strategy(self) -> 'ModelInputsStrategy':
        """Build model inputs strategy."""
        pass

    @abstractmethod
    def build_model_agent_strategy(self) -> 'ModelAgentStrategy':
        """Build model agent strategy."""
        pass

    @abstractmethod
    def build_engine_strategy(self, cache_config: 'CacheConfig',
                              scheduler_config: 'SchedulerConfig') -> 'EngineStrategy':
        """Build engine strategy."""
        pass

    @abstractmethod
    def build_sequence_strategy(self) -> 'SequenceStrategy':
        """Build sequence strategy."""
        pass
