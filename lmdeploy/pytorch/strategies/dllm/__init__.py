# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.pytorch.config import DLLMConfig, ModelConfig
from lmdeploy.pytorch.strategies.base.sequence import SequenceStrategy
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.pytorch.strategies.base.cudagraph import CudagraphStrategy
    from lmdeploy.pytorch.strategies.base.model_inputs import ModelInputsStrategy
    from lmdeploy.pytorch.strategies.base.sampling import SamplingStrategy
    from lmdeploy.pytorch.strategies.base.model_agent import ModelAgentStrategy
    from lmdeploy.pytorch.strategies.base.engine import EngineStrategy
    from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

from ..base import StrategyFactoryBase

logger = get_logger('lmdeploy')


class DLLMStrategyFactory(StrategyFactoryBase):

    def __init__(self, model_config: ModelConfig, dllm_config: DLLMConfig):
        """config."""
        self.model_config = model_config
        self.dllm_config = dllm_config

        # update dllm_block_length
        self.dllm_block_length = self._update_dllm_block_length()

    def _update_dllm_block_length(self):
        """Update dllm_block_length."""
        if self.dllm_config.block_length is None:
            dllm_block_length = self.model_config.dllm_block_length
            if dllm_block_length is None:
                dllm_block_length = 4
                logger.warning('Model does not provide dllm_block_length. '
                               f'Set dllm_block_length={dllm_block_length} as default.')
        else:
            dllm_block_length = self.dllm_config.block_length

        assert dllm_block_length is not None, 'dllm_block_length should be set in model_config or dllm_config'

        self.dllm_config.block_length = dllm_block_length
        self.model_config.dllm_block_length = dllm_block_length

        if self.dllm_config.denoising_steps is None:
            self.dllm_config.denoising_steps = dllm_block_length
        return dllm_block_length

    def build_cudagraph_strategy(self) -> 'CudagraphStrategy':
        """Build cudagraph strategy."""
        from .cudagraph import DLLMCudagraphStrategy
        return DLLMCudagraphStrategy(block_size=self.dllm_block_length)

    def build_sampling_strategy(self) -> 'SamplingStrategy':
        """Build sampling strategy."""
        from .sampling import DLLMSamplingStrategy
        pad_token_id = self.model_config.bos_token_id
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        return DLLMSamplingStrategy(pad_token_id, self.dllm_block_length)

    def build_model_inputs_strategy(self) -> 'ModelInputsStrategy':
        """Build model inputs strategy."""
        from .model_inputs import DLLMModelInputsStrategy
        return DLLMModelInputsStrategy(block_size=self.dllm_block_length)

    def build_model_agent_strategy(self) -> 'ModelAgentStrategy':
        """Build model agent strategy."""
        from .model_agent import DLLMModelAgentStrategy
        return DLLMModelAgentStrategy(dllm_config=self.dllm_config, dllm_mask_token=self.model_config.dllm_mask_token)

    def build_engine_strategy(self, cache_config: 'CacheConfig',
                              scheduler_config: 'SchedulerConfig') -> 'EngineStrategy':
        """Build engine strategy."""
        from .engine import DLLMEngineStrategy
        return DLLMEngineStrategy(cache_config=cache_config,
                                  scheduler_config=scheduler_config,
                                  dllm_block_length=self.dllm_block_length)

    def build_sequence_strategy(self) -> SequenceStrategy:
        from .sequence import DLLMSequenceStrategy
        return DLLMSequenceStrategy(block_size=self.dllm_block_length,
                                    dllm_mask_token=self.model_config.dllm_mask_token)
