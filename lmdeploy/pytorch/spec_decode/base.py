# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict

import torch

from ..config import CacheConfig, ModelConfig
from ..engine.logits_process import SamplingInputs
from ..model_inputs import ModelInputs
from ..strategies.base.model_agent import ExtraInputs


class BaseSpecModelAgent:
    """Speculative model agent."""

    def __init__(self, enable: bool = False):
        self._enabled = enable

    def is_enabled(self):
        return self._enabled

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        pass

    def set_model_config(self, model_config: ModelConfig):
        """Set model config."""
        pass

    def build_model(self, empty_init: bool, target_model=None, model_format=None, build_model_ctx=None):
        """Build draft model."""
        pass

    def build_graph_runner(self):
        """Build graph runner."""
        pass

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        """Build cache engine."""
        pass

    async def async_model_forward(self, next_token_ids: torch.Tensor, model_inputs: ModelInputs,
                                  extra_inputs: ExtraInputs, sampling_inputs: SamplingInputs):
        """Draft model forward."""
        return extra_inputs

    def warmup(self, max_batches: int, target_model_config: ModelConfig):
        """warmup."""
        pass

    def reset_graph_runner(self):
        'reset graph runner'
        pass

    def update_main_model_outputs(self, output: Dict[str, torch.Tensor], model_inputs: ModelInputs):
        """Update outputs of main model."""
        if not self.is_enabled():
            hidden_states = output.pop('hidden_states')
            return hidden_states, output

        hidden_states = output['hidden_states']
        if not model_inputs.is_decoding:
            logits_indices = model_inputs.seq_length.cumsum(0) - 1
            hidden_states = hidden_states[:, logits_indices]
        if 'aux_hidden_states' in output:
            # replace with aux
            output['hidden_states'] = output.pop('aux_hidden_states')
        return hidden_states, output
