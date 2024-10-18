# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext


class GraphRunner:
    """graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig,
                 device: torch.device, **kwargs):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.device = device
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config

    def __call__(self, **kwargs):
        """call graph runner forward."""
        return self.model(**kwargs)

    def get_model(self):
        """get model."""
        return self.model

    def get_logits(self, hidden_states: torch.Tensor):
        """get logits of model output."""
        if not hasattr(self.model, 'get_logits'):
            return hidden_states
        return self.model.get_logits(hidden_states)

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """prepare inputs."""
        return self.model.prepare_inputs_for_generation(
            past_key_values,
            inputs_embeds,
            context,
        )
