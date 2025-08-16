# Copyright (c) OpenMMLab. All rights reserved.
import functools
from dataclasses import dataclass
from typing import List

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext


@dataclass
class GraphRunnerMeta:
    padding_batch_size: int = None


@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size."""
    ret = []
    batch_size = 1
    while batch_size < max_batches:
        ret.append(batch_size)
        batch_size *= 2
    ret.append(max_batches)
    return ret


class GraphRunner:
    """Graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device, **kwargs):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.device = device
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self._runner_meta = GraphRunnerMeta()

    def __call__(self, **kwargs):
        """Call graph runner forward."""
        return self.model(**kwargs)

    def get_model(self):
        """Get model."""
        return self.model

    def get_logits(self, hidden_states: torch.Tensor):
        """Get logits of model output."""
        if not hasattr(self.model, 'get_logits'):
            return hidden_states
        return self.model.get_logits(hidden_states)

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""
        return self.model.prepare_inputs_for_generation(
            past_key_values,
            inputs_embeds,
            context,
        )

    def update_model_metas(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""
        if hasattr(self.model, 'update_model_metas'):
            return self.model.update_model_metas(
                past_key_values,
                inputs_embeds,
                context,
            )

        return None

    def get_input_processor(self):
        """Get input processor."""
        if hasattr(self.model, 'get_input_processor'):
            return self.model.get_input_processor()
        else:
            return None

    def reset(self):
        """Remove all graphs to prevent hanging on exit."""
        pass

    def get_meta(self):
        """Get graphrunner meta."""
        return self._runner_meta

    def update_inputs(self, inputs):
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        return _get_capture_batch_size_impl(self.cache_config.max_batches)
