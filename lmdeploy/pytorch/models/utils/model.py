# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Iterable, List, Optional, Tuple

import torch

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor
from lmdeploy.pytorch.model_inputs import StepContext


class DeployModelMixin:

    def forward(self, *args, **kwargs):
        """Forward of model."""
        raise NotImplementedError('Not Implemented')

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        raise NotImplementedError('Not Implemented')

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        raise NotImplementedError('Not Implemented')

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return hidden_states

    def rename_weight(self, name: str) -> str:
        """Rename weight."""
        return name

    def update_weights(self):
        """Update weights."""
        pass

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """Update model meta."""
        return None

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return None


def vlm_model(vlm_cls):
    if not issubclass(vlm_cls, torch.nn.Module):
        raise ValueError('Only subclasses of nn.Module can be decorated with @vlm_model.')

    @functools.wraps(vlm_cls)
    def wrapper(*args, **kwargs):
        from lmdeploy.pytorch.models.patch import get_build_model_context
        bm_ctx = get_build_model_context()
        disable_vision_encoder = bm_ctx.disable_vision_encoder
        if disable_vision_encoder:
            mod = torch.nn.Identity()
            mod._is_dummy_mod = True
            return mod
        else:
            return vlm_cls(*args, **kwargs)

    return wrapper
