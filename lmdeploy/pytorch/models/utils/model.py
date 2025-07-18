# Copyright (c) OpenMMLab. All rights reserved.
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


def _patch_vlm_init(vlm_cls):
    """Patch the class to add DeployModelMixin."""
    origin_init = vlm_cls.__init__

    def _new_init(self, *args, **kwargs):
        """New init method."""
        from lmdeploy.pytorch.models.patch import get_build_model_context
        bm_ctx = get_build_model_context()
        disable_vision_encoder = bm_ctx.disable_vision_encoder

        if disable_vision_encoder:
            # assume vls_cls is subclass of nn.Module
            super(vlm_cls, self).__init__()
            self._is_dummy_mod = True
            return

        origin_init(self, *args, **kwargs)

    vlm_cls.__init__ = _new_init
    return vlm_cls


def vlm_model(vlm_cls):
    """Decorator to mark a class as a VLM model."""
    vlm_cls = _patch_vlm_init(vlm_cls)
    return vlm_cls
