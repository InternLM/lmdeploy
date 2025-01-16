# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable, List, Optional, Tuple

import torch

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor
from lmdeploy.pytorch.model_inputs import StepContext


class DeployModelMixin:

    def forward(self, *args, **kwargs):
        """forward of model."""
        raise NotImplementedError('Not Implemented')

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        raise NotImplementedError('Not Implemented')

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        raise NotImplementedError('Not Implemented')

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return hidden_states

    def update_weights(self):
        """update weights."""
        pass

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """update model meta."""
        return None

    def get_input_processor(self) -> BaseModelInputProcessor:
        """get input processor."""
        return None
