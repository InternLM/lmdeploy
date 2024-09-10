# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from lmdeploy.pytorch.model_inputs import StepContext


class GraphRunner:
    """graph runner."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        self.model = model

    def __call__(self, **kwargs):
        """call graph runner forward."""
        return self.model(**kwargs)

    def get_model(self):
        """get model."""
        return self.model

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
