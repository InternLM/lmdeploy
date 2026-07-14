# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .model_agent import ExtraInputs, ExtraOutputs, StoppingCriteria

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.logits_process import SamplingInputsDelta
    from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta


@dataclass
class StepInputs(ABC):
    """Holds decoding-loop state between steps.

    Lifecycle in _async_step:

      Prefill path:
        1. reindex(delta)              — shrink batch (finished seqs removed)
        2. model forward
        3. merge_prefill(output)       — add prefill result into decode state

      Decode path:
        1. reindex(delta)              — shrink batch
        2. model forward
        3. step_decode(output)         — advance to next decode step
    """
    model_inputs: 'ModelInputs' = None
    extra_inputs: ExtraInputs = None
    stopping_criteria: StoppingCriteria = None
    sampling_delta: 'SamplingInputsDelta' = None

    @abstractmethod
    def merge_prefill(
        self,
        inputs: 'ModelInputs',
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: 'SamplingInputsDelta',
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ExtraOutputs,
    ):
        """Add prefill result into accumulated decode state."""

    @abstractmethod
    def reindex(self, delta: 'ModelInputsDelta'):
        """Shrink batch — keep only sequences at delta.indices."""

    @abstractmethod
    def step_decode(
        self,
        model_inputs: 'ModelInputs',
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: 'SamplingInputsDelta',
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ExtraOutputs,
    ):
        """Advance decode state for next step."""
