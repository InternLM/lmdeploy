# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

import torch

from lmdeploy.pytorch.engine.logits_process import SamplingInputs, SamplingInputsDelta
from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.model_inputs import ModelInputsDelta

from .model_agent import ExtraInputs

SeqList = List[SchedulerSequence]


class SamplingStrategy(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def make_sampling_inputs(self, seqs: SeqList) -> SamplingInputs:
        """Create sampling inputs from the sequences."""
        pass

    @abstractmethod
    def on_session_end(self, session_id: int) -> None:
        """Invoked on session ends."""
        pass

    @abstractmethod
    def merge_sampling_delta(
        self,
        sampling_delta: 'SamplingInputsDelta',
        other: 'SamplingInputsDelta',
    ) -> 'SamplingInputsDelta':
        """Merge two sampling deltas."""

    @abstractmethod
    def step_sampling_delta(
        self,
        sampling_delta: 'SamplingInputsDelta',
        next_token_ids: torch.Tensor,
        extra_inputs: 'ExtraInputs',
    ) -> 'SamplingInputsDelta':
        """Step next delta."""
        pass

    @abstractmethod
    def update_sampling_delta(
        self,
        sampling_delta: 'SamplingInputsDelta',
        delta: 'ModelInputsDelta',
    ) -> 'SamplingInputsDelta':
        """Update sampling delta with model inputs delta."""
        pass
