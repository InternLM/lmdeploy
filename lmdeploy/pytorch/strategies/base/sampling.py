# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence

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
