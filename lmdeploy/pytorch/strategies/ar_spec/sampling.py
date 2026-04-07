# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.messages import SchedulerSequence

from ..ar.sampling import ARSamplingStrategy

SeqList = list[SchedulerSequence]


class ARSpecSamplingStrategy(ARSamplingStrategy):
    """Sampling strategy for AR with spec models."""

    def __init__(self, pad_token_id: int, num_spec_tokens: int) -> None:
        super().__init__(pad_token_id)
        self.num_spec_tokens = num_spec_tokens
