# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
    from lmdeploy.pytorch.messages import SamplingParam, SchedulerSequence, SchedulerSession
    from lmdeploy.pytorch.model_inputs import ModelInputs
    SeqList = List[SchedulerSequence]


class SequenceStrategy(ABC):

    @abstractmethod
    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequence':
        """Make sequence."""
        pass

    @abstractmethod
    def update_running(self, running: 'SeqList', batched_outputs: 'BatchedOutputs',
                       model_inputs: 'ModelInputs') -> None:
        """Update running sequences."""
        pass
