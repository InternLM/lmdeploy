# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.model_inputs import ModelInputs
    SeqList = List[SchedulerSequence]


def to_device(self, device: str, non_blocking: bool = False):
    """To device."""
    out_dict = dict()
    for f in fields(self):
        k = f.name
        v = getattr(self, k)
        if isinstance(v, torch.Tensor):
            v = v.to(device, non_blocking=non_blocking)
        out_dict[k] = v

    return type(self)(**out_dict)


@dataclass
class ExtraInputs(ABC):

    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        return to_device(self, device, non_blocking)

    def broadcast(self, src: int, group, async_op=False):
        """Broadcast extra inputs."""
        pass


@dataclass
class ExtraOutputs(ABC):

    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        return to_device(self, device, non_blocking)

    def to_cpu(self):
        """To cpu."""
        return self.to_device('cpu', non_blocking=False)

    def to_numpy(self):
        """To numpy."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor) and v.dtype != torch.bfloat16:
                v = v.detach().numpy()
            elif hasattr(v, 'to_numpy'):
                v = v.to_numpy()
            out[k] = v
        return type(self)(**out)

    def to_tensor(self):
        """To tensor."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif hasattr(v, 'to_tensor'):
                v = v.to_tensor()
            out[k] = v
        return type(self)(**out)


@dataclass
class StoppingCriteria(ABC):
    """Base class for stopping criteria."""

    @abstractmethod
    def step(self,
             token_ids: torch.Tensor,
             stop_words: torch.Tensor,
             inputs: Optional['ModelInputs'] = None,
             extra_inputs: Optional[ExtraInputs] = None):
        """Check whether to stop generation."""
        pass

    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        return to_device(self, device, non_blocking)


class ModelAgentStrategy(ABC):
    """Base class for model agent strategies."""

    @abstractmethod
    def slice_outputs(self, inputs: torch.Tensor, seq_length: torch.LongTensor) -> torch.Tensor:
        """Slice outputs."""
        pass

    @abstractmethod
    def slice_extra_inputs(self, extra_inputs: ExtraInputs, seq_length: torch.LongTensor) -> ExtraInputs:
        """Slice outputs."""
        pass

    @abstractmethod
    def make_stopping_criteria(self, seqs: 'SeqList') -> StoppingCriteria:
        """Create stopping criteria."""
        pass

    @abstractmethod
    def make_extra_inputs(self, seqs: 'SeqList') -> ExtraInputs:
        """Create extra inputs."""
        pass

    @abstractmethod
    def make_extra_outputs(self, extra_inputs: ExtraInputs) -> ExtraOutputs:
        """Create extra outputs."""
        pass

    @abstractmethod
    def update_inputs_for_next_step(self, model_inputs: 'ModelInputs', sampling_inputs: 'SamplingInputs',
                                    next_token_ids: torch.Tensor, model_metas: Any, extra_inputs: ExtraInputs,
                                    **kwargs):
        """Step next inputs."""
        pass

    @abstractmethod
    def post_sampling(self, inputs: 'ModelInputs', logits: torch.Tensor, next_token_ids: torch.LongTensor,
                      extra_inputs: ExtraInputs):
        """Post sampling."""
        pass

    def make_dummy_next_token(self, inputs: 'ModelInputs', logits: torch.Tensor, extra_inputs: ExtraInputs):
        """Make dummy next token for broadcast."""
        with torch.inference_mode():
            next_token_ids = inputs.input_ids.new_zeros(logits.size(0))
        return next_token_ids, extra_inputs

    @abstractmethod
    @contextmanager
    def broadcast_next_token(self, next_token_ids: torch.Tensor, extra_inputs: ExtraInputs, dist_ctx: 'DistContext'):
        """Broadcast next token ids and extra inputs."""
