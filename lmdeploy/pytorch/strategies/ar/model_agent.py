# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch.profiler import record_function

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.model_inputs import ModelInputs

from ..base.model_agent import ExtraInputs, ExtraOutputs, ModelAgentStrategy, StoppingCriteria

SeqList = List[SchedulerSequence]


class ARExtraInputs(ExtraInputs):
    """Ar extra inputs."""


class ARExtraOutputs(ExtraOutputs):
    """Ar extra outputs."""


@dataclass
class ARStoppingCriteria(StoppingCriteria):
    num_appendable_ids: torch.Tensor

    @record_function('stopping_criteria')
    def step(self,
             token_ids: torch.Tensor,
             stop_words: torch.Tensor,
             inputs: Optional[ModelInputs] = None,
             extra_inputs: Optional[ARExtraInputs] = None):
        """Check whether to stop generation."""
        num_appendable_ids = self.num_appendable_ids - 1
        stopped = num_appendable_ids <= 0
        stop_pos = torch.zeros_like(num_appendable_ids)
        if stop_words is not None:
            sw_stopped = (token_ids[:, None] == stop_words).any(1)
            stopped = stopped | sw_stopped
            one_ids = torch.clamp_max(num_appendable_ids, 0)
            num_appendable_ids = torch.where(sw_stopped, one_ids, num_appendable_ids)

        # I don't know why assign inplace does not works...
        new_stopping = ARStoppingCriteria(num_appendable_ids=num_appendable_ids)
        return stopped, stop_pos, new_stopping


class ARModelAgentStrategy(ModelAgentStrategy):

    def slice_outputs(self, inputs: torch.Tensor, seq_length: torch.LongTensor) -> torch.Tensor:
        """Slice outputs."""
        # batch size == 1
        if len(seq_length) == 1:
            return inputs[-1:]

        if len(seq_length) == inputs.size(0):
            return inputs
        last_idx = seq_length.cumsum(-1) - 1
        return inputs[last_idx]

    def slice_extra_inputs(self, extra_inputs: ARExtraInputs, seq_length: torch.LongTensor) -> ARExtraInputs:
        """Slice outputs."""
        return extra_inputs

    def _step_sampling_inputs(self, sampling_inputs: SamplingInputs, next_token_ids: torch.Tensor):
        """step."""
        sampling_inputs.num_ignore_eos = sampling_inputs.num_ignore_eos - 1
        if sampling_inputs.random_offsets is not None:
            # random offset is used to generate random numbers for multinomial sampling
            # so we need to increase it by 1 at each step
            sampling_inputs.random_offsets += 1

        all_ids = sampling_inputs.all_ids
        if all_ids is not None:
            sampling_inputs.all_ids = torch.cat([all_ids, next_token_ids[:, None]], 1)

        return sampling_inputs

    def make_stopping_criteria(self, seqs: SeqList) -> ARStoppingCriteria:
        """Create stopping criteria."""
        num_appendable = [seq.sampling_param.max_new_tokens - seq.num_new_tokens for seq in seqs]
        num_appendable = torch.tensor(num_appendable)
        return ARStoppingCriteria(num_appendable_ids=num_appendable)

    def make_extra_inputs(self, seqs: 'SeqList') -> ExtraInputs:
        """Create extra inputs."""
        return ARExtraInputs()

    def make_extra_outputs(self, extra_inputs: ARExtraInputs) -> ARExtraOutputs:
        """Create extra outputs."""
        return ARExtraOutputs()

    def update_inputs_for_next_step(self, model_inputs: 'ModelInputs', sampling_inputs: 'SamplingInputs',
                                    next_token_ids: torch.Tensor, model_metas: Any, extra_inputs: ARExtraInputs,
                                    **kwargs):
        """Step next inputs."""
        model_inputs.model_metas = model_metas
        step_seqlens = model_inputs.seq_length
        model_inputs.step(next_token_ids, step_seqlens)
        self._step_sampling_inputs(sampling_inputs, next_token_ids)
        return model_inputs, extra_inputs

    def post_sampling(self, inputs: 'ModelInputs', logits: torch.Tensor, next_token_ids: torch.LongTensor,
                      extra_inputs: ARExtraInputs):
        """Post sampling."""
        return next_token_ids, extra_inputs

    @contextmanager
    def broadcast_next_token(self, next_token_ids: torch.Tensor, extra_inputs: ExtraInputs, dist_ctx: DistContext):
        """Broadcast next token ids and extra inputs."""
        tp_gpu_group = dist_ctx.tp_gpu_group
        handle = dist.broadcast(next_token_ids, src=0, group=tp_gpu_group, async_op=True)
        yield
        handle.wait()
