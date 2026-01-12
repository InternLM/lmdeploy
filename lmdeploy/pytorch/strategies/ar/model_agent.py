# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.profiler import record_function

from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

from ..base.model_agent import ExtraInputs, ExtraOutputs, ModelAgentStrategy, StoppingCriteria

SeqList = List[SchedulerSequence]


def get_model_inputs_next_decoding(inputs: ModelInputs, input_ids: torch.Tensor, max_q_seqlen: int,
                                   model_metas) -> ModelInputs:
    """Next decoding step."""
    if input_ids.dim() == 1:
        input_ids = input_ids[None, :]
    return ModelInputs(
        input_ids=input_ids,
        seq_length=torch.full_like(inputs.seq_length, max_q_seqlen),
        history_lengths=inputs.history_lengths + inputs.seq_length,
        block_offsets=inputs.block_offsets,
        is_decoding=True,
        num_ignored_history=inputs.num_ignored_history,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=inputs.max_kv_seqlen + max_q_seqlen,
        sum_kv_seqlen=inputs.sum_kv_seqlen + inputs.seq_length.numel() * inputs.max_q_seqlen,
        local_adapter_ids=inputs.local_adapter_ids,
        model_metas=model_metas,
        state_offsets=inputs.state_offsets,
    )


@dataclass
class ARExtraInputs(ExtraInputs):
    """Ar extra inputs."""


@dataclass
class ARExtraOutputs(ExtraOutputs):
    """Ar extra outputs."""


@dataclass
class ARStoppingCriteria(StoppingCriteria):
    num_appendable_ids: torch.Tensor

    def clone(self):
        """clone."""
        return ARStoppingCriteria(num_appendable_ids=self.num_appendable_ids)

    def merge(self, other: 'ARStoppingCriteria'):
        """Merge two stopping criteria."""
        new_num_appendable = torch.cat([self.num_appendable_ids, other.num_appendable_ids], dim=0)
        return ARStoppingCriteria(num_appendable_ids=new_num_appendable)

    def update(self, delta: ModelInputsDelta):
        """Update stopping criteria."""
        indices = delta.indices
        new_num_appendable = self.num_appendable_ids[indices]
        return ARStoppingCriteria(num_appendable_ids=new_num_appendable)

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

    def slice_extra_inputs(self, extra_inputs: ARExtraInputs, model_inputs: ModelInputs,
                           model_outputs: Dict[str, torch.Tensor], **kwargs) -> ARExtraInputs:
        """Slice outputs."""
        return extra_inputs

    @record_function('step_sampling_inputs')
    def step_sampling_inputs(self, sampling_inputs: SamplingInputs, next_token_ids: torch.Tensor, **kwargs):
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

    def make_extra_inputs(self, seqs: 'SeqList', model_inputs: 'ModelInputs') -> ExtraInputs:
        """Create extra inputs."""
        return ARExtraInputs()

    def make_extra_outputs(self, extra_inputs: ARExtraInputs) -> ARExtraOutputs:
        """Create extra outputs."""
        return ARExtraOutputs()

    def update_prefill_for_next_step(
        self,
        model_inputs: 'ModelInputs',
        extra_inputs: ARExtraInputs,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ARExtraOutputs,
    ) -> Tuple['ModelInputs', ARExtraInputs]:
        """Step next decoding."""
        inputs = get_model_inputs_next_decoding(model_inputs, next_token_ids, max_q_seqlen=1, model_metas=model_metas)
        return inputs, extra_inputs

    def update_decoding_for_next_step(self, model_inputs: 'ModelInputs', next_token_ids: torch.Tensor, model_metas: Any,
                                      extra_inputs: ARExtraInputs, **kwargs):
        """Step next inputs."""
        model_inputs.model_metas = model_metas
        step_seqlens = model_inputs.seq_length
        model_inputs.step(next_token_ids, step_seqlens)
        return model_inputs, extra_inputs

    def post_sampling(self, inputs: 'ModelInputs', logits: torch.Tensor, next_token_ids: torch.LongTensor,
                      extra_inputs: ARExtraInputs):
        """Post sampling."""
        return next_token_ids, extra_inputs

    @contextmanager
    def broadcast_next_token(self, next_token_ids: torch.Tensor, extra_inputs: ExtraInputs, dist_ctx: DistContext):
        """Broadcast next token ids and extra inputs."""
        tp_gpu_group = dist_ctx.attn_tp_group.gpu_group
        rank = dist.get_global_rank(tp_gpu_group, 0)
        handle = dist.broadcast(next_token_ids, src=rank, group=tp_gpu_group, async_op=True)
        yield
        handle.wait()
