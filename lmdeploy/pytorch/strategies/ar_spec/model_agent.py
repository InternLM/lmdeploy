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

from ..ar.model_agent import ARStoppingCriteria, get_model_inputs_next_decoding
from ..base.model_agent import ExtraInputs, ExtraOutputs, ModelAgentStrategy

SeqList = List[SchedulerSequence]


@dataclass
class ARSpecExtraInputs(ExtraInputs):
    """ARSpec extra inputs."""
    # draft model inputs
    target_logits: torch.Tensor = None
    target_hidden_states: torch.Tensor = None
    target_position_ids: torch.Tensor = None
    next_token_ids: torch.LongTensor = None
    last_token_indices: torch.LongTensor = None

    # draft model outputs
    output_draft_token_ids: torch.Tensor = None
    num_rejected_tokens: torch.Tensor = None
    output_token_ids: torch.Tensor = None

    def __repr__(self):
        return (f'ARSpecExtraInputs(next_token_ids={self.next_token_ids}, '
                f'output_draft_token_ids={self.output_draft_token_ids}, '
                f'last_token_indices={self.last_token_indices}, '
                f'num_rejected_tokens={self.num_rejected_tokens}, '
                f'output_token_ids={self.output_token_ids})')

    def broadcast(self, src: int, group, async_op=False):
        dist.broadcast(self.output_draft_token_ids, src=src, group=group, async_op=async_op)
        handle = dist.broadcast(self.num_rejected_tokens, src=src, group=group, async_op=async_op)
        return handle

    def merge(self, other: 'ARSpecExtraInputs'):
        """Merge extra inputs."""
        output_token_ids = torch.cat([self.output_token_ids, other.output_token_ids], dim=0)
        return ARSpecExtraInputs(output_token_ids=output_token_ids)


@dataclass
class ARSpecExtraOutputs(ExtraOutputs):
    """ARSpec extra outputs."""
    # output the draft tokens to seq only for last loop step
    draft_token_ids: torch.Tensor = None

    def __repr__(self):
        return (f'ARSpecExtraOutputs(draft_token_ids={self.draft_token_ids})')


@dataclass
class ARSpecStoppingCriteria(ARStoppingCriteria):
    num_appendable_ids: torch.Tensor

    def clone(self):
        """clone."""
        return ARSpecStoppingCriteria(num_appendable_ids=self.num_appendable_ids)

    def merge(self, other: 'ARSpecStoppingCriteria'):
        """Merge two stopping criteria."""
        new_num_appendable = torch.cat([self.num_appendable_ids, other.num_appendable_ids], dim=0)
        return ARSpecStoppingCriteria(num_appendable_ids=new_num_appendable)

    def update(self, delta: ModelInputsDelta):
        """Update stopping criteria."""
        indices = delta.indices
        new_num_appendable = self.num_appendable_ids[indices]
        return ARSpecStoppingCriteria(num_appendable_ids=new_num_appendable)

    @record_function('stopping_criteria')
    def step(self,
             next_token_ids: torch.Tensor,
             stop_words: torch.Tensor,
             inputs: Optional[ModelInputs] = None,
             extra_inputs: Optional[ARSpecExtraInputs] = None):
        """Check whether to stop generation."""
        token_ids = extra_inputs.output_token_ids

        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(-1)
        valid_tokens = token_ids > -1
        mask = (self.num_appendable_ids.unsqueeze(-1) - valid_tokens.cumsum(dim=-1)) <= 0
        if stop_words is not None:
            token_ids_rsp = token_ids.unsqueeze(-1).repeat(1, 1, stop_words.numel())
            stop_words_rsp = stop_words.reshape(1, 1, -1)
            assert stop_words_rsp.ndim == token_ids_rsp.ndim == 3
            stop_mask = (token_ids_rsp == stop_words_rsp).any(-1)
            mask = mask ^ stop_mask
        # find the index of first `1`,  if not found, would be 0
        index = torch.argmax(mask.int(), dim=-1, keepdim=True)
        # update index of 0 to -1 if not found
        stop_pos = torch.where(index == 0, mask[:, 0:1].int() - 1, index).ravel()
        stopped = stop_pos != -1
        num_valid_tokens = valid_tokens.sum(dim=-1)
        num_appendable_ids = self.num_appendable_ids - num_valid_tokens
        one_ids = torch.clamp_max(num_appendable_ids, 0)
        num_appendable_ids = torch.where(stopped, one_ids, num_appendable_ids)
        return stopped, stop_pos, ARSpecStoppingCriteria(num_appendable_ids=num_appendable_ids)


class ARSpecModelAgentStrategy(ModelAgentStrategy):

    def __init__(self, num_spec_tokens: int):
        self.num_spec_tokens = num_spec_tokens

    def slice_outputs(self, inputs: torch.Tensor, seq_length: torch.LongTensor) -> torch.Tensor:
        """Slice outputs."""
        # batch size == 1
        if len(seq_length) == 1:
            return inputs[-1:]

        if len(seq_length) == inputs.size(0):
            return inputs
        last_idx = seq_length.cumsum(-1) - 1
        return inputs[last_idx]

    def slice_extra_inputs(self, extra_inputs: ARSpecExtraInputs, model_inputs: ModelInputs,
                           model_outputs: Dict[str, torch.Tensor], **kwargs) -> ARSpecExtraInputs:
        """Slice outputs."""
        extra_inputs = ARSpecExtraInputs()
        extra_inputs.target_hidden_states = model_outputs.get('hidden_states')
        extra_inputs.target_position_ids = model_outputs.get('position_ids', None)
        if model_inputs.is_decoding:
            batch_size = model_inputs.seq_length.size(0)
            logits = model_outputs['logits'][0]
            extra_inputs.target_logits = logits.unflatten(0, (batch_size, -1))[:, :-1]
        return extra_inputs

    def step_sampling_inputs(self, sampling_inputs: SamplingInputs, next_token_ids: torch.Tensor, **kwargs):
        """step."""
        sampling_inputs.num_ignore_eos = sampling_inputs.num_ignore_eos - 1

        all_ids = sampling_inputs.all_ids
        if all_ids is not None:
            sampling_inputs.all_ids = torch.cat([all_ids, next_token_ids[:, None]], 1)

        return sampling_inputs

    def make_stopping_criteria(self, seqs: SeqList) -> ARSpecStoppingCriteria:
        """Create stopping criteria."""
        num_appendable = [seq.sampling_param.max_new_tokens - seq.num_new_tokens for seq in seqs]
        num_appendable = torch.tensor(num_appendable)
        return ARSpecStoppingCriteria(num_appendable_ids=num_appendable)

    def make_extra_inputs(self, seqs: 'SeqList', model_inputs: 'ModelInputs') -> ExtraInputs:
        """Create extra inputs."""
        return ARSpecExtraInputs()

    def update_extra_inputs(self, extra_inputs: ARSpecExtraInputs, delta: 'ModelInputsDelta') -> ARSpecExtraInputs:
        """Update extra inputs with model inputs delta."""
        indices = delta.indices
        output_token_ids = extra_inputs.output_token_ids[indices]
        return ARSpecExtraInputs(output_token_ids=output_token_ids)

    def make_extra_outputs(self, extra_inputs: ARSpecExtraInputs) -> ARSpecExtraOutputs:
        """Create extra outputs."""
        output = ARSpecExtraOutputs()
        output.draft_token_ids = extra_inputs.output_draft_token_ids
        return output

    def update_prefill_for_next_step(
        self,
        model_inputs: 'ModelInputs',
        extra_inputs: ARSpecExtraInputs,
        next_token_ids: torch.Tensor,
        model_metas: Any,
        extra_outputs: ARSpecExtraOutputs,
    ) -> Tuple['ModelInputs', ARSpecExtraInputs]:
        """Step next decoding."""
        next_token_ids = next_token_ids[:, None]
        next_token_ids = torch.cat([next_token_ids, extra_outputs.draft_token_ids], dim=-1)
        max_q_seqlen = next_token_ids.size(-1)
        next_token_ids = next_token_ids.flatten()[None, :]
        inputs = get_model_inputs_next_decoding(model_inputs,
                                                next_token_ids,
                                                max_q_seqlen=max_q_seqlen,
                                                model_metas=model_metas)
        extra_inputs = ARSpecExtraInputs(output_token_ids=extra_outputs.draft_token_ids)
        return inputs, extra_inputs

    def update_decoding_for_next_step(self, model_inputs: 'ModelInputs', next_token_ids: torch.Tensor, model_metas: Any,
                                      extra_inputs: ARSpecExtraInputs, extra_outputs: ARSpecExtraOutputs):
        """Step next inputs."""
        model_inputs.model_metas = model_metas
        step_seqlens = model_inputs.seq_length
        batch_size = step_seqlens.size(0)

        # update extra inputs
        extra_inputs.output_token_ids = extra_outputs.draft_token_ids

        # update inputs
        step_seqlens = model_inputs.seq_length - extra_inputs.num_rejected_tokens
        input_ids = next_token_ids.new_empty((batch_size, self.num_spec_tokens + 1))
        input_ids[:, 0] = next_token_ids
        input_ids[:, 1:] = extra_inputs.output_draft_token_ids
        input_ids = input_ids.flatten()[None, :]
        model_inputs.step(input_ids, step_seqlens)
        return model_inputs, extra_inputs

    def post_sampling(self, inputs: 'ModelInputs', logits: torch.Tensor, next_token_ids: torch.LongTensor,
                      extra_inputs: ARSpecExtraInputs):
        """Post sampling."""
        return next_token_ids, extra_inputs

    def make_dummy_next_token(self, inputs: 'ModelInputs', logits: torch.Tensor, extra_inputs: ExtraInputs):
        """Make dummy next token for broadcast."""
        with torch.inference_mode():
            next_token_ids = inputs.input_ids.new_zeros(logits.size(0))
            extra_inputs.output_draft_token_ids = inputs.input_ids.new_zeros((logits.size(0), self.num_spec_tokens))
            extra_inputs.num_rejected_tokens = inputs.input_ids.new_zeros(logits.size(0))
        return next_token_ids, extra_inputs

    @contextmanager
    def broadcast_next_token(self, next_token_ids: torch.Tensor, extra_inputs: ARSpecExtraInputs,
                             dist_ctx: DistContext):
        """Broadcast next token ids and extra inputs."""
        tp_gpu_group = dist_ctx.attn_tp_group.gpu_group
        rank = dist.get_global_rank(tp_gpu_group, 0)
        dist.broadcast(next_token_ids, src=rank, group=tp_gpu_group, async_op=True)
        handle = extra_inputs.broadcast(src=rank, group=tp_gpu_group, async_op=True)
        yield
        handle.wait()
