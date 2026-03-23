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
    state_offsets = inputs.state_offsets
    if state_offsets is not None:
        state_offsets = state_offsets.clone()
    return ModelInputs(
        input_ids=input_ids,
        seq_length=torch.full_like(inputs.seq_length, max_q_seqlen),
        history_lengths=inputs.history_lengths + inputs.seq_length,
        block_offsets=inputs.block_offsets,
        is_decoding=True,
        num_ignored_history=inputs.num_ignored_history.clone(),
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=inputs.max_kv_seqlen + max_q_seqlen,
        sum_kv_seqlen=inputs.sum_kv_seqlen + inputs.seq_length.numel() * inputs.max_q_seqlen,
        local_adapter_ids=inputs.local_adapter_ids,
        model_metas=model_metas,
        state_offsets=state_offsets,
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
    # Tail of previously generated tokens, shape [batch, tail_len].
    # Maintained across steps so that multi-token stop sequences spanning two
    # decode steps are detected without relying on the (pipelined) generated_ids
    # from SamplingInputs, which lags one step behind.
    stop_tail: Optional[torch.Tensor] = None

    def clone(self):
        """clone."""
        tail = self.stop_tail.clone() if self.stop_tail is not None else None
        return ARStoppingCriteria(num_appendable_ids=self.num_appendable_ids, stop_tail=tail)

    def merge(self, other: 'ARStoppingCriteria'):
        """Merge two stopping criteria."""
        new_num = torch.cat([self.num_appendable_ids, other.num_appendable_ids], dim=0)
        t0, t1 = self.stop_tail, other.stop_tail
        if t0 is None and t1 is None:
            new_tail = None
        else:
            bs0 = self.num_appendable_ids.size(0)
            bs1 = other.num_appendable_ids.size(0)
            dev = (t0 if t0 is not None else t1).device
            if t0 is None:
                t0 = torch.zeros(bs0, t1.size(1), dtype=torch.long, device=dev)
            if t1 is None:
                t1 = torch.zeros(bs1, t0.size(1), dtype=torch.long, device=dev)
            # Pad the shorter tail to the same length.
            l0, l1 = t0.size(1), t1.size(1)
            if l0 < l1:
                t0 = torch.nn.functional.pad(t0, (l1 - l0, 0))
            elif l1 < l0:
                t1 = torch.nn.functional.pad(t1, (l0 - l1, 0))
            new_tail = torch.cat([t0, t1], dim=0)
        return ARStoppingCriteria(num_appendable_ids=new_num, stop_tail=new_tail)

    def update(self, delta: ModelInputsDelta):
        """Update stopping criteria."""
        indices = delta.indices
        new_num = self.num_appendable_ids[indices]
        new_tail = self.stop_tail[indices] if self.stop_tail is not None else None
        return ARStoppingCriteria(num_appendable_ids=new_num, stop_tail=new_tail)

    @record_function('stopping_criteria')
    def step(self,
             token_ids: torch.Tensor,
             stop_words: torch.Tensor,
             inputs: Optional[ModelInputs] = None,
             extra_inputs: Optional[ARExtraInputs] = None,
             stop_word_lens: Optional[torch.Tensor] = None,
             generated_ids: Optional[torch.Tensor] = None):
        """Check whether to stop generation."""
        num_appendable_ids = self.num_appendable_ids - 1
        stopped = num_appendable_ids <= 0
        stop_pos = torch.zeros_like(num_appendable_ids)

        if stop_words is not None and stop_word_lens is not None:
            max_slen = int(stop_word_lens.max().item())
            tail_len = max(0, max_slen - 1)
            batch_size = stop_words.size(0)
            num_seqs = stop_words.size(1)
            sw_stopped = torch.zeros(batch_size, dtype=torch.bool, device=stop_words.device)

            # Normalise to [batch, step_len] as a view so in-place masking propagates.
            token_ids_was_1d = (token_ids.ndim == 1)
            if token_ids_was_1d:
                token_ids = token_ids.unsqueeze(1)

            new_tail = torch.zeros(
                (batch_size, tail_len), dtype=torch.long, device=token_ids.device) if tail_len > 0 else None

            for bidx in range(batch_size):
                step_tokens = token_ids[bidx]
                valid_tokens = step_tokens[step_tokens >= 0]

                # Retrieve the tail from the previous step for this batch item.
                if self.stop_tail is not None and tail_len > 0:
                    prev_tail = self.stop_tail[bidx].to(token_ids.device)
                    # Trim or pad to the current tail_len.
                    if prev_tail.size(0) >= tail_len:
                        prev_tail = prev_tail[-tail_len:]
                    else:
                        prev_tail = torch.nn.functional.pad(prev_tail, (tail_len - prev_tail.size(0), 0))
                else:
                    prev_tail = token_ids.new_zeros(tail_len)

                if valid_tokens.numel() == 0:
                    # No new tokens this step; carry the tail forward unchanged.
                    if new_tail is not None:
                        new_tail[bidx] = prev_tail
                    continue

                # History = tail of previous steps + tokens from this step.
                history = torch.cat([prev_tail, valid_tokens]) if tail_len > 0 else valid_tokens
                hist_len = history.size(0)
                stop_pos_bidx = valid_tokens.numel() - 1  # default: last valid token

                for si in range(num_seqs):
                    slen = int(stop_word_lens[bidx, si].item())
                    if slen <= 0 or hist_len < slen:
                        continue
                    target = stop_words[bidx, si, :slen]
                    # Scan positions whose end falls within the current step tokens
                    # (end_pos >= tail_len ensures at least one new token is included).
                    for end_pos in range(max(slen - 1, tail_len), hist_len):
                        if (history[end_pos - slen + 1:end_pos + 1] == target).all():
                            sw_stopped[bidx] = True
                            step_end_pos = end_pos - tail_len  # 0-indexed within valid_tokens
                            stop_pos_bidx = min(stop_pos_bidx, step_end_pos)
                            break
                    if sw_stopped[bidx]:
                        break

                if sw_stopped[bidx]:
                    stop_pos[bidx] = stop_pos_bidx
                    # Mask tokens generated after the stop position in the same step.
                    if token_ids.size(1) > (stop_pos_bidx + 1):
                        token_ids[bidx, stop_pos_bidx + 1:] = -1
                    effective_tokens = valid_tokens[:stop_pos_bidx + 1]
                else:
                    effective_tokens = valid_tokens

                # Update tail: last tail_len tokens of [prev_tail, effective_tokens].
                if new_tail is not None:
                    combined = torch.cat([prev_tail, effective_tokens])
                    new_tail[bidx] = combined[-tail_len:]

            if token_ids_was_1d and token_ids.size(1) == 1:
                token_ids = token_ids.squeeze(1)

            stopped = stopped | sw_stopped
            one_ids = torch.clamp_max(num_appendable_ids, 0)
            num_appendable_ids = torch.where(sw_stopped, one_ids, num_appendable_ids)

        else:
            new_tail = None

        new_stopping = ARStoppingCriteria(num_appendable_ids=num_appendable_ids, stop_tail=new_tail)
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
