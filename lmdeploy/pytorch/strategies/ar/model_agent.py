# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import record_function

from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

from ..base.model_agent import ExtraInputs, ExtraOutputs, ModelAgentStrategy, StoppingCriteria

SeqList = list[SchedulerSequence]


def get_model_inputs_next_decoding(inputs: ModelInputs, input_ids: torch.Tensor, max_q_seqlen: int,
                                   model_metas) -> ModelInputs:
    """Next decoding step."""
    if input_ids.dim() == 1:
        input_ids = input_ids[None, :]
    state_offsets = inputs.state_offsets
    if state_offsets is not None:
        state_offsets = state_offsets.clone()

    # mrope
    mrope_pos_ids = inputs.mrope_pos_ids
    if mrope_pos_ids is not None:
        index = inputs.seq_length.cumsum(0) - 1
        mrope_pos_ids = mrope_pos_ids[:, index] + 1
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
        mrope_pos_ids=mrope_pos_ids,
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
    stop_tail: torch.Tensor | None = None

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
             inputs: ModelInputs | None = None,
             extra_inputs: ARExtraInputs | None = None,
             stop_word_lens: torch.Tensor | None = None):
        """Check whether to stop generation."""
        num_appendable_ids = self.num_appendable_ids - 1
        stopped = num_appendable_ids <= 0
        stop_pos = torch.zeros_like(num_appendable_ids)

        if stop_words is None or stop_word_lens is None:
            new_tail = None
        else:
            # Set a uniform shape for token_ids for both single and multi-token stop words
            token_ids = token_ids.unsqueeze(1) if token_ids.ndim == 1 else token_ids

            sw_stopped, stop_pos, new_tail = self._check_stop_words(token_ids, stop_words, stop_word_lens)

            stopped = stopped | sw_stopped
            one_ids = torch.clamp_max(num_appendable_ids, 0)
            num_appendable_ids = torch.where(sw_stopped, one_ids, num_appendable_ids)

        return (stopped, stop_pos, ARStoppingCriteria(num_appendable_ids=num_appendable_ids, stop_tail=new_tail))

    def _check_stop_words(self, token_ids: torch.Tensor, stop_words: torch.Tensor, stop_word_lens: torch.Tensor):
        """Vectorized multi-token stop word detection.

        Args:
            token_ids: [batch, step_len], -1 for invalid positions.
                       Modified **in-place** (tokens after stop are set to -1).
            stop_words: [batch, num_seqs, max_slen]
            stop_word_lens: [batch, num_seqs]

        Returns:
            sw_stopped: [batch] bool
            stop_pos:   [batch] long – step-relative index of the stop token
            new_tail:   [batch, tail_len] or None
        """
        max_slen = int(stop_word_lens.max().item())

        if max_slen <= 1:
            # Fast path when every stop word is a single token
            return self._check_single_stop_words(token_ids, stop_words, stop_word_lens)

        # General path for multi-token stop words
        return self._check_multi_stop_words(token_ids, stop_words, stop_word_lens, max_slen)

    def _check_single_stop_words(self, token_ids: torch.Tensor, stop_words: torch.Tensor, stop_word_lens: torch.Tensor):
        """Fast path: every stop word is a single token, AR always has L==1."""
        batch_size = token_ids.size(0)
        device = token_ids.device
        targets = stop_words[:, :, 0]  # [B, S]
        valid = (stop_word_lens == 1)  # [B, S]
        # token_ids [B, 1] broadcasts against targets [B, S]
        sw_stopped = ((token_ids == targets) & valid).any(1)  # [B]
        stop_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        return sw_stopped, stop_pos, None

    def _check_multi_stop_words(self, token_ids: torch.Tensor, stop_words: torch.Tensor, stop_word_lens: torch.Tensor,
                                max_slen: int):
        """General path for multi-token stop words.

        Per-length unfold loop (each length needs its own window count), but
        iterates ``range(1, max_slen+1)`` instead of calling the GPU-syncing
        ``stop_word_lens.unique().tolist()``.
        """
        tail_len = max_slen - 1
        batch_size = token_ids.size(0)
        step_len = token_ids.size(1)
        device = token_ids.device

        # -- 1. build history = [prev_tail | token_ids] --
        prev_tail = self._get_prev_tail(batch_size, tail_len, device)
        if prev_tail is not None:
            history = torch.cat([prev_tail, token_ids], dim=1)
        else:
            history = token_ids
        hist_len = history.size(1)

        # -- 2. sliding-window matching per length --
        NO_MATCH = hist_len
        best_end = history.new_full((batch_size, ), NO_MATCH)
        for slen in range(1, max_slen + 1):
            if hist_len < slen:
                continue
            windows = history.unfold(1, slen, 1)  # [B, W, slen]
            targets = stop_words[:, :, :slen]  # [B, S, slen]
            len_mask = (stop_word_lens == slen)  # [B, S]

            match = (windows.unsqueeze(2) == targets.unsqueeze(1)).all(-1)
            match = match & len_mask.unsqueeze(1)
            match_any = match.any(2)  # [B, W]

            min_win = max(0, tail_len - slen + 1)
            if min_win > 0:
                match_any[:, :min_win] = False

            has_match = match_any.any(1)
            first_win = match_any.int().argmax(1)
            end_pos = first_win + slen - 1
            better = has_match & (end_pos < best_end)
            best_end = torch.where(better, end_pos, best_end)

        sw_stopped = best_end < NO_MATCH

        # -- 3. compute stop_pos and mask trailing tokens --
        step_stop_pos = best_end - tail_len
        stop_pos = torch.where(sw_stopped, step_stop_pos, sw_stopped.new_zeros(batch_size, dtype=torch.long))

        col_idx = torch.arange(step_len, device=device)
        after_stop = (col_idx > step_stop_pos.unsqueeze(1)) & sw_stopped.unsqueeze(1)
        token_ids[after_stop] = -1

        # -- 4. update tail --
        new_tail = self._build_new_tail(history, tail_len, sw_stopped, best_end, token_ids)

        return sw_stopped, stop_pos, new_tail

    def _get_prev_tail(self, batch_size: int, tail_len: int, device: torch.device) -> torch.Tensor | None:
        """Return the previous tail padded/trimmed to ``tail_len``."""
        if tail_len <= 0:
            return None
        if self.stop_tail is None:
            return torch.zeros(batch_size, tail_len, dtype=torch.long, device=device)
        prev = self.stop_tail.to(device)
        pt_len = prev.size(1)
        if pt_len < tail_len:
            prev = torch.nn.functional.pad(prev, (tail_len - pt_len, 0), value=-1)
        elif pt_len > tail_len:
            prev = prev[:, -tail_len:]
        return prev

    @staticmethod
    def _build_new_tail(history: torch.Tensor, tail_len: int, sw_stopped: torch.Tensor, best_end: torch.Tensor,
                        token_ids: torch.Tensor) -> torch.Tensor | None:
        """Gather the last ``tail_len`` valid tokens from *history*."""
        if tail_len <= 0:
            return None
        valid_counts = (token_ids >= 0).sum(1)
        effective_end = torch.where(sw_stopped, best_end, tail_len + valid_counts - 1)
        effective_end = effective_end.clamp(min=tail_len - 1)

        offsets = torch.arange(tail_len, device=history.device)
        indices = (effective_end - tail_len + 1).unsqueeze(1) + offsets.unsqueeze(0)
        indices = indices.clamp(min=0, max=history.size(1) - 1)
        return history.gather(1, indices)


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
                           model_outputs: dict[str, torch.Tensor], **kwargs) -> ARExtraInputs:
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
    ) -> tuple['ModelInputs', ARExtraInputs]:
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
