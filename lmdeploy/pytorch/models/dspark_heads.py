# Copyright (c) OpenMMLab. All rights reserved.
"""Sequential logit-bias heads used by DSpark draft models."""

from __future__ import annotations

import torch
from torch import nn
from torch.profiler import record_function

from lmdeploy.pytorch.nn.embedding import ParallelEmbedding, ParallelLMHead


class VanillaMarkovHead(nn.Module):
    """Low-rank previous-token transition bias."""

    head_type = 'vanilla'

    def __init__(self, vocab_size: int, draft_vocab_size: int, markov_rank: int,
                 dtype=None, device=None):
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f'DSpark markov_rank must be positive, got {markov_rank}.')
        self.markov_rank = int(markov_rank)
        self.markov_w1 = ParallelEmbedding(vocab_size,
                                           self.markov_rank,
                                           padding_idx=None,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=False)
        self.markov_w2 = ParallelLMHead(draft_vocab_size,
                                        self.markov_rank,
                                        bias=False,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=False)

    def init_state(self, batch_size: int, dtype: torch.dtype,
                   device: torch.device):
        return None

    def step(self, token_ids: torch.Tensor, hidden_states: torch.Tensor | None,
             state: torch.Tensor | None):
        del hidden_states, state
        embedding = self.markov_w1(token_ids.long())
        return self.markov_w2(embedding), None


class GatedMarkovHead(VanillaMarkovHead):
    head_type = 'gated'

    def __init__(self, vocab_size: int, draft_vocab_size: int, markov_rank: int,
                 hidden_size: int, dtype=None, device=None):
        super().__init__(vocab_size, draft_vocab_size, markov_rank,
                         dtype=dtype, device=device)
        self.gate_proj = nn.Linear(hidden_size + markov_rank,
                                   markov_rank,
                                   bias=True,
                                   dtype=dtype,
                                   device=device)
        self.gate_proj.weight.requires_grad_(False)
        self.gate_proj.bias.requires_grad_(False)

    def step(self, token_ids: torch.Tensor, hidden_states: torch.Tensor | None,
             state: torch.Tensor | None):
        del state
        if hidden_states is None:
            raise ValueError('Gated DSpark head requires per-position hidden states.')
        embedding = self.markov_w1(token_ids.long())
        gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_states, embedding], dim=-1)))
        return self.markov_w2(gate.to(embedding.dtype) * embedding), None


class RNNMarkovHead(VanillaMarkovHead):
    head_type = 'rnn'

    def __init__(self, vocab_size: int, draft_vocab_size: int, markov_rank: int,
                 hidden_size: int, dtype=None, device=None):
        super().__init__(vocab_size, draft_vocab_size, markov_rank,
                         dtype=dtype, device=device)
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size,
                                    3 * markov_rank,
                                    bias=True,
                                    dtype=dtype,
                                    device=device)
        self.joint_proj.weight.requires_grad_(False)
        self.joint_proj.bias.requires_grad_(False)

    def init_state(self, batch_size: int, dtype: torch.dtype,
                   device: torch.device):
        return torch.zeros(batch_size,
                           self.markov_rank,
                           dtype=dtype,
                           device=device)

    def step(self, token_ids: torch.Tensor, hidden_states: torch.Tensor | None,
             state: torch.Tensor | None):
        if hidden_states is None or state is None:
            raise ValueError('RNN DSpark head requires hidden states and initialized state.')
        embedding = self.markov_w1(token_ids.long())
        raw_gate, raw_candidate, raw_output = self.joint_proj(
            torch.cat([state, embedding, hidden_states], dim=-1)).chunk(3, dim=-1)
        gate = torch.sigmoid(raw_gate)
        candidate = torch.tanh(raw_candidate)
        new_state = gate * state + (1 - gate) * candidate
        return self.markov_w2(torch.tanh(raw_output)), new_state


def build_dspark_head(config, dtype=None, device=None):
    """Build the checkpoint-declared sequential DSpark head."""
    vocab_size = int(config.vocab_size)
    draft_vocab_size = int(getattr(config, 'draft_vocab_size', None)
                           or vocab_size)
    markov_rank = int(getattr(config, 'markov_rank'))
    head_type = str(getattr(config, 'markov_head_type', 'vanilla')).lower()
    kwargs = dict(vocab_size=vocab_size,
                  draft_vocab_size=draft_vocab_size,
                  markov_rank=markov_rank,
                  dtype=dtype,
                  device=device)
    if head_type == 'vanilla':
        return VanillaMarkovHead(**kwargs)
    kwargs['hidden_size'] = int(config.hidden_size)
    if head_type == 'gated':
        return GatedMarkovHead(**kwargs)
    if head_type == 'rnn':
        return RNNMarkovHead(**kwargs)
    raise ValueError(f'Unsupported DSpark markov_head_type={head_type!r}.')


@record_function('dspark_proposal_epilogue')
def compute_dspark_proposal_ids(model: nn.Module,
                                hidden_states: torch.Tensor,
                                input_ids: torch.Tensor) -> torch.Tensor:
    """Run the fixed sequential DSpark proposal inside model forward.

    Keeping this tensor-only, fixed-width loop behind the model boundary makes base logits, every Markov step, greedy
    sampling, vocabulary mapping, and the proposal output part of CUDA Graph capture/replay.
    """
    query_len = int(model.dspark_draft_query_len)
    num_spec_tokens = int(model.dspark_num_speculative_tokens)
    if hidden_states.size(0) == 1:
        hidden_states = hidden_states[0]
    batch_size = input_ids.numel() // query_len
    hidden_states = hidden_states.reshape(
        batch_size, query_len, *hidden_states.shape[1:])
    if model.dspark_sample_from_anchor:
        sample_hidden = hidden_states
    else:
        sample_hidden = hidden_states[:, 1:]
    if sample_hidden.size(1) != num_spec_tokens:
        raise RuntimeError(
            'DSpark sampled hidden-row count does not match the configured '
            f'width: {sample_hidden.size(1)} vs {num_spec_tokens}.')

    base_logits = model.compute_base_logits(sample_hidden)
    prev = input_ids.reshape(batch_size, query_len)[:, 0]
    state = model.init_sequential_state(
        batch_size, sample_hidden.dtype, sample_hidden.device)
    proposal_ids = torch.empty(
        batch_size, num_spec_tokens, dtype=torch.long,
        device=sample_hidden.device)
    for idx in range(num_spec_tokens):
        bias, state = model.sequential_head_step(
            prev, sample_hidden[:, idx], state)
        draft_ids = (base_logits[:, idx] + bias).argmax(dim=-1)
        prev = model.map_draft_to_target(draft_ids)
        proposal_ids[:, idx].copy_(prev)
    return proposal_ids
