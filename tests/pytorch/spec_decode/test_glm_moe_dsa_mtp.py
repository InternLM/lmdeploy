# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

import torch
from torch import nn

from lmdeploy.pytorch.models.deepseek_v32 import DSATopKIndicesBuffer
from lmdeploy.pytorch.models.glm_moe_dsa_mtp import GlmMoeDsaMultiTokenPredictor
from lmdeploy.pytorch.spec_decode.proposers.base import BaseSpecProposer
from lmdeploy.pytorch.spec_decode.proposers.deepseek_mtp import DeepseekMTP
from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs


class _GuidedHelper:

    async def prepare_bitmask(self, logits, processors):
        return None

    def apply_bitmask(self, logits, bitmask):
        raise AssertionError('no bitmask was returned')

    async def accept_draft_tokens(self, draft_token_ids, processors):
        return None


class _DummyDraft(nn.Module):

    uses_dsa_topk_buffer = True

    def __init__(self):
        super().__init__()
        self.compacted = None

    def compact_topk_indices(self, row_indices):
        self.compacted = row_indices

    def prepare_hidden_states_for_logits(self, hidden_states):
        return hidden_states + 10


class _DummyTarget(nn.Module):

    def get_logits(self, hidden_states):
        self.hidden_states = hidden_states
        logits = hidden_states.new_zeros(1, hidden_states.size(1), 8)
        logits[..., 5] = 1
        return logits


def test_proposer_reuses_topk_and_recycles_raw_hidden_states():
    proposer = object.__new__(DeepseekMTP)
    proposer.model = _DummyDraft()
    proposer.target_model = _DummyTarget()
    proposer.guided_helper = _GuidedHelper()
    hidden_states = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    outputs = dict(hidden_states=hidden_states, model_metas=[{'keep': 1}, {'keep': 2}])
    extra_inputs = ARSpecExtraInputs(last_token_indices=torch.tensor([2, 0]))

    draft_ids, model_metas, recurrent_hidden = asyncio.run(
        proposer.get_outputs(outputs, model_inputs=None, extra_inputs=extra_inputs))

    selected_hidden = hidden_states[:, [2, 0]]
    assert draft_ids.tolist() == [[5], [5]]
    assert torch.equal(recurrent_hidden, selected_hidden)
    assert torch.equal(proposer.target_model.hidden_states, selected_hidden + 10)
    assert proposer.model.compacted is extra_inputs.last_token_indices
    assert [meta['keep'] for meta in model_metas] == [1, 2]
    assert all(meta['skip_topk'] for meta in model_metas)


def test_proposer_binds_target_embedding_and_topk_buffer(monkeypatch):

    class _Draft(nn.Module):

        def __init__(self):
            super().__init__()
            self.embed_tokens = None
            self.topk_indices_buffer = None

        def set_input_embeddings(self, embed_tokens):
            self.embed_tokens = embed_tokens

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_topk_indices_buffer(self, topk_indices_buffer):
            self.topk_indices_buffer = topk_indices_buffer

    class _Target(nn.Module):

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.model = nn.Module()
            self.model.topk_indices_buffer = DSATopKIndicesBuffer(topk=2)

        def get_input_embeddings(self):
            return self.embedding

    draft = _Draft()
    target = _Target()
    proposer = object.__new__(DeepseekMTP)

    def build_model(self, empty_init, target_model=None, build_model_ctx=None):
        self.model = draft
        self.target_model = target_model

    monkeypatch.setattr(BaseSpecProposer, 'build_model', build_model)
    proposer.build_model(empty_init=True, target_model=target)

    assert draft.embed_tokens is target.embedding
    assert draft.topk_indices_buffer is target.model.topk_indices_buffer


def test_glm_mtp_prepares_postnorm_hidden_for_logits():

    class _SharedHead(nn.Module):

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return hidden_states + 10

    class _Layer(nn.Module):

        def __init__(self):
            super().__init__()
            self.shared_head = _SharedHead()

        def forward(self, input_ids, position_ids, previous_hidden_states, past_key_value, **kwargs):
            return previous_hidden_states + 3

    predictor = object.__new__(GlmMoeDsaMultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.mtp_start_layer_idx = 5
    predictor.num_mtp_layers = 1
    predictor.layers = nn.ModuleDict({'5': _Layer()})
    previous_hidden = torch.zeros(1, 1, 4)

    raw_hidden = predictor(torch.zeros(1, 1, dtype=torch.long),
                           torch.zeros(1, 1, dtype=torch.long),
                           previous_hidden,
                           past_key_values=[[None]],
                           inputs_embeds=torch.zeros_like(previous_hidden))
    logits_hidden = predictor.prepare_hidden_states_for_logits(raw_hidden)

    expected = previous_hidden + 13
    assert torch.equal(logits_hidden, expected)
    assert predictor.layers['5'].shared_head.calls == 1


def test_topk_buffer_compaction_preserves_selected_order():
    buffer = DSATopKIndicesBuffer(topk=2)
    rows = torch.tensor([[0, 1], [10, 11], [20, 21]], dtype=torch.int32)
    buffer.write(rows)

    compacted = buffer.compact(torch.tensor([2, 0]))

    assert torch.equal(compacted, rows[[2, 0]])
