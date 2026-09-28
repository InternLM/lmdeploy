# Copyright (c) OpenMMLab. All rights reserved.
"""Additional GLM prefix contracts: transport, DP counts and ancestor identity."""
import asyncio
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.configurations.glm5_next import update_cache_config
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
from lmdeploy.pytorch.strategies.ar.model_inputs import get_model_inputs_next_decoding
from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


def inputs(start=0, length=3, **kwargs):
    return ModelInputs(input_ids=torch.arange(start, start + length)[None],
                       seq_length=torch.tensor([length]), history_lengths=torch.tensor([start]),
                       block_offsets=torch.zeros((1, 4), dtype=torch.long), is_decoding=False,
                       num_ignored_history=torch.tensor([0]), max_q_seqlen=length,
                       max_kv_seqlen=start + length, sum_kv_seqlen=start + length, **kwargs)


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('first,last,adjust', [(True, False, -1), (False, False, 0), (False, True, 1)])
def test_dp_draft_counts_match_actual_shift_contract(monkeypatch, full, first, last, adjust):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.dist_config = SimpleNamespace(world_size=2)
    agent.dist_ctx = SimpleNamespace(cpu_group=None)
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: True)
    agent.enable_microbatch = False
    agent.state = SimpleNamespace(is_sleeping=False)
    agent.patched_model = SimpleNamespace(update_inputs=lambda value: value)

    class Gather:
        def __init__(self, values, size, **kwargs):
            self.values = torch.tensor([values] * size)

        async def async_wait(self):
            return self.values

    monkeypatch.setattr('lmdeploy.pytorch.engine.model_agent.agent.DistGatherScalar', Gather)
    value = inputs(is_chunk=True, is_first_chunk=first, is_last_chunk=last, draft_full_prefill=full)
    value.build_dp_meta = lambda counts: setattr(value, 'dp_meta', SimpleNamespace())
    result, sleeping = asyncio.run(agent._prepare_dp_v1(value))
    assert not sleeping
    assert result.dp_meta.dp_draft_num_tokens == [3 if full else 3 + adjust] * 2


def test_lookahead_transport_and_decode_reset():
    value = inputs(draft_full_prefill=True, draft_chunk_next_token_ids=torch.tensor([3]))
    transported = value.to_device('cpu').clone()
    assert transported.draft_full_prefill
    assert transported.draft_chunk_next_token_ids.tolist() == [3]
    decode = get_model_inputs_next_decoding(transported, torch.tensor([[3]]), 1, None)
    assert not decode.draft_full_prefill and decode.draft_chunk_next_token_ids is None
    # Also exercise ModelInputs.step() independently of merge_prefill.
    transported.is_decoding = True
    stepped = transported.step(torch.tensor([[3]]))
    assert not stepped.draft_full_prefill and stepped.draft_chunk_next_token_ids is None


def test_full_draft_chunks_equal_unpartitioned_shifted_input_and_hidden():
    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {}
    collected_ids, collected_hidden = [], []
    hidden = torch.arange(18).reshape(1, 9, 2).float()
    for chunk in range(3):
        start = chunk * 3
        value = inputs(start=start, is_chunk=True, is_first_chunk=chunk == 0,
                       is_last_chunk=chunk == 2, draft_full_prefill=True,
                       draft_chunk_next_token_ids=torch.tensor([start + 3]) if chunk < 2 else None)
        extra = ARSpecExtraInputs(target_hidden_states=hidden[:, start:start + 3],
                                 next_token_ids=torch.tensor([9]), last_token_indices=torch.tensor([2]))
        draft, _ = agent._prepare_inputs_from_main(value, extra)
        assert draft.history_lengths.item() == start
        collected_ids.append(draft.input_ids)
        collected_hidden.append(draft.target_hidden_states)
    assert torch.equal(torch.cat(collected_ids, 1), torch.arange(1, 10)[None])
    assert torch.equal(torch.cat(collected_hidden, 1), hidden)
    assert agent._prev_chunk_last == {}


@pytest.mark.parametrize('changed,expected_hit', [(256, 0), (512, 256), (768, 512)])
def test_shifted_dependency_at_every_ancestor_owner(changed, expected_hit):
    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=64,
                        enable_prefix_caching=True, states_shapes=[((1,), torch.float32)],
                        num_state_caches=8, prefix_cache_state_budget=4)
    update_cache_config(cache, token_lookahead=1)
    scheduler = Scheduler(SchedulerConfig(max_batches=2, max_session_len=16384), cache,
                          SequenceMeta(64, strategy=ARSpecSequenceStrategy()))
    trie = scheduler.block_trie
    tokens = list(range(770))
    producer = scheduler.add_session(0).add_sequence(tokens)
    scheduler.block_manager.allocate(producer)
    trie.allocate(producer)
    for step in (256, 512, 768):
        assert trie.state_checkpoints.reserve_save(producer, step=step) >= 0
        assert trie.state_checkpoints.publish_save(producer)
    branch = tokens.copy()
    branch[changed] = 9000
    consumer = scheduler.add_session(1).add_sequence(branch)
    trie.match(consumer)
    assert consumer.num_history_ids == expected_hit
    scheduler.block_manager.allocate(consumer)
    private = consumer.logical_blocks.get_real_blocks().copy()
    trie.allocate(consumer)
    assert (consumer.logical_blocks.get_real_blocks() == private).all()


@pytest.mark.parametrize('embedded', [False, True])
def test_full_draft_packed_prefill_keeps_each_request_boundary(embedded):
    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {}
    agent.proposer = SimpleNamespace(embed_input_ids=lambda ids: ids.float().unsqueeze(-1).repeat(1, 2))
    value = inputs(length=6, draft_full_prefill=True)
    value.input_ids = torch.tensor([[10, 11, 12, 20, 30, 31]])
    value.seq_length = torch.tensor([3, 1, 2])
    value.history_lengths = torch.tensor([256, 0, 512])
    value.block_offsets = torch.zeros((3, 4), dtype=torch.long)
    value.num_ignored_history = torch.zeros(3, dtype=torch.long)
    value.max_q_seqlen = 3
    value.max_kv_seqlen = 514
    value.sum_kv_seqlen = 774
    hidden = torch.arange(12).reshape(1, 6, 2).float()
    positions = torch.tensor([[256, 257, 258, 0, 512, 513]])
    embeddings = value.input_ids.float().unsqueeze(-1).repeat(1, 1, 2) if embedded else None
    extra = ARSpecExtraInputs(target_hidden_states=hidden, target_position_ids=positions,
                             target_inputs_embeds=embeddings, next_token_ids=torch.tensor([13, 21, 32]),
                             last_token_indices=torch.tensor([2, 3, 5]))
    draft, updated = agent._prepare_inputs_from_main(value, extra)
    expected = torch.tensor([[11, 12, 13, 21, 31, 32]])
    assert torch.equal(draft.input_ids, expected)
    assert torch.equal(draft.target_hidden_states, hidden)
    assert torch.equal(draft.target_position_ids, positions)
    assert torch.equal(draft.history_lengths, value.history_lengths)
    assert torch.equal(draft.seq_length, value.seq_length)
    assert draft.sum_kv_seqlen == value.sum_kv_seqlen
    assert updated.last_token_indices.tolist() == [2, 3, 5]
    if embedded:
        assert torch.equal(draft.target_inputs_embeds, expected.float().unsqueeze(-1).repeat(1, 1, 2))
    else:
        assert draft.target_inputs_embeds is None
    assert agent._prev_chunk_last == {}


@pytest.mark.parametrize('spans', [[], [(500, 570)], [(512, 582)], [(0, 700)],
                                  [(512, 1212)], [(500, 570), (570, 1250)],
                                  [(500, 570), (571, 641)], [(64, 704)]])
def test_multimodal_full_frontier_chunks_equal_unpartitioned_embeddings(spans):
    """Keep a vision token off the vocabulary-only chunk lookahead path."""
    from lmdeploy.pytorch.engine.inputs_maker import LongContextChunker
    from lmdeploy.pytorch.long_context import get_long_context_chunk_limit, plan_long_context_chunk
    from lmdeploy.pytorch.multimodal.data_type import MultiModalData
    from lmdeploy.vl.constants import Modality

    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=64,
                        enable_prefix_caching=True, states_shapes=[((1,), torch.float32)],
                        num_state_caches=8, prefix_cache_state_budget=4)
    update_cache_config(cache, token_lookahead=1)
    scheduler = Scheduler(SchedulerConfig(max_batches=2, max_session_len=8192), cache,
                          SequenceMeta(64, strategy=ARSpecSequenceStrategy()))
    n = 2050
    tokens = list(range(n))
    mm = {Modality.IMAGE: [MultiModalData(torch.ones(end - start, 2), start, end)
                           for start, end in spans]}
    seq = scheduler.add_session(0).add_sequence(tokens, multimodals=mm)
    # Distinct actual vision embeddings, never equal to placeholder embeddings.
    embeddings = torch.arange(n + 1).float()[None, :, None].repeat(1, 1, 2)
    for start, end in spans:
        embeddings[:, start:end] += 10000
    hidden = torch.arange(n * 2).reshape(1, n, 2).float()
    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {}
    agent.proposer = SimpleNamespace(embed_input_ids=lambda ids: ids.float().unsqueeze(-1).repeat(1, 2))
    chunker = LongContextChunker(512)
    chunker.set_seq(seq)
    collected, seen, ends = [], [], []
    while seq.num_history_ids < n:
        start = seq.num_history_ids
        size, payload = chunker.next_chunk_size()
        end = start + size
        assert size > 0
        # Scheduler and input maker must allocate/forward exactly the same span.
        plan = plan_long_context_chunk(seq, get_long_context_chunk_limit(seq, chunker.max_prefill_num),
                                       include_multimodals=False)
        assert plan.chunk_end == end
        assert all(not (a <= end < b) for a, b in spans)
        seen.extend((data.start, data.end) for data in (payload or {}).get(Modality.IMAGE, []))
        last = end == n
        value = inputs(start=start, length=size, is_chunk=not last, is_first_chunk=start == 0,
                       draft_full_prefill=True,
                       draft_chunk_next_token_ids=None if last else torch.tensor([end]))
        extra = ARSpecExtraInputs(target_hidden_states=hidden[:, start:end],
                                 target_inputs_embeds=embeddings[:, start:end],
                                 next_token_ids=torch.tensor([n]), last_token_indices=torch.tensor([size - 1]))
        draft, _ = agent._prepare_inputs_from_main(value, extra)
        assert torch.equal(draft.target_hidden_states, hidden[:, start:end])
        collected.append(draft.target_inputs_embeds)
        ends.append(end)
        seq.set_step(end)
        chunker.set_seq(seq)
    assert seen == spans
    assert torch.equal(torch.cat(collected, 1), embeddings[:, 1:])
    if not spans:
        assert ends == [512, 1024, 1536, 2048, 2050]
    # A full-frontier checkpoint at image.start would contain the same wrong
    # dependency, so publication and matching must reject/rewind that boundary.
    scheduler.block_manager.allocate(seq)
    for start, end in spans:
        if start:
            assert not seq.is_prefix_cache_boundary_safe(start)
            assert seq.clamp_prefix_cache_match_step(start) < start
            if start % 64 == 0:
                assert scheduler.block_trie.state_checkpoints.reserve_save(seq, step=start) == -1
