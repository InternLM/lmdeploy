# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.engine.inputs_maker import _ForwardInputsTask
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.kv_connector import (
    KVConnectorOutput,
    KVConnectorResult,
    KVConnectorStepInput,
    KVLoadResult,
)
from lmdeploy.pytorch.kv_connector.mooncake.store import scheduler as scheduler_module
from lmdeploy.pytorch.kv_connector.mooncake.store.data import build_prefix_block_hashes
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.prefix_cache_state import PrefixRecomputeOverlap
from lmdeploy.pytorch.spec_decode.guided_spec_helper import GuidedSpecHelper
from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


def _cache_config(role='kv_both'):
    return CacheConfig(
        max_batches=1,
        block_size=4,
        num_cpu_blocks=0,
        num_gpu_blocks=8,
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role=role,
        ),
    )


def _request(
    token_ids,
    *,
    seq_id=17,
    adapter_name=None,
    multimodal=False,
    embeddings=False,
):
    return SimpleNamespace(
        seq_id=seq_id,
        num_history_ids=0,
        adapter_name=adapter_name,
        all_ids=np.asarray(token_ids, dtype=np.int64),
        history_multimodals=SimpleNamespace(empty=lambda: not multimodal),
        history_embeddings=[object()] if embeddings else [],
        prefix_cache=SimpleNamespace(recompute_overlap=PrefixRecomputeOverlap()),
        clamp_prefix_cache_match_step=lambda step: step,
        get_prefix_cache_max_candidate_step=lambda: len(token_ids) - 1,
        get_prefix_cache_max_match_step=lambda: (len(token_ids) - 1) // 4 * 4,
    )


def _connector_step(
    running=(),
    token_lens=(),
    block_ids=(),
    logical_block_ids=(),
):
    return KVConnectorStepInput(
        running=list(running),
        connector_token_lens=tuple(token_lens),
        connector_block_ids=tuple(block_ids),
        connector_logical_block_ids=tuple(logical_block_ids),
    )


# Keep the name used by the MTP save-boundary tests while sharing the main
# branch's structured connector-step fixture.
_scheduler_output = _connector_step


def test_prefix_block_hashes_are_stable_chained_and_incremental():
    tokens = np.arange(13, dtype=np.int64)
    expected = (
        'bbc0293b578f95f34bff8c5d55741da798df85e75457b19a537d5a8a6592b9e1',
        'd81c87475b16382cb7a79aa3d582ed248abd8f9136b04058fea1ea3bc88687b7',
        '1d9148682b04441ba83bb4f39375e41e45cf05ed3554d23275a859ab428d4c77',
    )

    full = build_prefix_block_hashes(tokens, 4, extra_identity='adapter-a')
    prefix = build_prefix_block_hashes(tokens[:8], 4, extra_identity='adapter-a')
    extended = build_prefix_block_hashes(
        tokens,
        4,
        extra_identity='adapter-a',
        previous_hashes=prefix,
    )

    assert tuple(block_hash.hex() for block_hash in full) == expected
    assert build_prefix_block_hashes(tokens.tolist(), 4, extra_identity='adapter-a') == full
    assert extended == full
    assert build_prefix_block_hashes(tokens[:12], 4, extra_identity='adapter-a') == full
    assert build_prefix_block_hashes(tokens, 4, extra_identity='adapter-b') != full


def test_scheduler_extends_hashes_and_reports_pending_miss_and_hit(monkeypatch):
    scheduler = MooncakeStoreScheduler(_cache_config())
    request = _request(range(9), adapter_name='adapter-a')
    lookup_results = iter((None, 0, 12))
    lookup_calls = []
    hash_extensions = []
    original_build = scheduler_module.build_prefix_block_hashes

    def record_build(token_ids, block_size, **kwargs):
        hash_extensions.append(len(kwargs['previous_hashes']))
        return original_build(token_ids, block_size, **kwargs)

    def lookup(req_id, token_len, block_hashes, non_block):
        lookup_calls.append((req_id, token_len, tuple(block_hashes), non_block))
        return next(lookup_results)

    monkeypatch.setattr(scheduler_module, 'build_prefix_block_hashes', record_build)
    scheduler.client.lookup = lookup

    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)

    request.all_ids = np.arange(13, dtype=np.int64)
    request.get_prefix_cache_max_candidate_step = lambda: 12
    request.get_prefix_cache_max_match_step = lambda: 12
    assert scheduler.get_num_new_matched_tokens(request, 4) == (8, True)

    assert hash_extensions == [0, 2]
    assert lookup_calls[0][3] is True
    assert lookup_calls[2][2][:2] == lookup_calls[0][2]
    scheduler.shutdown()


def test_scheduler_reuses_positive_lookup_until_allocation():
    scheduler = MooncakeStoreScheduler(_cache_config())
    request = _request(range(17))
    scheduler.client.lookup = Mock(return_value=12)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
    # Paging can reject capacity without binding destination blocks. A retry
    # must reuse the retained positive result instead of repeating remote I/O.
    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
    scheduler.client.lookup.assert_called_once()

    scheduler.update_state_after_alloc(request, (21, 22, 23), 12)
    assert request.seq_id not in scheduler._lookup_plans
    scheduler.shutdown()


@pytest.mark.parametrize(
    ('remote_hit', 'local_hit', 'expected'),
    [
        pytest.param(0, 0, 0, id='miss'),
        pytest.param(4, 0, 0, id='only-one-block'),
        pytest.param(8, 0, 4, id='partial-hit'),
        pytest.param(12, 0, 8, id='full-hit'),
        pytest.param(8, 4, 0, id='no-extension-after-trimming'),
        pytest.param(12, 4, 4, id='local-prefix'),
        pytest.param(12, 5, 3, id='partial-local-block'),
    ],
)
def test_spec_external_lookup_drops_last_actual_hit_block(remote_hit, local_hit, expected):
    block_size = 4
    paging_scheduler = Scheduler(
        scheduler_config=SchedulerConfig(
            max_batches=1,
            max_session_len=64,
            max_request_output_len=16,
            eviction_type='recompute',
        ),
        cache_config=CacheConfig(
            max_batches=1,
            block_size=block_size,
            num_cpu_blocks=0,
            num_gpu_blocks=8,
            enable_prefix_caching=True,
        ),
        seq_meta=SequenceMeta(block_size, strategy=ARSpecSequenceStrategy()),
    )
    request = paging_scheduler.add_session(0).add_sequence(range(13))
    mooncake_scheduler = MooncakeStoreScheduler(_cache_config())
    mooncake_scheduler.client.lookup = Mock(side_effect=(None, remote_hit))

    # The prompt cap alone cannot protect a shorter remote hit. Query the full
    # candidate prefix, then drop the last block from the actual lookup result.
    assert request.get_prefix_cache_max_match_step() == 8
    assert mooncake_scheduler.get_num_new_matched_tokens(request, local_hit) == (None, False)
    assert mooncake_scheduler.get_num_new_matched_tokens(request, local_hit) == (expected, expected > 0)

    mooncake_scheduler.client.lookup.assert_called_with(
        request.seq_id,
        12,
        build_prefix_block_hashes(request.all_ids[:12], block_size),
        non_block=True,
    )
    if expected:
        # Allocation retries reuse the already-trimmed result; loading must not
        # drop another block or include the block reserved for recomputation.
        assert mooncake_scheduler.get_num_new_matched_tokens(request, local_hit) == (expected, True)
        assert mooncake_scheduler.client.lookup.call_count == 2
        load_start = local_hit // block_size * block_size
        load_end = remote_hit - block_size
        block_ids = tuple(range(load_start // block_size, load_end // block_size))
        mooncake_scheduler.update_state_after_alloc(request, block_ids, load_end - load_start)
        load = mooncake_scheduler.build_connector_meta(_scheduler_output()).load_requests[0]
        assert load.remote_block_count == load_end // block_size
        assert load.block_ids == block_ids
        assert load.block_hashes == build_prefix_block_hashes(request.all_ids, block_size)[
            load_start // block_size:load_end // block_size]
    else:
        assert mooncake_scheduler.build_connector_meta(_scheduler_output()) is None
    mooncake_scheduler.shutdown()


@pytest.mark.parametrize(
    ('role', 'multimodal', 'embeddings'),
    [
        ('kv_producer', False, False),
        ('kv_both', True, False),
        ('kv_both', False, True),
    ],
)
def test_scheduler_filters_non_consumers_and_non_text_requests(
    role,
    multimodal,
    embeddings,
):
    scheduler = MooncakeStoreScheduler(_cache_config(role))
    if scheduler.client is not None:
        scheduler.client.lookup = Mock(side_effect=AssertionError('lookup must not run'))
    request = _request(range(9), multimodal=multimodal, embeddings=embeddings)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    if role == 'kv_producer':
        assert scheduler.client is None
    else:
        scheduler.client.lookup.assert_not_called()
    scheduler.shutdown()


def test_scheduler_cancel_retains_hashes_until_request_finishes():
    scheduler = MooncakeStoreScheduler(_cache_config())
    request = _request(range(9))
    scheduler.client.lookup = Mock(return_value=0)
    scheduler.client.discard = Mock()
    scheduler.get_num_new_matched_tokens(request, 0)
    tracker = scheduler._request_hash_trackers[request.seq_id]

    scheduler.cancel_lookup(request.seq_id)
    assert scheduler._request_hash_trackers[request.seq_id] is tracker

    assert scheduler.request_finished(request) is None
    assert request.seq_id not in scheduler._request_hash_trackers
    assert scheduler.client.discard.call_args_list == [
        ((request.seq_id, ), {}),
        ((request.seq_id, ), {}),
    ]
    scheduler.shutdown()


def test_scheduler_load_failure_falls_back_until_next_request():
    scheduler = MooncakeStoreScheduler(_cache_config())
    request = _request(range(13))
    scheduler.client.lookup = Mock(return_value=12)

    assert scheduler.get_num_new_matched_tokens(request, 4) == (8, True)
    scheduler.update_state_after_alloc(request, (31, 32), 8)

    metadata = scheduler.build_connector_meta(_connector_step())
    assert metadata is not None
    assert len(metadata.load_requests) == 1
    load_request = metadata.load_requests[0]
    assert load_request.request_id == request.seq_id
    assert load_request.block_ids == (31, 32)
    assert scheduler.build_connector_meta(_connector_step()).load_requests == ()

    assert scheduler.update_connector_output(
        KVConnectorOutput(invalid_block_ids={32})) == KVConnectorResult()
    result = scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={request.seq_id})
    )
    assert result == KVConnectorResult(
        load_results=(KVLoadResult(request.seq_id, False), ),
    )

    retry_save = scheduler.build_connector_meta(
        _connector_step(
            running=(request, ),
            token_lens=(12, ),
            block_ids=((30, 31, 32), ),
            logical_block_ids=((40, 41, 42), ),
        )).save_requests[0]
    assert retry_save.start_block == 1
    assert retry_save.block_ids == (31, 32)

    assert scheduler.get_num_new_matched_tokens(request, 4) == (0, False)
    assert scheduler.client.lookup.call_count == 1
    request.num_history_ids = 5
    assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
    assert scheduler.client.lookup.call_count == 1

    next_request = _request(range(13), seq_id=18)
    scheduler.on_new_request(next_request)
    assert scheduler.get_num_new_matched_tokens(next_request, 4) == (8, True)
    assert scheduler.client.lookup.call_count == 2
    scheduler.shutdown()


def test_scheduler_builds_incremental_save_operations_and_poll_metadata():
    scheduler = MooncakeStoreScheduler(_cache_config('kv_producer'))
    request = _request(range(17), adapter_name='adapter-a')

    first = scheduler.build_connector_meta(
        _connector_step(
            running=(request, ),
            token_lens=(10, ),
            block_ids=((31, 32, 33, 34, 35), ),
            logical_block_ids=((41, 42, 43, 44, 45), ),
        ))
    assert first is not None
    assert len(first.save_requests) == 1
    first_save = first.save_requests[0]
    assert first_save.save_id == 0
    assert first_save.request_id == request.seq_id
    assert first_save.start_block == 0
    assert first_save.block_ids == (31, 32)
    assert first_save.logical_block_ids == (41, 42)
    assert first_save.block_hashes == build_prefix_block_hashes(
        request.all_ids[:8], 4, extra_identity='adapter-a')
    assert first.get_save_block_leases()[0].logical_block_ids == (41, 42)

    # The connector keeps the engine issuing no-forward polling steps while a
    # previous save is still running.
    poll = scheduler.build_connector_meta(_connector_step())
    assert poll is not None
    assert poll.save_requests == ()

    second = scheduler.build_connector_meta(
        _connector_step(
            running=(request, ),
            token_lens=(14, ),
            block_ids=((31, 32, 33, 34, 35), ),
            logical_block_ids=((41, 42, 43, 44, 45), ),
        ))
    second_save = second.save_requests[0]
    assert second_save.save_id == 1
    assert second_save.start_block == 2
    assert second_save.block_ids == (33, )
    assert second_save.logical_block_ids == (43, )

    result = scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={0, 999}))
    assert result.completed_save_ids == frozenset({0})
    assert scheduler.build_connector_meta(_connector_step()) is not None

    result = scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={1}))
    assert result.completed_save_ids == frozenset({1})
    assert scheduler.build_connector_meta(_connector_step()) is None
    scheduler.shutdown()


@pytest.mark.parametrize(
    ('history', 'token_len', 'final_chunk'),
    [
        pytest.param(0, 8, False, id='regular-aligned'),
        pytest.param(0, 10, False, id='regular-partial-tail'),
        pytest.param(4, 8, False, id='regular-cached-prefix-aligned'),
        pytest.param(4, 10, False, id='regular-cached-prefix-partial-tail'),
        pytest.param(8, 12, True, id='final-chunk-aligned'),
        pytest.param(8, 14, True, id='final-chunk-partial-tail'),
    ],
)
def test_mtp_prefill_save_boundary_follows_written_rows(history, token_len, final_chunk):
    """Check real target sampling, MTP input preparation and save planning.

    CPU rows record the token paired with each KV row, without model weights or GPU kernels. The final MTP row must use
    the accepted target token; unverified draft rows and incomplete blocks must stay out of saves.
    """
    scheduler = MooncakeStoreScheduler(_cache_config('kv_producer'))
    request = _request(range(token_len))
    task = _ForwardInputsTask(SimpleNamespace(scheduler=scheduler, spec_decoding=True), prefill=True)
    agent = object.__new__(SpecModelAgent)
    agent._init_runtime_state()
    agent.misc_config = SimpleNamespace(logprobs_mode=None)
    agent.guided_helper = GuidedSpecHelper(None)
    sampling_inputs = SamplingInputs(max_top_k=1, batch_size=1, logits_processors=[[]], max_num_logprobs=-1)
    block_ids = tuple(range((token_len + 3) // 4 + 1))
    target_rows = torch.full((len(block_ids) * 4, ), -1, dtype=torch.long)
    mtp_rows = torch.full_like(target_rows, -1)

    def prefill(start, end, **chunk_flags):
        inputs = ModelInputs(
            input_ids=torch.arange(start, end).view(1, -1),
            seq_length=torch.tensor([end - start]),
            history_lengths=torch.tensor([start]),
            block_offsets=torch.tensor([block_ids]),
            num_ignored_history=torch.zeros(1, dtype=torch.long),
            is_decoding=False,
            max_q_seqlen=end - start,
            max_kv_seqlen=end,
            sum_kv_seqlen=end,
            **chunk_flags,
        )
        logits = torch.zeros(1, 100)
        logits[0, 99] = 10
        extra = ARSpecExtraInputs(
            target_logits=logits,
            target_hidden_states=inputs.input_ids.float().unsqueeze(-1),
        )
        sampled = asyncio.run(agent._rejection_sampling(inputs, extra, sampling_inputs))
        draft, _ = agent._prepare_inputs_from_main(inputs, sampled)
        draft_start = draft.history_lengths.item()
        draft_end = draft_start + draft.seq_length.item()
        torch.testing.assert_close(
            draft.target_hidden_states.flatten(), torch.arange(draft_start, draft_end).float())
        target_rows[start:end] = inputs.input_ids.flatten()
        mtp_rows[draft_start:draft_end] = draft.input_ids.flatten()
        if not inputs.is_chunk or inputs.is_last_chunk:
            assert sampled.output_token_ids.tolist() == [[99]]
            assert draft.input_ids[0, -1].item() == 99
            # Later draft forwards may populate these rows, but they have not
            # been accepted by the target and must not extend the save range.
            mtp_rows[draft_end:] = -2
        token_lens = task._get_connector_token_lens(inputs)
        assert token_lens == (min(end, draft_end), )
        metadata = scheduler.build_connector_meta(_scheduler_output(
            running=(request, ),
            token_lens=token_lens,
            block_ids=(block_ids, ),
            logical_block_ids=(block_ids, ),
        ))
        return draft, token_lens, metadata.save_requests[0]

    if final_chunk:
        first_draft, first_boundary, first_save = prefill(0, history, is_chunk=True, is_first_chunk=True)
        assert first_boundary == (7, )
        assert first_draft.seq_length.tolist() == [7]
        assert first_save.block_ids == (0, )
        assert mtp_rows[7].item() == -1
        first_saved_mtp_rows = mtp_rows[:4].clone()
    else:
        # Existing target and MTP history, as after a safe prefix load.
        target_rows[:history] = torch.arange(history)
        mtp_rows[:history] = torch.arange(1, history + 1)

    draft, boundary, save = prefill(history, token_len, is_chunk=final_chunk, is_last_chunk=final_chunk)
    assert boundary == (token_len, )
    torch.testing.assert_close(target_rows[:token_len], torch.arange(token_len))
    torch.testing.assert_close(mtp_rows[:token_len], torch.tensor([*range(1, token_len), 99]))
    first_block = 1 if final_chunk else 0
    full_blocks = token_len // 4
    assert save.start_block == first_block
    assert save.block_ids == tuple(range(first_block, full_blocks))
    assert save.block_hashes == build_prefix_block_hashes(request.all_ids, 4)[first_block:full_blocks]
    assert torch.all(mtp_rows[first_block * 4:full_blocks * 4] >= 0)
    if final_chunk:
        assert draft.history_lengths.tolist() == [7]
        assert mtp_rows[7].item() == 8
        torch.testing.assert_close(mtp_rows[:4], first_saved_mtp_rows)
        assert agent._prev_chunk_last == {}
    scheduler.shutdown()


def test_new_request_restarts_save_planning_from_first_block():
    scheduler = MooncakeStoreScheduler(_cache_config('kv_producer'))
    request = _request(range(17))
    output = _connector_step(
        running=(request, ),
        token_lens=(10, ),
        block_ids=((31, 32, 33, 34), ),
        logical_block_ids=((41, 42, 43, 44), ),
    )
    first = scheduler.build_connector_meta(output).save_requests[0]

    output.connector_token_lens = (14, )
    second = scheduler.build_connector_meta(output).save_requests[0]
    assert (first.start_block, second.start_block) == (0, 2)

    scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={first.save_id, second.save_id}))
    assert scheduler.build_connector_meta(output) is None

    scheduler.request_finished(request)
    next_request = _request(range(17), seq_id=18)
    scheduler.on_new_request(next_request)
    output.running = [next_request]
    save = scheduler.build_connector_meta(output).save_requests[0]
    assert save.start_block == 0
    assert save.block_ids == (31, 32, 33)
    scheduler.shutdown()


def test_finished_request_keeps_immutable_save_operation_until_completion():
    scheduler = MooncakeStoreScheduler(_cache_config('kv_producer'))
    request = _request(range(9))
    metadata = scheduler.build_connector_meta(
        _connector_step(
            running=(request, ),
            token_lens=(8, ),
            block_ids=((1, 2, 3), ),
            logical_block_ids=((11, 12, 13), ),
        ))
    save_id = metadata.save_requests[0].save_id

    scheduler.request_finished(request)
    assert scheduler.build_connector_meta(_connector_step()) is not None
    result = scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={save_id}))
    assert result.completed_save_ids == frozenset({save_id})
    assert scheduler.build_connector_meta(_connector_step()) is None
    scheduler.shutdown()


def test_worker_drain_discards_save_ids_whose_outputs_were_dropped():
    scheduler = MooncakeStoreScheduler(_cache_config('kv_producer'))
    request = _request(range(9))
    output = _connector_step(
        running=(request, ),
        token_lens=(8, ),
        block_ids=((1, 2), ),
        logical_block_ids=((11, 12), ),
    )
    metadata = scheduler.build_connector_meta(output)
    assert metadata.save_requests
    assert scheduler.build_connector_meta(_connector_step()) is not None

    scheduler.finish_transfers_after_worker_drain()

    assert scheduler.build_connector_meta(_connector_step()) is None
    assert scheduler.build_connector_meta(output) is None

    next_request = _request(range(9), seq_id=18)
    scheduler.on_new_request(next_request)
    output.running = [next_request]
    save = scheduler.build_connector_meta(output).save_requests[0]
    assert save.start_block == 0
    scheduler.shutdown()


def test_successful_remote_load_is_not_saved_back_and_save_filters_non_text():
    scheduler = MooncakeStoreScheduler(_cache_config())
    request = _request(range(17))
    scheduler.client.lookup = Mock(return_value=12)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
    scheduler.update_state_after_alloc(request, (21, 22, 23), 12)
    scheduler.build_connector_meta(_connector_step())
    result = scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={request.seq_id}))
    assert result.load_results == (KVLoadResult(request.seq_id, True), )

    no_resave = scheduler.build_connector_meta(
        _connector_step(
            running=(request, ),
            token_lens=(13, ),
            block_ids=((21, 22, 23, 24, 25), ),
            logical_block_ids=((31, 32, 33, 34, 35), ),
        ))
    assert no_resave is None

    multimodal = _request(range(17), multimodal=True)
    assert scheduler.build_connector_meta(
        _connector_step(
            running=(multimodal, ),
            token_lens=(16, ),
            block_ids=((1, 2, 3, 4), ),
            logical_block_ids=((11, 12, 13, 14), ),
        )) is None
    scheduler.shutdown()


def test_shorter_remote_prefix_rewinds_future_save_boundary():
    scheduler = MooncakeStoreScheduler(_cache_config())
    request = _request(range(17))
    scheduler.client.lookup = Mock(side_effect=(12, 4))

    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
    scheduler.update_state_after_alloc(request, (21, 22, 23), 12)
    scheduler.build_connector_meta(_connector_step())
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={request.seq_id}))

    assert scheduler.get_num_new_matched_tokens(request, 4) == (0, False)
    metadata = scheduler.build_connector_meta(
        _connector_step(
            running=(request, ),
            token_lens=(13, ),
            block_ids=((21, 22, 23, 24), ),
            logical_block_ids=((31, 32, 33, 34), ),
        ))
    save = metadata.save_requests[0]
    assert save.start_block == 1
    assert save.block_ids == (22, 23)
    scheduler.shutdown()


def test_scheduler_rejects_sliding_window_cache():
    config = _cache_config()
    config.window_size = 16

    with pytest.raises(ValueError, match='sliding-window'):
        MooncakeStoreScheduler(config)
