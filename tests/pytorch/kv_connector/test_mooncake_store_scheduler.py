# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.kv_connector import KVConnectorOutput, KVConnectorResult, KVLoadResult
from lmdeploy.pytorch.kv_connector.mooncake.store import scheduler as scheduler_module
from lmdeploy.pytorch.kv_connector.mooncake.store.data import build_prefix_block_hashes
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler


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
        get_prefix_cache_max_match_step=lambda: (len(token_ids) - 1) // 4 * 4,
    )


def _scheduler_output(
    running=(),
    token_lens=(),
    block_ids=(),
    logical_block_ids=(),
):
    return SimpleNamespace(
        running=list(running),
        connector_token_lens=tuple(token_lens),
        connector_block_ids=tuple(block_ids),
        connector_logical_block_ids=tuple(logical_block_ids),
    )


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
    request.get_prefix_cache_max_match_step = lambda: 12
    assert scheduler.get_num_new_matched_tokens(request, 4) == (8, True)

    assert hash_extensions == [0, 2]
    assert lookup_calls[0][3] is True
    assert lookup_calls[2][2][:2] == lookup_calls[0][2]
    scheduler.shutdown()


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

    metadata = scheduler.build_connector_meta(_scheduler_output())
    assert metadata is not None
    assert len(metadata.load_requests) == 1
    load_request = metadata.load_requests[0]
    assert load_request.request_id == request.seq_id
    assert load_request.block_ids == (31, 32)
    assert scheduler.build_connector_meta(_scheduler_output()).load_requests == ()

    assert scheduler.update_connector_output(
        KVConnectorOutput(invalid_block_ids={32})) == KVConnectorResult()
    result = scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={request.seq_id})
    )
    assert result == KVConnectorResult(
        load_results=(KVLoadResult(request.seq_id, False), ),
    )

    retry_save = scheduler.build_connector_meta(
        _scheduler_output(
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
        _scheduler_output(
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
    poll = scheduler.build_connector_meta(_scheduler_output())
    assert poll is not None
    assert poll.save_requests == ()

    second = scheduler.build_connector_meta(
        _scheduler_output(
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
    assert scheduler.build_connector_meta(_scheduler_output()) is not None

    result = scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={1}))
    assert result.completed_save_ids == frozenset({1})
    assert scheduler.build_connector_meta(_scheduler_output()) is None
    scheduler.shutdown()


def test_new_request_restarts_save_planning_from_first_block():
    scheduler = MooncakeStoreScheduler(_cache_config('kv_producer'))
    request = _request(range(17))
    output = _scheduler_output(
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
        _scheduler_output(
            running=(request, ),
            token_lens=(8, ),
            block_ids=((1, 2, 3), ),
            logical_block_ids=((11, 12, 13), ),
        ))
    save_id = metadata.save_requests[0].save_id

    scheduler.request_finished(request)
    assert scheduler.build_connector_meta(_scheduler_output()) is not None
    result = scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={save_id}))
    assert result.completed_save_ids == frozenset({save_id})
    assert scheduler.build_connector_meta(_scheduler_output()) is None
    scheduler.shutdown()


def test_worker_drain_discards_save_ids_whose_outputs_were_dropped():
    scheduler = MooncakeStoreScheduler(_cache_config('kv_producer'))
    request = _request(range(9))
    output = _scheduler_output(
        running=(request, ),
        token_lens=(8, ),
        block_ids=((1, 2), ),
        logical_block_ids=((11, 12), ),
    )
    metadata = scheduler.build_connector_meta(output)
    assert metadata.save_requests
    assert scheduler.build_connector_meta(_scheduler_output()) is not None

    scheduler.finish_transfers_after_worker_drain()

    assert scheduler.build_connector_meta(_scheduler_output()) is None
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
    scheduler.build_connector_meta(_scheduler_output())
    result = scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={request.seq_id}))
    assert result.load_results == (KVLoadResult(request.seq_id, True), )

    no_resave = scheduler.build_connector_meta(
        _scheduler_output(
            running=(request, ),
            token_lens=(13, ),
            block_ids=((21, 22, 23, 24, 25), ),
            logical_block_ids=((31, 32, 33, 34, 35), ),
        ))
    assert no_resave is None

    multimodal = _request(range(17), multimodal=True)
    assert scheduler.build_connector_meta(
        _scheduler_output(
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
    scheduler.build_connector_meta(_scheduler_output())
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={request.seq_id}))

    assert scheduler.get_num_new_matched_tokens(request, 4) == (0, False)
    metadata = scheduler.build_connector_meta(
        _scheduler_output(
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
