# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
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


def _request(token_ids, *, adapter_name=None, multimodal=False, embeddings=False):
    return SimpleNamespace(
        seq_id=17,
        adapter_name=adapter_name,
        all_ids=np.asarray(token_ids, dtype=np.int64),
        history_multimodals=SimpleNamespace(empty=lambda: not multimodal),
        history_embeddings=[object()] if embeddings else [],
        get_prefix_cache_max_match_step=lambda: (len(token_ids) - 1) // 4 * 4,
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
    scheduler.client.lookup = Mock(side_effect=AssertionError('lookup must not run'))
    request = _request(range(9), multimodal=multimodal, embeddings=embeddings)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
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

    assert scheduler.request_finished(request, ()) == (False, None)
    assert request.seq_id not in scheduler._request_hash_trackers
    assert scheduler.client.discard.call_args_list == [
        ((request.seq_id, ), {}),
        ((request.seq_id, ), {}),
    ]
    scheduler.shutdown()
