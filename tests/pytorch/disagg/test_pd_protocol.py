# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeCachePoolInfo,
    DistServeKVTransferEndpointInfo,
    MigrationProtocol,
    PDHandoff,
)
from lmdeploy.pytorch.disagg.schema import validate_pd_peer


def test_peer_layout_ignores_capacity_but_requires_semantics():
    pool = DistServeCachePoolInfo(shape=(32, 8, 64), dtype='torch.uint8', element_size=1, entry_axis=1,
                                  group='state')
    endpoint = DistServeKVTransferEndpointInfo(protocol=MigrationProtocol.RDMA, endpoint_info='{}',
                                               version=2, schema='model-and-mtp', cache_pools=(pool,))
    remote = endpoint.model_copy(deep=True)
    remote.cache_pools[0].shape = (32, 20, 64)
    validate_pd_peer(endpoint, remote)
    remote.cache_pools[0].group = 'draft'
    with pytest.raises(ValueError, match='layout differs'):
        validate_pd_peer(endpoint, remote)
    remote.version = 1
    with pytest.raises(ValueError, match='version 2'):
        validate_pd_peer(endpoint, remote)


def test_handoff_requires_exact_prefix_and_real_replay_tail():
    fields = dict(transfer_id='operation', engine_epoch='source', schema='model', prompt_digest='tokens',
                  prompt_token_count=66, valid_token_count=65, block_size=64, block_ids=[3, 8],
                  state_ids=[7, 11], auxiliary_id=2)
    handoff = PDHandoff(**fields)
    assert PDHandoff.model_validate_json(handoff.model_dump_json()) == handoff
    with pytest.raises(ValueError, match='prompt tail'):
        PDHandoff(**(fields | dict(valid_token_count=66)))
    with pytest.raises(ValueError, match='exactly'):
        PDHandoff(**(fields | dict(block_ids=[3])))
    assert PDHandoff(**(fields | dict(block_ids=[]))).state_ids == [7, 11]
