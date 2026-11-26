# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

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
                                               version=2, cache_schema='model-and-mtp', cache_pools=(pool,))
    remote = endpoint.model_copy(deep=True)
    remote.cache_pools[0].shape = (32, 20, 64)
    validate_pd_peer(endpoint, remote)
    remote.cache_pools[0].group = 'draft'
    with pytest.raises(ValueError, match='layout differs'):
        validate_pd_peer(endpoint, remote)
    remote.version = 1
    with pytest.raises(ValueError, match='version 2'):
        validate_pd_peer(endpoint, remote)


def test_handoff_requires_full_prompt_and_first_token_progress():
    fields = dict(transfer_id='operation', engine_epoch='source', cache_schema='model', prompt_digest='tokens',
                  prompt_token_count=66, valid_token_count=66, first_token_id=10, random_seed=42, random_offset=67,
                  first_token_logprobs=[([-0.2, -1.7], [10, 20])],
                  block_size=64, block_ids=[3, 8],
                  state_ids=[7, 11], auxiliary_id=2)
    handoff = PDHandoff(**fields)
    assert PDHandoff.model_validate_json(handoff.model_dump_json()) == handoff
    assert handoff.prefill_cached_tokens == 0
    with pytest.raises(ValueError, match='cache hits'):
        PDHandoff(**(fields | dict(prefill_cached_tokens=67)))
    with pytest.raises(ValueError, match='complete prompt'):
        PDHandoff(**(fields | dict(valid_token_count=65)))
    with pytest.raises(ValueError, match='exactly'):
        PDHandoff(**(fields | dict(block_ids=[3])))
    assert PDHandoff(**(fields | dict(block_ids=[]))).state_ids == [7, 11]
    for span in [(-1, 4, 'image', 'hash'), (4, 4, 'image', 'hash'), (4, 67, 'image', 'hash'),
                 (0, 4, 'image', '')]:
        with pytest.raises(ValueError, match='multimodal spans'):
            PDHandoff(**(fields | dict(multimodal_spans=[span])))
    with pytest.raises(ValueError, match='continuation positions'):
        PDHandoff(**(fields | dict(mrope_position_delta=-66)))
    with pytest.raises(ValueError):
        PDHandoff(**(fields | dict(version=3)))


def test_engine_info_uses_runtime_attention_tp():
    from lmdeploy.pytorch.disagg.config import MigrationBackend
    from lmdeploy.pytorch.engine.engine import Engine

    engine = SimpleNamespace(
        dist_config=SimpleNamespace(tp=8, attn_tp=2, dp=2, ep=4, dp_rank=1),
        cache_config=SimpleNamespace(block_size=64, num_cpu_blocks=0, num_gpu_blocks=100,
                                     migration_backend=MigrationBackend.Mooncake),
        scheduler=SimpleNamespace(pd_coordinator=SimpleNamespace(epoch='incarnation')))
    info = Engine.get_pd_engine_info(engine)
    assert (info['tp_size'], info['dp_size'], info['ep_size'], info['dp_rank']) == (2, 2, 4, 1)
    assert (info['engine_epoch'], info['protocol_version']) == ('incarnation', 2)
