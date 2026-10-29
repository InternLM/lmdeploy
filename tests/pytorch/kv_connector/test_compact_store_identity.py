# Copyright (c) OpenMMLab. All rights reserved.
"""Worker-derived joint KV/state identity, independent of
capacity/addresses."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SSMStateCapacity, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.layout import PackedBlockCacheLayout, RowBlockCacheLayout
from lmdeploy.pytorch.engine.cache_engine.plan import BlockCachePlan
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec
from lmdeploy.pytorch.engine.cache_engine.state import SSMStatePlan
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.kv_connector.mooncake.store.data import (
    MooncakeStoreKeyMetadata,
    build_compact_store_schema,
    build_store_key,
)


def _plan(*, dtype=torch.float32, shape=(4,), names=('key', 'value'), alignment=256, layers=2, ratio=1):
    specs = tuple(CacheTensorSpec(name, CacheDesc(shape=list(shape), dtype=dtype, alignment=alignment))
                  for name in names)
    return BlockCachePlan(specs, PackedBlockCacheLayout(specs, layers), ratio)


def _state():
    return SSMStatePlan([StateCacheSpec('rec', (2, 3), torch.float32),
                             StateCacheSpec('conv', (4, 3), torch.bfloat16)], SSMStateCapacity(3, 1, 2, 1))


def _schema(plan=None, **kwargs):
    options = dict(block_plans=[_plan() if plan is None else plan], spec_identity='ar',
                   tp_size=1, block_size=64, quant_policy=0)
    options.update(kwargs)
    return build_compact_store_schema(_state().unit_specs, **options)


@pytest.mark.parametrize('change', ['dtype', 'shape', 'order', 'alignment', 'layers', 'ratio',
                                   'draft', 'spec', 'tp', 'block', 'quant'])
def test_equal_byte_or_semantically_different_contracts_never_share_either_key_group(change):
    plan = _plan()
    options = {}
    if change == 'dtype':
        plan = _plan(dtype=torch.int32)
    elif change == 'shape':
        plan = _plan(shape=(2, 2))
    elif change == 'order':
        plan = _plan(names=('value', 'key'))
    elif change == 'alignment':
        plan = _plan(alignment=128)
    elif change == 'layers':
        plan = _plan(layers=3)
    elif change == 'ratio':
        plan = _plan(ratio=2)
    else:
        options = {'draft': dict(block_plans=[plan, _plan()]),
                   'spec': dict(spec_identity='mtp3'),
                   'tp': dict(tp_size=2), 'block': dict(block_size=128), 'quant': dict(quant_policy=1)}[change]
    a, b = _schema(), _schema(plan, **options)
    assert a != b
    metadata = MooncakeStoreKeyMetadata('same-serving-alias', '', 1, 64, state_schema=a)
    for group in (0, 1):
        assert build_store_key(metadata, 0, bytes(32), group_id=group) != build_store_key(
            replace(metadata, state_schema=b), 0, bytes(32), group_id=group)


def test_worker_fingerprint_uses_meta_plan_not_live_capacity_or_pointer(monkeypatch):
    plan = _plan()
    original = BlockCachePlan.allocate
    calls = []

    def checked(self, num_logical_blocks, device):
        calls.append((num_logical_blocks, device))
        assert (num_logical_blocks, device) == (1, 'meta')
        return original(self, num_logical_blocks, device)

    monkeypatch.setattr(BlockCachePlan, 'allocate', checked)
    assert _schema(plan) == _schema(_plan())
    assert calls == [(1, 'meta')] * 2


def test_row_consumer_mapping_and_packed_physical_layout_are_identity():
    specs = tuple(replace(s, consumer_rows=(0, 1)) for s in _plan().tensor_specs)
    a = BlockCachePlan(specs, RowBlockCacheLayout(specs), 1)
    b_specs = tuple(replace(s, consumer_rows=(1, 0)) for s in specs)
    b = BlockCachePlan(b_specs, RowBlockCacheLayout(b_specs), 1)
    assert _schema(a) != _schema(b)
    assert _schema(a) != _schema()  # same fields/bytes, different owning pools


def _agent(mtp=False):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=4,
        compact_state_capacity=_state().capacity,
        kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_both',
            kv_connector_extra_config=dict(model_name='mutable-alias', cache_prefix='weights-v1')))
    agent.model_config = SimpleNamespace(hf_config=SimpleNamespace(to_dict=lambda: {'architecture': 'target'}),
                                        dtype=torch.bfloat16)
    agent.block_cache_plan = _plan()
    agent.state_cache_engine = SimpleNamespace(compact_plan=_state())
    spec = SimpleNamespace(method='qwen3_5_mtp', num_speculative_tokens=3, model_config=agent.model_config)
    agent.spec_agent = SimpleNamespace(specdecode_config=spec if mtp else None,
                                       cache_engine=SimpleNamespace(block_cache_plan=_plan()) if mtp else None)
    agent.dist_config = SimpleNamespace(attn_tp=1)
    return agent


@pytest.mark.parametrize('mtp', [False, True])
def test_agent_rebuild_identity_is_idempotent_and_capacity_independent(mtp):
    agent = _agent(mtp)
    first = agent._get_compact_store_schema()
    agent.cache_config.num_gpu_blocks = 700
    agent.state_cache_engine.compact_plan = _state()
    assert agent._get_compact_store_schema() == first
    extra = agent.cache_config.kv_transfer_config.kv_connector_extra_config
    assert extra['model_name'] == 'mutable-alias'
    # Namespace versioning is outside the wire-format fingerprint.
    metadata = MooncakeStoreKeyMetadata(extra['model_name'], extra['cache_prefix'], 1, 64, state_schema=first)
    for group in (0, 1):
        key = build_store_key(metadata, 0, bytes(32), group_id=group)
        assert key != build_store_key(replace(metadata, cache_prefix='weights-v2'), 0, bytes(32), group_id=group)
        assert key != build_store_key(replace(metadata, state_schema=None), 0, bytes(32), group_id=group)
    extra['cache_prefix'] = 'weights-v2'
    assert agent._get_compact_store_schema() == first


def test_legacy_and_no_store_agent_do_not_require_identity_or_retained_plans():
    agent = _agent()
    agent.cache_config.compact_state_capacity = None
    del agent.state_cache_engine
    assert agent._get_compact_store_schema() is None
    agent.cache_config.compact_state_capacity = _state().capacity
    agent.cache_config.kv_transfer_config = None
    assert agent._get_compact_store_schema() is None
