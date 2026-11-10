from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


def config(extra, connector='MooncakeStoreConnector'):
    return SimpleNamespace(kv_transfer_config=KVTransferConfig(
        kv_connector=connector, kv_role='kv_both', kv_connector_extra_config=extra), num_gpu_blocks=16384)


def agent_with(extra):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.cache_config = config(extra)
    agent.spec_agent = SimpleNamespace(set_cache_config=Mock())
    agent.memdecode_agent = None
    return agent


@pytest.mark.parametrize('shared', [False, True])
def test_layout_refresh_preserves_committed_runtime_epoch(shared):
    extra = {'rl_weight_versioning': True, 'rl_share_across_engines': shared,
             'weight_version': 'committed', 'cache_prefix': 'run', 'model_name': 'model'}
    agent = agent_with(extra.copy())
    incoming = config({**extra, 'weight_version': 'stale-bootstrap'})
    agent.set_cache_config(incoming)
    assert agent.cache_config is incoming
    assert agent.cache_config.num_gpu_blocks == 16384
    assert incoming.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'committed'


@pytest.mark.parametrize('change', [
    {'cache_prefix': 'new-run'}, {'model_name': 'new-model'},
    {'rl_share_across_engines': False}, {'rl_weight_versioning': False},
])
def test_scope_changes_do_not_inherit_runtime_epoch(change):
    extra = {'rl_weight_versioning': True, 'rl_share_across_engines': True,
             'weight_version': 'old', 'cache_prefix': 'run', 'model_name': 'model'}
    agent = agent_with(extra.copy())
    incoming = config({**extra, **change, 'weight_version': 'new'})
    agent.set_cache_config(incoming)
    assert incoming.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'new'


@pytest.mark.parametrize('old_enabled,new_enabled', [(False, True), (True, False)])
def test_versioning_mode_change_does_not_inherit(old_enabled, new_enabled):
    agent = agent_with({'rl_weight_versioning': old_enabled, 'weight_version': 'old'})
    incoming = config({'rl_weight_versioning': new_enabled, 'weight_version': 'new'})
    agent.set_cache_config(incoming)
    assert incoming.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'new'


def test_no_transfer_config_keeps_normal_layout_refresh():
    agent = agent_with({})
    incoming = SimpleNamespace(kv_transfer_config=None)
    agent.set_cache_config(incoming)
    assert agent.cache_config is incoming


@pytest.mark.parametrize('version', [None, 'step-17'])
def test_resolved_version_survives_refresh_into_actual_worker_key(version, monkeypatch, tmp_path):
    import json

    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.kv_connector.mooncake.store import worker as worker_module
    from lmdeploy.pytorch.kv_connector.mooncake.store.data import build_rl_store_cache_prefix

    path = tmp_path / 'mooncake.json'
    path.write_text(json.dumps({'metadata_server': 'P2PHANDSHAKE', 'master_server_address': '127.0.0.1:50051'}))
    extra = {'rl_weight_versioning': True, 'rl_share_across_engines': True, 'weight_version': 'bootstrap',
             'cache_prefix': 'run', 'model_name': 'model', 'mooncake_config_path': str(path)}
    agent = agent_with(extra.copy())
    agent.kv_connector = None
    agent.dist_config = SimpleNamespace(world_size=1)
    agent.dist_ctx = SimpleNamespace(rank=0, cpu_group=None)
    agent._advance_rl_store_weight_version(version=version)
    committed = agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version']
    incoming = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=4,
                           kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_both',
                                                              kv_connector_extra_config=extra.copy()))
    agent.set_cache_config(incoming)
    store = SimpleNamespace(setup=Mock(return_value=0), close=Mock())
    monkeypatch.setattr(worker_module, '_get_local_hostname', lambda: '127.0.0.1')
    worker = worker_module.MooncakeStoreWorker(incoming, store_factory=lambda: store)
    try:
        assert committed != 'bootstrap'
        assert version is None or committed == version
        assert worker.key_metadata.cache_prefix == build_rl_store_cache_prefix('run', committed)
    finally:
        worker.shutdown()
    store.close.assert_called_once()
