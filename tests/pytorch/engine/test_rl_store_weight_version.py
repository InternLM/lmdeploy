from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.engine.model_agent import agent as module


def make_agent(extra=None, world_size=1, rank=0):
    agent = module.BaseModelAgent.__new__(module.BaseModelAgent)
    transfer = None if extra is None else KVTransferConfig(
        kv_connector='MooncakeStoreConnector', kv_role='kv_both', kv_connector_extra_config=dict(extra))
    agent.cache_config = SimpleNamespace(kv_transfer_config=transfer)
    agent.kv_connector = None
    agent.dist_config = SimpleNamespace(world_size=world_size)
    agent.dist_ctx = SimpleNamespace(rank=rank, cpu_group=object())
    return agent


@pytest.mark.parametrize('extra', [None, {}, {'rl_weight_versioning': False}])
@pytest.mark.parametrize('version', [None, 'step-17'])
def test_default_keeps_static_store_namespace(extra, version, monkeypatch):
    agent = make_agent(extra)
    broadcast = Mock()
    monkeypatch.setattr(module.dist, 'broadcast_object_list', broadcast)
    monkeypatch.setattr(module.dist, 'all_gather_object', lambda output, value, group: None)
    agent._advance_rl_store_weight_version()
    broadcast.assert_not_called()
    if extra is not None:
        assert 'weight_version' not in agent.cache_config.kv_transfer_config.kv_connector_extra_config


def test_version_changes_each_commit_and_leaves_base_prefix(monkeypatch):
    agent = make_agent({'rl_weight_versioning': True, 'cache_prefix': 'run-a'})
    cfg = agent.cache_config.kv_transfer_config.kv_connector_extra_config
    agent._advance_rl_store_weight_version()
    first = cfg['weight_version']
    agent._advance_rl_store_weight_version()
    assert first != cfg['weight_version'] and cfg['cache_prefix'] == 'run-a'


def test_live_connector_rejected_before_weight_write():
    agent = make_agent({'rl_weight_versioning': True})
    agent.kv_connector = object()
    with pytest.raises(RuntimeError, match='require sleep'):
        agent._check_rl_store_weights_update()
    with pytest.raises(RuntimeError, match='require sleep'):
        agent._load_updated_weight_bucket([], 'test')
    assert 'weight_version' not in agent.cache_config.kv_transfer_config.kv_connector_extra_config


def test_dp_tp_ranks_receive_common_version(monkeypatch):
    agent = make_agent({'rl_weight_versioning': True}, world_size=4, rank=2)
    def broadcast(version, src, group):
        assert src == 0 and group is agent.dist_ctx.cpu_group and version == [None]
        version[0] = 'new-committed-weights'
    monkeypatch.setattr(module.dist, 'broadcast_object_list', broadcast)
    monkeypatch.setattr(module.dist, 'all_gather_object', lambda output, value, group: None)
    agent._advance_rl_store_weight_version()
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'new-committed-weights'


def test_failed_collective_does_not_publish_new_version(monkeypatch):
    agent = make_agent({'rl_weight_versioning': True, 'weight_version': 'old'}, world_size=4)
    monkeypatch.setattr(module.dist, 'all_gather_object', Mock(side_effect=RuntimeError('collective failed')))
    monkeypatch.setattr(module.dist, 'broadcast_object_list', Mock(side_effect=RuntimeError('collective failed')))
    with pytest.raises(RuntimeError, match='collective failed'):
        agent._advance_rl_store_weight_version()
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'old'


def test_sleep_changes_namespace_even_without_new_weights():
    agent = make_agent({'rl_weight_versioning': True, 'weight_version': 'before-sleep'})
    agent._advance_rl_store_weight_version(reason='sleep')
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] != 'before-sleep'


class TestCrossEngineStoreEpoch:
    def test_two_engines_accept_same_epoch_only_after_commit(self):
        agents = [make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': True,
                              'weight_version': f'bootstrap-{i}'}) for i in range(2)]
        for agent in agents:
            agent._advance_rl_store_weight_version(version='a' * 32)
        versions = {a.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] for a in agents}
        assert versions == {'a' * 32}

    def test_sleep_invalidates_and_requires_new_commit(self):
        a = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': True})
        a._advance_rl_store_weight_version(version='a' * 32)
        a._advance_rl_store_weight_version(reason='sleep')
        assert a.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] != 'a' * 32
        a.memdecode_agent = None
        with pytest.raises(RuntimeError, match='successful full IPC'):
            a.wakeup(tags=['kv_cache'])
        a._advance_rl_store_weight_version(version='b' * 32)
        assert a._rl_store_needs_shared_commit is False

    def test_missing_version_uses_internal_version(self):
        a = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': True, 'weight_version': 'private'})
        a._rl_store_needs_shared_commit = True
        a._advance_rl_store_weight_version()
        assert a.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] != 'private'
        assert a._rl_store_needs_shared_commit is False

    def test_engine_disagreement_does_not_publish(self, monkeypatch):
        a = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': True,
                        'weight_version': 'private'}, world_size=4)
        def gather(values, value, group):
            values[-1] = 'b' * 32
        monkeypatch.setattr(module.dist, 'all_gather_object', gather)
        with pytest.raises(ValueError, match='disagree'):
            a._advance_rl_store_weight_version(version='a' * 32)
        assert a.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'private'

    @pytest.mark.parametrize('body', [{}, {'weight_version': None}, {'weight_version': 'step-17'}])
    def test_optional_wire_weight_version(self, body):
        from lmdeploy.serve.openai.protocol import UpdateParamsRequest
        request = UpdateParamsRequest(serialized_named_tensors='empty', finished=True, **body)
        assert request.weight_version == body.get('weight_version')



@pytest.mark.parametrize('values', [
    ['a' * 32, None], [None, 'a' * 32], ['a' * 32, 'b' * 32], ['', ''],
])
@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('shared', [False, True])
def test_mixed_epoch_rejection_collective_order(values, rank, shared, monkeypatch):
    agent = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': shared,
                        'weight_version': 'private'}, world_size=2, rank=rank)
    agent._rl_store_needs_shared_commit = True
    def gather(output, value, group):
        assert value == values[rank] and group is agent.dist_ctx.cpu_group
        output[:] = values
    exchange = Mock(side_effect=gather)
    monkeypatch.setattr(module.dist, 'all_gather_object', exchange)
    message = 'non-empty' if values == ['', ''] else 'disagree'
    with pytest.raises(ValueError, match=message):
        agent._advance_rl_store_weight_version(version=values[rank])
    exchange.assert_called_once()
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'private'
    assert agent._rl_store_needs_shared_commit is True


@pytest.mark.parametrize('rank', [0, 1])
def test_uniform_epoch_collective_commit(rank, monkeypatch):
    agent = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': True,
                        'weight_version': 'private'}, world_size=2, rank=rank)
    agent._rl_store_needs_shared_commit = True
    def gather(output, value, group):
        assert value == 'a' * 32 and group is agent.dist_ctx.cpu_group
        output[:] = ['a' * 32, 'a' * 32]
    exchange = Mock(side_effect=gather)
    monkeypatch.setattr(module.dist, 'all_gather_object', exchange)
    agent._advance_rl_store_weight_version(version='a' * 32)
    exchange.assert_called_once()
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'a' * 32
    assert agent._rl_store_needs_shared_commit is False


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('rank', [0, 1])
def test_all_none_generates_engine_internal_version(shared, rank, monkeypatch):
    agent = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': shared,
                        'weight_version': 'private'}, world_size=2, rank=rank)
    agent._rl_store_needs_shared_commit = True
    gathered = Mock()
    def broadcast(output, src, group):
        assert src == 0 and group is agent.dist_ctx.cpu_group
        output[0] = 'internally-generated'
    monkeypatch.setattr(module.dist, 'all_gather_object', gathered)
    monkeypatch.setattr(module.dist, 'broadcast_object_list', broadcast)
    agent._advance_rl_store_weight_version()
    gathered.assert_called_once()
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'internally-generated'
    assert agent._rl_store_needs_shared_commit is False


def test_external_version_works_without_cross_engine_opt_in():
    agent = make_agent({'rl_weight_versioning': True, 'weight_version': 'old'})
    agent._advance_rl_store_weight_version(version='step-17')
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'step-17'


@pytest.mark.parametrize('finished', [False, True])
@pytest.mark.parametrize('version', [None, 'step-17'])
def test_update_params_commits_request_version_only_after_finalization(finished, version, monkeypatch):
    import base64
    import pickle
    from contextlib import nullcontext

    from lmdeploy.serve.openai.protocol import UpdateParamsRequest

    agent = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': True, 'weight_version': 'private'})
    agent._rl_store_needs_shared_commit = True
    model = object()
    agent.patched_model = SimpleNamespace(get_model=lambda: model)
    agent.spec_agent = SimpleNamespace(get_model=lambda: None)
    agent._split_updated_weights = lambda weights: (weights, [])
    agent.all_context = nullcontext
    agent._update_params_ipc_tensor = object()
    agent._update_params_ipc_event = object()
    finalized = []
    def finalize(target):
        assert target is model
        assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'private'
        finalized.append(target)
    monkeypatch.setattr(module, 'process_weights_after_loading', finalize)
    monkeypatch.setattr(module.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(module.torch.cuda, 'empty_cache', lambda: None)
    payload = base64.b64encode(pickle.dumps([])).decode()
    request = UpdateParamsRequest(serialized_named_tensors=payload, finished=finished, weight_version=version)
    agent.update_params(request)
    actual = agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version']
    if finished:
        assert finalized == [model]
        if version is not None:
            assert actual == version
        else:
            assert actual != 'private'
        assert agent._rl_store_needs_shared_commit is False
        assert agent._update_params_ipc_tensor is None and agent._update_params_ipc_event is None
    else:
        assert not finalized and actual == 'private'
        assert agent._rl_store_needs_shared_commit is True


def test_update_params_failed_finalizer_does_not_commit_version(monkeypatch):
    import base64
    import pickle
    from contextlib import nullcontext

    from lmdeploy.serve.openai.protocol import UpdateParamsRequest

    agent = make_agent({'rl_weight_versioning': True, 'rl_share_across_engines': True, 'weight_version': 'private'})
    agent._rl_store_needs_shared_commit = True
    agent.patched_model = SimpleNamespace(get_model=lambda: object())
    agent.spec_agent = SimpleNamespace(get_model=lambda: None)
    agent._split_updated_weights = lambda weights: (weights, [])
    agent.all_context = nullcontext
    monkeypatch.setattr(module, 'process_weights_after_loading', Mock(side_effect=RuntimeError('finalize failed')))
    request = UpdateParamsRequest(serialized_named_tensors=base64.b64encode(pickle.dumps([])).decode(),
                                  finished=True, weight_version='step-17')
    with pytest.raises(RuntimeError, match='finalize failed'):
        agent.update_params(request)
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['weight_version'] == 'private'
    assert agent._rl_store_needs_shared_commit is True
