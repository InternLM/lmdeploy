# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import sys
from types import ModuleType

import pytest

from lmdeploy import KVTransferConfig as PublicKVTransferConfig
from lmdeploy.cli.serve import SubCliServe
from lmdeploy.cli.utils import ArgumentHelper, FlexibleArgumentParser
from lmdeploy.messages import KVTransferConfig, PytorchEngineConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


def _engine_config(**kwargs):
    return PytorchEngineConfig(max_batch_size=1, **kwargs)


@pytest.fixture(scope='module')
def serve_parser():
    SubCliServe.add_parser_api_server()
    return SubCliServe.parser


def test_kv_transfer_config_defaults_disabled():
    engine_config = _engine_config()
    cache_config = ConfigBuilder.build_cache_config(engine_config)

    assert PublicKVTransferConfig is KVTransferConfig
    assert engine_config.kv_transfer_config is None
    assert cache_config.kv_transfer_config is None


def test_kv_transfer_config_normalizes_dict_and_reaches_cache_config():
    raw_config = {
        'kv_connector': 'MooncakeStoreConnector',
        'kv_role': 'kv_both',
        'kv_connector_extra_config': {
            'lookup_async': True,
            'lookup_rpc_port': 12345,
        },
    }

    engine_config = _engine_config(kv_transfer_config=raw_config)
    transfer_config = engine_config.kv_transfer_config

    assert isinstance(transfer_config, KVTransferConfig)
    assert transfer_config.is_kv_transfer_instance
    assert transfer_config.is_kv_producer
    assert transfer_config.is_kv_consumer
    assert transfer_config.kv_connector_extra_config['lookup_async'] is True
    assert transfer_config.kv_connector_extra_config['lookup_rpc_port'] == 12345

    cache_config = ConfigBuilder.build_cache_config(engine_config)
    restored_cache_config = pickle.loads(pickle.dumps(cache_config))
    assert restored_cache_config.kv_transfer_config == transfer_config


def test_kv_transfer_extra_config_default_is_not_shared():
    first = KVTransferConfig()
    second = KVTransferConfig()

    first.kv_connector_extra_config['key'] = 'value'

    assert second.kv_connector_extra_config == {}


@pytest.mark.parametrize(
    ('config', 'error', 'match'),
    [
        ({'kv_connector': 'MooncakeStoreConnector'}, ValueError, 'kv_role must be specified'),
        ({'kv_role': 'kv_consumer'}, ValueError, 'kv_connector must be specified'),
        ({'kv_connector': ' ', 'kv_role': 'kv_both'}, ValueError, 'non-empty string'),
        ({'kv_connector': 'MooncakeStoreConnector', 'kv_role': 'invalid'}, ValueError, 'unsupported kv_role'),
        ({
            'kv_connector': 'MooncakeStoreConnector',
            'kv_role': 'kv_both',
            'kv_connector_extra_config': [],
        }, TypeError, 'must be a dict'),
        ([], TypeError, 'KVTransferConfig, dict, or None'),
        (True, TypeError, 'KVTransferConfig, dict, or None'),
    ],
)
def test_kv_transfer_config_rejects_invalid_values(config, error, match):
    with pytest.raises(error, match=match):
        _engine_config(kv_transfer_config=config)


def test_kv_transfer_config_cli_supports_json_and_dotted_syntax():
    parser = FlexibleArgumentParser()
    ArgumentHelper.kv_transfer_config(parser)

    json_config = parser.parse_args([
        '--kv-transfer-config',
        '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config":{"lookup_async":true,"lookup_rpc_port":12345}}',
    ]).kv_transfer_config
    assert json_config['kv_connector'] == 'MooncakeStoreConnector'
    assert json_config['kv_connector_extra_config']['lookup_async'] is True
    assert json_config['kv_connector_extra_config']['lookup_rpc_port'] == 12345

    dotted_config = parser.parse_args([
        '--kv-transfer-config.kv_connector',
        'MooncakeStoreConnector',
        '--kv-transfer-config.kv_role',
        'kv_both',
        '--kv-transfer-config.kv_connector_extra_config.lookup_async',
        'true',
    ]).kv_transfer_config
    assert dotted_config == {
        'kv_connector': 'MooncakeStoreConnector',
        'kv_role': 'kv_both',
        'kv_connector_extra_config': {
            'lookup_async': True,
        },
    }


def test_api_server_kv_transfer_config_reaches_pytorch_engine(monkeypatch, serve_parser):
    captured = {}
    api_server_module = ModuleType('lmdeploy.serve.openai.api_server')

    def fake_serve(*args, **kwargs):
        captured.update(kwargs)

    api_server_module.serve = fake_serve
    monkeypatch.setitem(sys.modules, 'lmdeploy.serve.openai.api_server', api_server_module)
    monkeypatch.setattr('lmdeploy.cli.serve.get_chat_template', lambda *args, **kwargs: None)
    monkeypatch.setattr('lmdeploy.cli.serve.get_speculative_config', lambda *args, **kwargs: None)

    args = serve_parser.parse_args([
        'api_server',
        'test-model',
        '--backend',
        'pytorch',
        '--max-batch-size',
        '1',
        '--kv-transfer-config',
        '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}',
    ])
    SubCliServe.api_server(args)

    transfer_config = captured['backend_config'].kv_transfer_config
    assert transfer_config == KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_both')


def test_api_server_rejects_kv_transfer_config_for_turbomind(monkeypatch, serve_parser):
    monkeypatch.setattr('lmdeploy.archs.autoget_backend', lambda *args, **kwargs: 'turbomind')
    args = serve_parser.parse_args([
        'api_server',
        'test-model',
        '--backend',
        'turbomind',
        '--max-batch-size',
        '1',
        '--kv-transfer-config',
        '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}',
    ])

    with pytest.raises(ValueError, match='only by the PyTorch engine'):
        SubCliServe.api_server(args)
