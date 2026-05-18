# Copyright (c) OpenMMLab. All rights reserved.

import os

import pytest


def test_routing_strategy_values():
    from lmdeploy.serve.proxy.config import RoutingStrategy
    assert RoutingStrategy.RANDOM.value == 'random'
    assert RoutingStrategy.MIN_EXPECTED_LATENCY.value == 'min_expected_latency'
    assert RoutingStrategy.MIN_OBSERVED_LATENCY.value == 'min_observed_latency'
    assert RoutingStrategy.MIN_CACHE_USAGE.value == 'min_cache_usage'


def test_routing_strategy_from_str():
    from lmdeploy.serve.proxy.config import RoutingStrategy
    assert RoutingStrategy.from_str('random') == RoutingStrategy.RANDOM
    assert RoutingStrategy.from_str('min_expected_latency') == RoutingStrategy.MIN_EXPECTED_LATENCY
    assert RoutingStrategy.from_str('min_observed_latency') == RoutingStrategy.MIN_OBSERVED_LATENCY
    assert RoutingStrategy.from_str('min_cache_usage') == RoutingStrategy.MIN_CACHE_USAGE


def test_routing_strategy_from_str_invalid():
    from lmdeploy.serve.proxy.config import RoutingStrategy
    with pytest.raises(ValueError, match='Invalid strategy'):
        RoutingStrategy.from_str('nonexistent')


def test_serving_strategy_values():
    from lmdeploy.serve.proxy.config import ServingStrategy
    assert ServingStrategy.HYBRID.value == 'Hybrid'
    assert ServingStrategy.DIST_SERVE.value == 'DistServe'


def test_proxy_config_defaults():
    from lmdeploy.serve.proxy.config import ProxyConfig
    config = ProxyConfig()
    assert config.server_name == '0.0.0.0'
    assert config.server_port == 8000
    assert config.routing_strategy.value == 'min_expected_latency'
    assert config.serving_strategy.value == 'Hybrid'
    assert config.disable_cache_status is False
    assert config.metrics_poll_interval == 5.0
    assert config.api_keys is None
    assert config.ssl is False


def test_proxy_config_env_override():
    import importlib
    config_module = importlib.import_module('lmdeploy.serve.proxy.config')
    os.environ['LMDEPLOY_PROXY_POLL_METRICS_INTERVAL'] = '10'
    try:
        importlib.reload(config_module)
        config = config_module.ProxyConfig()
        assert config.metrics_poll_interval == 10.0
    finally:
        del os.environ['LMDEPLOY_PROXY_POLL_METRICS_INTERVAL']
        importlib.reload(config_module)


def test_error_codes():
    from lmdeploy.serve.proxy.config import ErrorCodes
    assert ErrorCodes.MODEL_NOT_FOUND.value == 10400
    assert ErrorCodes.SERVICE_UNAVAILABLE.value == 10401
    assert ErrorCodes.API_TIMEOUT.value == 10402


def test_api_server_exception():
    from lmdeploy.serve.proxy.config import APIServerException
    exc = APIServerException(status_code=500, body=b'error')
    assert exc.status_code == 500
    assert exc.body == b'error'
    assert 'content-type' in exc.headers
