import argparse

import pytest

from lmdeploy.cli.serve import SubCliServe


def proxy_args(**overrides):
    args = {
        'server_name': 'router.local',
        'server_port': 30000,
        'routing_strategy': 'cache_aware',
        'serving_strategy': 'Hybrid',
        'dummy_prefill': False,
        'disable_cache_status': False,
        'migration_protocol': 'RDMA',
        'link_type': 'RoCE',
        'disable_gdr': False,
        'api_keys': None,
        'ssl': False,
        'log_level': 'INFO',
    }
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.mark.parametrize(
    'serving_strategy,expected_pd_disaggregation',
    [('Hybrid', False), ('DistServe', True)])
def test_maps_serving_strategy_to_router(serving_strategy, expected_pd_disaggregation):
    router_args = SubCliServe.build_router_args(proxy_args(serving_strategy=serving_strategy))

    assert router_args.lmdeploy_pd_disaggregation is expected_pd_disaggregation


@pytest.mark.parametrize('legacy_strategy', ['min_expected_latency', 'min_observed_latency'])
def test_maps_legacy_routing_strategy_to_cache_aware(legacy_strategy):
    router_args = SubCliServe.build_router_args(proxy_args(routing_strategy=legacy_strategy))

    assert router_args.policy == 'cache_aware'


def test_maps_router_arguments():
    router_args = SubCliServe.build_router_args(
        proxy_args(
            dummy_prefill=True,
            migration_protocol='NVLINK',
            link_type='IB',
            disable_gdr=True,
            api_keys=['secret'],
            log_level='WARNING',
        ))

    assert router_args.host == 'router.local'
    assert router_args.port == 30000
    assert router_args.lmdeploy_migration_protocol == 'nvlink'
    assert router_args.lmdeploy_rdma_link_type == 'ib'
    assert router_args.lmdeploy_dummy_prefill is True
    assert router_args.lmdeploy_disable_gdr is True
    assert router_args.api_key == 'secret'
    assert router_args.log_level == 'warning'


def test_rejects_multiple_api_keys():
    with pytest.raises(ValueError, match='one API key'):
        SubCliServe.build_router_args(proxy_args(api_keys=['secret-one', 'secret-two']))


def test_rejects_ssl():
    with pytest.raises(ValueError, match='does not currently support TLS'):
        SubCliServe.build_router_args(proxy_args(ssl=True))
