import logging

import pytest

from lmdeploy.turbomind.converter import _deep_merge


@pytest.fixture(autouse=True)
def _caplog_lmdeploy(caplog):
    caplog.set_level(logging.WARNING, logger='lmdeploy')
    logger = logging.getLogger('lmdeploy')
    logger.propagate = True
    yield
    logger.propagate = False


class TestDeepMerge:

    def test_flat_override(self):
        base = {'a': 1, 'b': 2}
        _deep_merge(base, {'b': 99})
        assert base == {'a': 1, 'b': 99}

    def test_nested_override(self):
        base = {'rope_scaling': {'rope_type': 'default', 'factor': 1.0}}
        _deep_merge(base, {'rope_scaling': {'factor': 4.0}})
        assert base == {'rope_scaling': {'rope_type': 'default', 'factor': 4.0}}

    def test_new_key_warns(self, caplog):
        base = {'a': 1}
        _deep_merge(base, {'nonexistent_key': 'val'})
        assert base['nonexistent_key'] == 'val'
        assert 'nonexistent_key' in caplog.text

    def test_nested_new_key_warns(self, caplog):
        base = {'rope_scaling': {'factor': 1.0}}
        _deep_merge(base, {'rope_scaling': {'brand_new': 'yes'}})
        assert base['rope_scaling']['brand_new'] == 'yes'
        assert 'brand_new' in caplog.text

    def test_empty_override_is_noop(self):
        base = {'a': 1}
        _deep_merge(base, {})
        assert base == {'a': 1}

    def test_scalar_overrides_dict(self):
        base = {'a': {'nested': 1}}
        _deep_merge(base, {'a': 'flat'})
        assert base == {'a': 'flat'}
