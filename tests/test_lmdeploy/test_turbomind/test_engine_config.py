import pytest
from pydantic import ValidationError

from lmdeploy import TurbomindEngineConfig


def test_linear_prefix_cache_interval_blocks_default():
    config = TurbomindEngineConfig(enable_prefix_caching=True)
    assert config.linear_prefix_cache_interval_blocks == 64


def test_linear_prefix_cache_interval_blocks_validation():
    with pytest.raises(ValidationError, match='invalid linear_prefix_cache_interval_blocks'):
        TurbomindEngineConfig(linear_prefix_cache_interval_blocks=0)


def test_linear_prefix_cache_interval_blocks_override():
    config = TurbomindEngineConfig(enable_prefix_caching=True, linear_prefix_cache_interval_blocks=4)
    assert config.linear_prefix_cache_interval_blocks == 4
