# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for lmdeploy.pytorch.engine.config_builder module."""

import pytest

from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
from lmdeploy.pytorch.config import (
    BackendConfig,
    CacheConfig,
    DistConfig,
    MiscConfig,
    SchedulerConfig,
)
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


class TestUpdateEngineConfig:
    """Test update_engine_config method."""

    def test_config_is_deep_copied(self):
        """Test that input config is deep copied to avoid mutation."""
        original = PytorchEngineConfig(max_batch_size=8, device_type='cuda')
        result = ConfigBuilder.update_engine_config(original)
        # Modify result
        result.max_batch_size = 16
        # Original should be unchanged
        assert original.max_batch_size == 8

    def test_dllm_block_length_adjusts_batch_size(self):
        """Test that dllm_block_length adjusts max_batch_size if needed."""
        # Create config where block_length * batch_size > prefill_token_num
        config = PytorchEngineConfig(
            max_batch_size=100,
            dllm_block_length=64,
            max_prefill_token_num=512,  # 100 * 64 = 6400 > 512
            device_type='cuda',
        )
        result = ConfigBuilder.update_engine_config(config)
        # Should adjust: 512 // 64 = 8
        assert result.max_batch_size == 8

    def test_dllm_no_adjustment_when_within_limits(self):
        """Test no adjustment when within limits."""
        config = PytorchEngineConfig(
            max_batch_size=4,
            dllm_block_length=64,
            max_prefill_token_num=512,  # 4 * 64 = 256 < 512
            device_type='cuda',
        )
        result = ConfigBuilder.update_engine_config(config)
        assert result.max_batch_size == 4

    def test_dp_disabled_when_tp_and_ep_are_one(self):
        """Test that dp is disabled when tp=1 and ep=1."""
        config = PytorchEngineConfig(dp=2, tp=1, ep=1, max_batch_size=8, device_type='cuda')
        result = ConfigBuilder.update_engine_config(config)
        assert result.dp == 1
        assert result.dp_rank == 0

    def test_dp_enabled_with_tp(self):
        """Test that dp remains enabled with tp > 1."""
        config = PytorchEngineConfig(dp=2, tp=2, ep=1, max_batch_size=8, device_type='cuda')
        result = ConfigBuilder.update_engine_config(config)
        assert result.dp == 2


class TestBuildSchedulerConfig:
    """Test build_scheduler_config method."""

    def test_basic_scheduler_config(self):
        """Test basic scheduler config creation."""
        engine_config = PytorchEngineConfig(
            max_batch_size=8,
            session_len=2048,
            prefill_interval=16,
            device_type='cuda',
        )
        result = ConfigBuilder.build_scheduler_config(engine_config)
        assert isinstance(result, SchedulerConfig)
        assert result.max_batches == 8
        assert result.max_session_len == 2048
        assert result.prefill_interval == 16


class TestBuildCacheConfig:
    """Test build_cache_config method."""

    def test_basic_cache_config(self):
        """Test basic cache config creation."""
        engine_config = PytorchEngineConfig(
            max_batch_size=8,
            block_size=16,
            kernel_block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
            cache_max_entry_count=0.8,
            max_prefill_token_num=512,
            enable_prefix_caching=False,
            quant_policy=0,
            device_type='cuda',
        )
        result = ConfigBuilder.build_cache_config(engine_config)
        assert isinstance(result, CacheConfig)
        assert result.max_batches == 8
        assert result.block_size == 16
        assert result.num_reserved_gpu_blocks == 1

    def test_cache_config_defaults(self):
        """Test cache config with minimal engine config."""
        engine_config = PytorchEngineConfig(device_type='cuda')
        result = ConfigBuilder.build_cache_config(engine_config)
        assert isinstance(result, CacheConfig)
        # Should have sensible defaults
        assert result.num_reserved_gpu_blocks == 1


class TestBuildBackendConfig:
    """Test build_backend_config method."""

    def test_basic_backend_config(self):
        """Test basic backend config creation."""
        engine_config = PytorchEngineConfig(
            eager_mode=False,
            device_type='cuda',
        )
        result = ConfigBuilder.build_backend_config(engine_config)
        assert isinstance(result, BackendConfig)
        assert result.eager_mode is False
        assert result.device_type == 'cuda'

    def test_eager_mode_backend_config(self):
        """Test backend config with eager mode."""
        engine_config = PytorchEngineConfig(eager_mode=True, device_type='ascend')
        result = ConfigBuilder.build_backend_config(engine_config)
        assert result.eager_mode is True
        assert result.device_type == 'ascend'


class TestBuildDistConfig:
    """Test build_dist_config method."""

    def test_dist_config_creation(self):
        """Test dist config creation from engine config."""
        engine_config = PytorchEngineConfig(
            tp=1,
            ep=1,
            dp=1,
            dp_rank=0,
            device_type='cuda',
        )
        result = ConfigBuilder.build_dist_config(engine_config)
        assert isinstance(result, DistConfig)


class TestBuildMiscConfig:
    """Test build_misc_config method."""

    def test_misc_config_creation(self):
        """Test misc config creation from engine config."""
        engine_config = PytorchEngineConfig(device_type='cuda')
        result = ConfigBuilder.build_misc_config(engine_config)
        assert isinstance(result, MiscConfig)


class TestBuildSpecDecodeConfig:
    """Test build_specdecode_config method."""

    def test_none_speculative_config_returns_none(self):
        """Test that None speculative config returns None."""
        engine_config = PytorchEngineConfig(device_type='cuda')
        cache_config = CacheConfig(
            max_batches=8,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )
        dist_config = DistConfig()
        
        result = ConfigBuilder.build_specdecode_config(
            target_model='test_model',
            speculative_config=None,
            engine_config=engine_config,
            cache_config=cache_config,
            dist_config=dist_config,
        )
        assert result is None


class TestConfigBuilderIntegration:
    """Integration tests for ConfigBuilder workflow."""

    def test_full_config_building_workflow(self):
        """Test building all configs in sequence."""
        engine_config = PytorchEngineConfig(
            max_batch_size=8,
            session_len=2048,
            block_size=16,
            device_type='cuda',
        )

        # Update engine config
        updated_engine = ConfigBuilder.update_engine_config(engine_config)

        # Build all sub-configs
        scheduler = ConfigBuilder.build_scheduler_config(updated_engine)
        cache = ConfigBuilder.build_cache_config(updated_engine)
        backend = ConfigBuilder.build_backend_config(updated_engine)
        dist = ConfigBuilder.build_dist_config(updated_engine)
        misc = ConfigBuilder.build_misc_config(updated_engine)

        # Verify all are created
        assert isinstance(scheduler, SchedulerConfig)
        assert isinstance(cache, CacheConfig)
        assert isinstance(backend, BackendConfig)
        assert isinstance(dist, DistConfig)
        assert isinstance(misc, MiscConfig)

    def test_config_building_preserves_values(self):
        """Test that config building preserves key values."""
        engine_config = PytorchEngineConfig(
            max_batch_size=16,
            session_len=4096,
            block_size=32,
            device_type='cuda',
        )

        updated = ConfigBuilder.update_engine_config(engine_config)
        scheduler = ConfigBuilder.build_scheduler_config(updated)
        cache = ConfigBuilder.build_cache_config(updated)

        assert scheduler.max_batches == 16
        assert scheduler.max_session_len == 4096
        assert cache.block_size == 32
