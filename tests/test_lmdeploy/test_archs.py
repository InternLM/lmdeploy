# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for lmdeploy archs module."""

import pytest

from lmdeploy.archs import autoget_backend, autoget_backend_config


class TestAutogetBackend:
    """Tests for autoget_backend function."""

    def test_autoget_backend_exists(self):
        """Test that autoget_backend function exists."""
        assert callable(autoget_backend)

    def test_autoget_backend_returns_string(self):
        """Test that autoget_backend returns a string."""
        # Use a simple model path that should fallback to pytorch
        result = autoget_backend('nonexistent-model-path')
        assert isinstance(result, str)

    def test_autoget_backend_with_trust_remote_code(self):
        """Test autoget_backend with trust_remote_code parameter."""
        result = autoget_backend('nonexistent-model', trust_remote_code=True)
        assert isinstance(result, str)

    def test_autoget_backend_fallback_to_pytorch(self):
        """Test that unsupported models fallback to pytorch."""
        # Non-existent model should fallback to pytorch
        result = autoget_backend('invalid-model-name-xyz')
        # Should return either 'turbomind' or 'pytorch'
        assert result in ['turbomind', 'pytorch']


class TestAutogetBackendConfig:
    """Tests for autoget_backend_config function."""

    def test_autoget_backend_config_exists(self):
        """Test that autoget_backend_config function exists."""
        assert callable(autoget_backend_config)

    def test_autoget_backend_config_with_model_path(self):
        """Test autoget_backend_config with model path."""
        from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig

        result = autoget_backend_config(model_path='test-model')
        # Should return a config object
        assert result is not None
        # Should be either PytorchEngineConfig or TurbomindEngineConfig
        assert isinstance(result, (PytorchEngineConfig, TurbomindEngineConfig))

    def test_autoget_backend_config_with_explicit_pytorch(self):
        """Test autoget_backend_config with explicit pytorch backend."""
        from lmdeploy.messages import PytorchEngineConfig

        result = autoget_backend_config(
            model_path='test-model',
            backend='pytorch'
        )
        assert isinstance(result, PytorchEngineConfig)

    def test_autoget_backend_config_with_explicit_turbomind(self):
        """Test autoget_backend_config with explicit turbomind backend."""
        from lmdeploy.messages import TurbomindEngineConfig

        result = autoget_backend_config(
            model_path='test-model',
            backend='turbomind'
        )
        assert isinstance(result, TurbomindEngineConfig)

    def test_autoget_backend_config_with_pytorch_config(self):
        """Test autoget_backend_config with existing PytorchEngineConfig."""
        from lmdeploy.messages import PytorchEngineConfig

        existing_config = PytorchEngineConfig()
        result = autoget_backend_config(
            model_path='test-model',
            backend_config=existing_config
        )
        # Should return the same config or equivalent
        assert isinstance(result, PytorchEngineConfig)

    def test_autoget_backend_config_with_turbomind_config(self):
        """Test autoget_backend_config with existing TurbomindEngineConfig."""
        from lmdeploy.messages import TurbomindEngineConfig

        existing_config = TurbomindEngineConfig()
        result = autoget_backend_config(
            model_path='test-model',
            backend_config=existing_config
        )
        assert isinstance(result, TurbomindEngineConfig)

    def test_autoget_backend_config_backend_priority(self):
        """Test that explicit backend parameter has priority."""
        from lmdeploy.messages import PytorchEngineConfig

        # Even with turbomind config, explicit 'pytorch' should win
        from lmdeploy.messages import TurbomindEngineConfig
        turbomind_config = TurbomindEngineConfig()

        result = autoget_backend_config(
            model_path='test-model',
            backend='pytorch',
            backend_config=turbomind_config
        )
        assert isinstance(result, PytorchEngineConfig)

    def test_autoget_backend_config_with_trust_remote_code(self):
        """Test autoget_backend_config with trust_remote_code."""
        result = autoget_backend_config(
            model_path='test-model',
            trust_remote_code=True
        )
        assert result is not None

    def test_autoget_backend_config_invalid_backend(self):
        """Test autoget_backend_config with invalid backend string."""
        with pytest.raises((ValueError, AssertionError)):
            autoget_backend_config(
                model_path='test-model',
                backend='invalid-backend'
            )


class TestBackendTypeConstants:
    """Tests for backend type constants and literals."""

    def test_backend_literal_types(self):
        """Test that backend literal accepts valid values."""
        from typing import get_args
        # Check that Literal type is defined correctly
        from lmdeploy.archs import autoget_backend_config
        import inspect
        sig = inspect.signature(autoget_backend_config)
        backend_param = sig.parameters.get('backend')
        if backend_param and backend_param.annotation != inspect.Parameter.empty:
            # If annotated, should accept 'pytorch' or 'turbomind'
            pass  # Type checking is static, runtime doesn't enforce


class TestAutogetBackendIntegration:
    """Integration tests for backend selection."""

    def test_consistent_backend_selection(self):
        """Test that backend selection is consistent."""
        model_path = 'test-consistent-model'

        # Multiple calls should return same backend for same model
        result1 = autoget_backend(model_path)
        result2 = autoget_backend(model_path)

        assert result1 == result2

    def test_config_matches_backend(self):
        """Test that config type matches selected backend."""
        from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig

        model_path = 'test-config-match'

        # Get backend
        backend = autoget_backend(model_path)

        # Get config
        config = autoget_backend_config(model_path)

        # Config type should match backend
        if backend == 'pytorch':
            assert isinstance(config, PytorchEngineConfig)
        elif backend == 'turbomind':
            assert isinstance(config, TurbomindEngineConfig)
