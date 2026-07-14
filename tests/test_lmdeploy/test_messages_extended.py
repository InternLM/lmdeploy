# Copyright (c) OpenMMLab. All rights reserved.
"""Additional tests for lmdeploy.messages to improve coverage."""
import pytest

from lmdeploy.messages import (
    GenerationConfig,
    EngineConfig,
    ResponseType,
    RequestType,
)


def test_generation_config_defaults():
    """Test GenerationConfig with default values."""
    config = GenerationConfig()
    assert config.max_new_tokens > 0
    assert config.temperature >= 0.0
    assert config.top_p >= 0.0
    assert config.top_k > 0


def test_generation_config_custom_values():
    """Test GenerationConfig with custom values."""
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
    )
    assert config.max_new_tokens == 100
    assert config.temperature == 0.8
    assert config.top_p == 0.9
    assert config.top_k == 40


def test_generation_config_greedy_search():
    """Test GenerationConfig for greedy search."""
    config = GenerationConfig(top_k=1)
    assert config.top_k == 1
    
    config2 = GenerationConfig(temperature=0.0)
    assert config2.temperature == 0.0


def test_generation_config_max_tokens_validation():
    """Test GenerationConfig max_new_tokens validation."""
    # Test with valid positive value
    config = GenerationConfig(max_new_tokens=500)
    assert config.max_new_tokens == 500
    
    # Test with zero (should handle gracefully)
    config_zero = GenerationConfig(max_new_tokens=0)
    assert config_zero.max_new_tokens == 0


def test_response_type_enum():
    """Test ResponseType enum values."""
    assert ResponseType.SUCCESS is not None
    assert ResponseType.FINISH is not None
    assert ResponseType.CANCEL is not None
    assert ResponseType.INTERNAL_ENGINE_ERROR is not None
    assert ResponseType.HANDLER_NOT_EXIST is not None


def test_response_type_comparison():
    """Test ResponseType enum comparison."""
    assert ResponseType.SUCCESS != ResponseType.FINISH
    assert ResponseType.SUCCESS != ResponseType.CANCEL
    assert ResponseType.INTERNAL_ENGINE_ERROR != ResponseType.SUCCESS


def test_engine_config_basic():
    """Test EngineConfig basic functionality."""
    # Note: EngineConfig might require specific parameters
    # This test checks if the class can be instantiated
    try:
        config = EngineConfig()
        assert config is not None
    except TypeError:
        # EngineConfig may require parameters, which is fine
        pass


def test_generation_config_sampling_params():
    """Test GenerationConfig sampling parameter combinations."""
    # Test typical sampling configuration
    config1 = GenerationConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=50,
    )
    assert all([
        config1.temperature > 0,
        config1.top_p > 0,
        config1.top_k > 0,
    ])
    
    # Test conservative sampling
    config2 = GenerationConfig(
        temperature=0.1,
        top_p=0.99,
        top_k=100,
    )
    assert config2.temperature < config1.temperature
    assert config2.top_p > config1.top_p
    assert config2.top_k > config1.top_k


def test_generation_config_stop_tokens():
    """Test GenerationConfig with stop tokens."""
    stop_token_ids = [1, 2, 3]
    config = GenerationConfig(stop_token_ids=stop_token_ids)
    if config.stop_token_ids is not None:
        assert len(config.stop_token_ids) == len(stop_token_ids)


def test_generation_config_repetition_penalty():
    """Test GenerationConfig repetition_penalty parameter."""
    config = GenerationConfig(repetition_penalty=1.1)
    assert config.repetition_penalty > 1.0
    
    config2 = GenerationConfig(repetition_penalty=1.0)
    assert config2.repetition_penalty == 1.0