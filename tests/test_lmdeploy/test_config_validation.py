"""Tests for engine config validation logic.

These tests verify parameter validation in TurbomindEngineConfig,
PytorchEngineConfig, and related configuration classes.
They do NOT require GPU or model loading - pure logic validation only.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest


class MockQuantPolicy:
    """Mock QuantPolicy for testing without importing the real one."""

    class _QuantPolicy:
        def __init__(self, v):
            self.value = v

        def __str__(self):
            return str(self.value)

    # Simulate QuantPolicy enum values used by TurboMind
    NONE = 0
    INT4 = 4
    INT8 = 8
    FP8 = 16
    FP8_E5M2 = 17


# --- TurbomindEngineConfig validation tests ---

@dataclass
class TurbomindEngineConfig:
    """Simplified dataclass mirroring lmdeploy.messages.TurbomindEngineConfig validation."""
    dtype: str = 'auto'
    tp: int = 1
    cache_max_entry_count: float = 0.8
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    max_prefill_token_num: int = 8192
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    async_: int = 1

    def __post_init__(self):
        assert self.dtype in ['auto', 'float16', 'bfloat16'], \
            f'dtype must be auto/float16/bfloat16, got {self.dtype}'
        assert isinstance(self.tp, int) and self.tp >= 1, \
            'tp must be a positive integer'
        assert self.cache_max_entry_count > 0, \
            'invalid cache_max_entry_count'
        assert self.quant_policy in (0, 4, 8), \
            f'invalid quant_policy for TurboMind: {self.quant_policy}'
        assert self.rope_scaling_factor >= 0, \
            'invalid rope_scaling_factor'
        assert self.max_prefill_token_num >= 0, \
            'invalid max_prefill_token_num'
        assert self.num_tokens_per_iter >= 0, \
            'invalid num_tokens_per_iter'
        assert self.async_ in (0, 1), \
            'async_ must be 0 (disabled) or 1 (enabled)'


class TestTurbomindEngineConfigValidation:

    def test_default_config(self):
        """Default config should pass validation."""
        cfg = TurbomindEngineConfig()
        assert cfg.dtype == 'auto'
        assert cfg.tp == 1

    def test_valid_dtype_values(self):
        """All valid dtype values should pass."""
        for dtype in ['auto', 'float16', 'bfloat16']:
            cfg = TurbomindEngineConfig(dtype=dtype)
            assert cfg.dtype == dtype

    @pytest.mark.parametrize('invalid_dtype', ['int8', 'fp32', '', 'float32'])
    def test_invalid_dtype(self, invalid_dtype):
        """Invalid dtype should raise AssertionError."""
        with pytest.raises(AssertionError, match='dtype must be'):
            TurbomindEngineConfig(dtype=invalid_dtype)

    @pytest.mark.parametrize('tp', [1, 2, 4, 8])
    def test_valid_tp_values(self, tp):
        """Valid tp values should pass."""
        cfg = TurbomindEngineConfig(tp=tp)
        assert cfg.tp == tp

    @pytest.mark.parametrize('invalid_tp', [0, -1, 1.5])
    def test_invalid_tp(self, invalid_tp):
        """Invalid tp should raise AssertionError."""
        with pytest.raises((AssertionError, TypeError)):
            TurbomindEngineConfig(tp=invalid_tp)

    def test_cache_max_entry_count_positive(self):
        """Cache max entry count must be positive."""
        cfg = TurbomindEngineConfig(cache_max_entry_count=0.5)
        assert cfg.cache_max_entry_count == 0.5
        with pytest.raises(AssertionError, match='invalid cache_max'):
            TurbomindEngineConfig(cache_max_entry_count=0)
        with pytest.raises(AssertionError, match='invalid cache_max'):
            TurbomindEngineConfig(cache_max_entry_count=-0.1)

    @pytest.mark.parametrize('qp', [0, 4, 8])
    def test_valid_quant_policy(self, qp):
        """Valid quant policies should pass."""
        cfg = TurbomindEngineConfig(quant_policy=qp)
        assert cfg.quant_policy == qp

    @pytest.mark.parametrize('invalid_qp', [16, 17, -1, 1])
    def test_invalid_quant_policy(self, invalid_qp):
        """TurboMind does not support FP8 quantization."""
        with pytest.raises(AssertionError, match='invalid quant_policy'):
            TurbomindEngineConfig(quant_policy=invalid_qp)

    def test_rope_scaling_factor_non_negative(self):
        """Rope scaling factor must be >= 0."""
        cfg = TurbomindEngineConfig(rope_scaling_factor=0.5)
        assert cfg.rope_scaling_factor == 0.5
        with pytest.raises(AssertionError):
            TurbomindEngineConfig(rope_scaling_factor=-1)

    def test_async_must_be_0_or_1(self):
        """Async must be 0 (disabled) or 1 (enabled)."""
        cfg = TurbomindEngineConfig(async_=0)
        assert cfg.async_ == 0
        cfg = TurbomindEngineConfig(async_=1)
        assert cfg.async_ == 1
        with pytest.raises(AssertionError):
            TurbomindEngineConfig(async_=2)


# --- PytorchEngineConfig validation tests ---

@dataclass
class PytorchEngineConfig:
    """Simplified dataclass mirroring lmdeploy.messages.PytorchEngineConfig validation."""
    dtype: str = 'auto'
    tp: int = 1
    dp: int = 1
    session_len: Optional[int] = None
    max_batch_size: Optional[int] = None
    cache_max_entry_count: float = 0.8
    block_size: int = 64
    quant_policy: int = 0
    max_prefill_token_num: int = 8192
    thread_safe: bool = False
    enable_prefix_caching: bool = False
    eager_mode: bool = False

    def __post_init__(self):
        assert self.dtype in ['auto', 'float16', 'bfloat16', 'int4'], \
            f'dtype must be auto/float16/bfloat16/int4, got {self.dtype}'
        assert isinstance(self.tp, int) and self.tp >= 1, \
            'tp must be a positive integer'
        assert isinstance(self.dp, int) and self.dp >= 1, \
            'dp must be a positive integer'
        assert self.cache_max_entry_count > 0, \
            'invalid cache_max_entry_count'
        assert self.block_size > 0, \
            'block_size must be positive'
        assert self.max_prefill_token_num >= 0, \
            'invalid max_prefill_token_num'


class TestPytorchEngineConfigValidation:

    def test_default_config(self):
        """Default PyTorch config should pass validation."""
        cfg = PytorchEngineConfig()
        assert cfg.dtype == 'auto'
        assert cfg.tp == 1

    def test_valid_dtype_values(self):
        """All valid dtype values should pass."""
        for dtype in ['auto', 'float16', 'bfloat16', 'int4']:
            cfg = PytorchEngineConfig(dtype=dtype)
            assert cfg.dtype == dtype

    @pytest.mark.parametrize('invalid_dtype', ['fp8', '', 'int8'])
    def test_invalid_dtype(self, invalid_dtype):
        """Invalid dtype should raise AssertionError."""
        with pytest.raises(AssertionError, match='dtype must be'):
            PytorchEngineConfig(dtype=invalid_dtype)

    @pytest.mark.parametrize('invalid_tp', [0, -1])
    def test_invalid_tp(self, invalid_tp):
        """Invalid tp should raise AssertionError."""
        with pytest.raises((AssertionError, TypeError)):
            PytorchEngineConfig(tp=invalid_tp)

    def test_cache_max_entry_count_positive(self):
        """Cache max entry count must be positive."""
        cfg = PytorchEngineConfig(cache_max_entry_count=0.5)
        assert cfg.cache_max_entry_count == 0.5
        with pytest.raises(AssertionError):
            PytorchEngineConfig(cache_max_entry_count=-0.1)

    def test_block_size_positive(self):
        """Block size must be positive."""
        cfg = PytorchEngineConfig(block_size=128)
        assert cfg.block_size == 128
        with pytest.raises(AssertionError):
            PytorchEngineConfig(block_size=0)

    def test_dp_default(self):
        """Data parallelism defaults to 1."""
        cfg = PytorchEngineConfig()
        assert cfg.dp == 1


# --- GenerationConfig validation tests ---

@dataclass
class GenerationConfig:
    """Simplified dataclass mirroring lmdeploy.messages.GenerationConfig validation."""
    n: int = 1
    max_new_tokens: int = 512
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50
    min_p: float = 0.0
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    skip_special_tokens: bool = True

    def __post_init__(self):
        assert self.n >= 1, 'n must be >= 1'
        assert self.max_new_tokens >= 0, 'max_new_tokens must be >= 0'
        assert 0 <= self.top_p <= 1.0, 'top_p must be in [0, 1]'
        assert self.top_k >= 0, 'top_k must be >= 0'
        assert 0.0 <= self.min_p <= 1.0, 'min_p must be in [0, 1]'
        assert self.temperature > 0, 'temperature must be > 0'
        assert self.repetition_penalty > 0, 'repetition_penalty must be > 0'


class TestGenerationConfigValidation:

    def test_default_config(self):
        """Default generation config should pass validation."""
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.8

    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_valid_n(self, n):
        """Valid n values should pass."""
        cfg = GenerationConfig(n=n)
        assert cfg.n == n

    def test_invalid_n(self):
        """n must be >= 1."""
        with pytest.raises(AssertionError):
            GenerationConfig(n=0)

    @pytest.mark.parametrize('tokens', [0, 1, 128, 2048])
    def test_valid_max_new_tokens(self, tokens):
        """Valid max_new_tokens values should pass."""
        cfg = GenerationConfig(max_new_tokens=tokens)
        assert cfg.max_new_tokens == tokens

    def test_invalid_max_new_tokens(self):
        """max_new_tokens must be >= 0."""
        with pytest.raises(AssertionError):
            GenerationConfig(max_new_tokens=-1)

    @pytest.mark.parametrize('p', [0.0, 0.5, 1.0])
    def test_valid_top_p(self, p):
        """Valid top_p values should pass."""
        cfg = GenerationConfig(top_p=p)
        assert cfg.top_p == p

    @pytest.mark.parametrize('invalid_p', [-0.1, 1.5])
    def test_invalid_top_p(self, invalid_p):
        """top_p must be in [0, 1]."""
        with pytest.raises(AssertionError):
            GenerationConfig(top_p=invalid_p)

    @pytest.mark.parametrize('temp', [0.1, 1.0, 2.0])
    def test_valid_temperature(self, temp):
        """Valid temperature values should pass."""
        cfg = GenerationConfig(temperature=temp)
        assert cfg.temperature == temp

    def test_invalid_temperature(self):
        """temperature must be > 0."""
        with pytest.raises(AssertionError):
            GenerationConfig(temperature=0)
        with pytest.raises(AssertionError):
            GenerationConfig(temperature=-1)

    @pytest.mark.parametrize('penalty', [0.5, 1.0, 2.0])
    def test_valid_repetition_penalty(self, penalty):
        """Valid repetition_penalty values should pass."""
        cfg = GenerationConfig(repetition_penalty=penalty)
        assert cfg.repetition_penalty == penalty