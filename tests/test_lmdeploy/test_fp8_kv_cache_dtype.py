# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from types import SimpleNamespace

import pytest
import torch
from pydantic_core import ValidationError

from lmdeploy.cli.utils import ArgumentHelper
from lmdeploy.messages import KVCacheDType, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.cache_engine import (
    CacheDesc,
    CacheEngine,
    _describe_kv_cache_dtype,
    _get_fp8_cache_tensor_dtype,
)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('auto', KVCacheDType.AUTO),
        ('0', KVCacheDType.AUTO),
        ('int4', KVCacheDType.INT4),
        ('4', KVCacheDType.INT4),
        ('int8', KVCacheDType.INT8),
        ('8', KVCacheDType.INT8),
        ('fp8', KVCacheDType.FP8),
        ('fp8_e4m3', KVCacheDType.FP8),
        ('16', KVCacheDType.FP8),
        ('fp8_e5m2', KVCacheDType.FP8_E5M2),
        ('17', KVCacheDType.FP8_E5M2),
        ('turbo_quant', KVCacheDType.TURBO_QUANT),
        ('42', KVCacheDType.TURBO_QUANT),
    ],
)
def test_kv_cache_dtype_aliases(value, expected):
    parser = argparse.ArgumentParser()
    ArgumentHelper.kv_cache_dtype(parser)

    assert parser.parse_args(['--kv-cache-dtype', value]).kv_cache_dtype == expected


@pytest.mark.parametrize('value', ['none', 'bad'])
def test_kv_cache_dtype_rejects_invalid_aliases(value):
    parser = argparse.ArgumentParser()
    ArgumentHelper.kv_cache_dtype(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(['--kv-cache-dtype', value])


def test_removed_cli_flag_is_rejected():
    parser = argparse.ArgumentParser()
    ArgumentHelper.kv_cache_dtype(parser)
    old_flag = '--' + 'quant' + '-policy'

    with pytest.raises(SystemExit):
        parser.parse_args([old_flag, 'fp8'])


def test_kv_cache_dtype_uses_configured_default():
    parser = argparse.ArgumentParser()
    ArgumentHelper.kv_cache_dtype(parser, default=KVCacheDType.FP8.value)

    assert parser.parse_args([]).kv_cache_dtype == KVCacheDType.FP8


def test_pytorch_config_accepts_fp8_kv_cache_dtypes():
    config = PytorchEngineConfig(kv_cache_dtype=KVCacheDType.FP8_E5M2)

    assert config.kv_cache_dtype == KVCacheDType.FP8_E5M2


@pytest.mark.parametrize('value', [KVCacheDType.FP8.value, 'fp8', 'fp8_e4m3'])
def test_pytorch_config_normalizes_kv_cache_dtype_value(value):
    config = PytorchEngineConfig(kv_cache_dtype=value)

    assert config.kv_cache_dtype == KVCacheDType.FP8


def test_pytorch_config_rejects_invalid_kv_cache_dtype():
    with pytest.raises(ValueError, match='invalid kv_cache_dtype: 99'):
        PytorchEngineConfig(kv_cache_dtype=99)


def test_pytorch_config_rejects_removed_keyword():
    old_kw = 'quant' + '_policy'
    with pytest.raises(TypeError):
        PytorchEngineConfig(**{old_kw: KVCacheDType.FP8})


@pytest.mark.parametrize('kv_cache_dtype', [KVCacheDType.FP8, KVCacheDType.FP8_E5M2])
def test_turbomind_config_rejects_fp8_kv_cache_dtypes(kv_cache_dtype):
    with pytest.raises(ValidationError, match='invalid kv_cache_dtype'):
        TurbomindEngineConfig(kv_cache_dtype=kv_cache_dtype)


def test_turbomind_config_rejects_removed_keyword():
    old_kw = 'quant' + '_policy'
    with pytest.raises(ValidationError):
        TurbomindEngineConfig(**{old_kw: KVCacheDType.INT8})


def test_fp8_kv_cache_dtype_mapping():
    assert _get_fp8_cache_tensor_dtype(KVCacheDType.FP8) is torch.float8_e4m3fn
    assert _get_fp8_cache_tensor_dtype(KVCacheDType.FP8_E5M2) is torch.float8_e5m2


def test_fp8_kv_cache_log_description():
    assert _describe_kv_cache_dtype(KVCacheDType.FP8) == 'fp8_e4m3 KV cache'
    assert _describe_kv_cache_dtype(KVCacheDType.FP8_E5M2) == 'fp8_e5m2 KV cache'
    assert _describe_kv_cache_dtype(KVCacheDType.AUTO) is None


def test_fp8_quant_cache_descs_are_empty():
    model_config = SimpleNamespace(dtype=torch.float16)
    k_desc = CacheDesc(shape=[4, 16, 2, 128], dtype=torch.float8_e4m3fn)
    v_desc = CacheDesc(shape=[4, 16, 2, 128], dtype=torch.float8_e4m3fn)

    normal_cache_config = CacheConfig(max_batches=1,
                                      block_size=16,
                                      num_cpu_blocks=0,
                                      num_gpu_blocks=1,
                                      kv_cache_dtype=KVCacheDType.FP8)

    assert CacheEngine.get_quant_cache_descs(k_desc, v_desc, model_config, normal_cache_config) == []
