# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from types import SimpleNamespace

import pytest
import torch
from pydantic_core import ValidationError

from lmdeploy.cli.utils import ArgumentHelper
from lmdeploy.messages import PytorchEngineConfig, QuantPolicy, TurbomindEngineConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.cache_engine import CacheDesc, CacheEngine, _get_fp8_cache_dtype


def test_quant_policy_fp8_aliases():
    parser = argparse.ArgumentParser()
    ArgumentHelper.quant_policy(parser)

    assert parser.parse_args(['--quant-policy', 'fp8']).quant_policy == QuantPolicy.FP8
    assert parser.parse_args(['--quant-policy', 'fp8_e4m3']).quant_policy == QuantPolicy.FP8
    assert parser.parse_args(['--quant-policy', 'fp8_e5m2']).quant_policy == QuantPolicy.FP8_E5M2
    assert parser.parse_args(['--quant-policy', 'fp8_per_token_head']).quant_policy == QuantPolicy.FP8_PER_TOKEN_HEAD
    assert parser.parse_args(['--quant-policy',
                              'fp8_e4m3_per_token_head']).quant_policy == QuantPolicy.FP8_PER_TOKEN_HEAD
    assert parser.parse_args(['--quant-policy',
                              'fp8_e5m2_per_token_head']).quant_policy == QuantPolicy.FP8_E5M2_PER_TOKEN_HEAD
    assert parser.parse_args(['--quant-policy', '17']).quant_policy == QuantPolicy.FP8_E5M2


def test_pytorch_config_accepts_fp8_quant_policies():
    config = PytorchEngineConfig(quant_policy=QuantPolicy.FP8_E5M2, calculate_kv_scales=True)

    assert config.quant_policy == QuantPolicy.FP8_E5M2
    assert config.calculate_kv_scales is True
    assert (PytorchEngineConfig(quant_policy=QuantPolicy.FP8_PER_TOKEN_HEAD).quant_policy ==
            QuantPolicy.FP8_PER_TOKEN_HEAD)


def test_turbomind_config_rejects_fp8_e5m2_quant_policy():
    with pytest.raises(ValidationError, match='invalid quant_policy'):
        TurbomindEngineConfig(quant_policy=QuantPolicy.FP8_E5M2)


def test_fp8_kv_cache_dtype_mapping():
    assert _get_fp8_cache_dtype(QuantPolicy.FP8) is torch.float8_e4m3fn
    assert _get_fp8_cache_dtype(QuantPolicy.FP8_E5M2) is torch.float8_e5m2
    assert _get_fp8_cache_dtype(QuantPolicy.FP8_PER_TOKEN_HEAD) is torch.float8_e4m3fn
    assert _get_fp8_cache_dtype(QuantPolicy.FP8_E5M2_PER_TOKEN_HEAD) is torch.float8_e5m2


def test_fp8_quant_cache_desc_split():
    model_config = SimpleNamespace(dtype=torch.float16)
    k_desc = CacheDesc(shape=[4, 16, 2, 128], dtype=torch.float8_e4m3fn)
    v_desc = CacheDesc(shape=[4, 16, 2, 128], dtype=torch.float8_e4m3fn)

    normal_cache_config = CacheConfig(max_batches=1,
                                      block_size=16,
                                      num_cpu_blocks=0,
                                      num_gpu_blocks=1,
                                      quant_policy=QuantPolicy.FP8)
    per_token_cache_config = CacheConfig(max_batches=1,
                                         block_size=16,
                                         num_cpu_blocks=0,
                                         num_gpu_blocks=1,
                                         quant_policy=QuantPolicy.FP8_PER_TOKEN_HEAD)

    assert CacheEngine.get_quant_cache_descs(k_desc, v_desc, model_config, normal_cache_config) == []
    descs = CacheEngine.get_quant_cache_descs(k_desc, v_desc, model_config, per_token_cache_config)
    assert [desc.shape for desc in descs] == [[4, 16, 2, 1], [4, 16, 2, 1]]
