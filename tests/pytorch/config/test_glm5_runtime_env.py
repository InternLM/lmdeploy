# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest.mock import Mock

import pytest

from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
from lmdeploy.pytorch.config import DistConfig
from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder
from lmdeploy.pytorch.engine.executor import base_worker


@pytest.mark.parametrize('nvls', [None, '0', '1'])
def test_glm5_leaves_nvls_policy_to_runtime(monkeypatch, nvls):
    if nvls is None:
        monkeypatch.delenv('NCCL_NVLS_ENABLE', raising=False)
    else:
        monkeypatch.setenv('NCCL_NVLS_ENABLE', nvls)
    hf_config = Glm5NextConfig(text_config={
        'num_hidden_layers': 4,
        'layer_types': ['linear_attention'] * 3 + ['deepseek_sparse_attention'],
        'linear_num_heads': 64,
        'linear_head_dim': 128,
        'linear_conv_kernel_dim': 4,
        'index_kpool': 4,
    })
    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    config = Glm5NextModelConfigBuilder.build(hf_config, tp=8, device_type='cuda')
    worker = base_worker.WorkerWrapperBase.__new__(base_worker.WorkerWrapperBase)
    worker.model_config = config
    worker.dist_config = DistConfig(tp=8)
    worker.world_size = 8
    worker.device_type = 'cuda'
    seen = []
    monkeypatch.setattr(base_worker, 'init_process_group',
                        lambda rank, size: seen.append(os.environ.get('NCCL_NVLS_ENABLE')))
    monkeypatch.setattr(base_worker, 'get_backend', Mock())
    monkeypatch.setattr(base_worker.DistContext, 'build', Mock())

    worker.init_process_group(rank=0)

    assert seen == [nvls]
    assert os.environ.get('NCCL_NVLS_ENABLE') == nvls


@pytest.mark.parametrize('tp', [1, 4, 8])
@pytest.mark.parametrize('draft', [False, True])
def test_glm5_nope_cache_geometry_for_target_and_mtp(monkeypatch, tp, draft):
    import torch

    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.engine.cache_engine.schema import build_k_cache_desc, build_v_cache_desc

    hf_config = Glm5NextConfig(text_config={
        'num_hidden_layers': 4,
        'layer_types': ['linear_attention'] * 3 + ['deepseek_sparse_attention'],
        'linear_num_heads': 64, 'linear_head_dim': 128, 'linear_conv_kernel_dim': 4,
        'index_kpool': 4, 'kv_lora_rank': 512, 'qk_rope_head_dim': 0,
    })
    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    config = Glm5NextModelConfigBuilder.build(hf_config, tp=tp, device_type='cuda',
                                             num_spec_tokens=5, is_draft_model=draft)
    cache_config = CacheConfig(max_batches=8, block_size=64, num_cpu_blocks=0, num_gpu_blocks=16)
    key = build_k_cache_desc(config, cache_config, world_size=tp)
    value = build_v_cache_desc(config, cache_config, world_size=tp)
    assert key.shape == [64, 1, 512]
    assert key.dtype == torch.bfloat16
    assert key.size == 64 * 512 * 2
    assert value.size == 0
    assert not config.use_mla_fp8_cache
