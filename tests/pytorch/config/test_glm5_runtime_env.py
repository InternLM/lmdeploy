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
