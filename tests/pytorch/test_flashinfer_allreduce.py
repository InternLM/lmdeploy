# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.comm import flashinfer_allreduce as module


@pytest.mark.parametrize('shape,dtype', [((16, 4096), torch.bfloat16), ((64, 1152), torch.float16)])
def test_workspace_mismatch_preserves_inputs_and_existing_workspace(monkeypatch, shape, dtype):
    monkeypatch.setattr(module.dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(module.dist, 'get_rank', lambda group: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (9, 0))
    provider = module.FlashInferAllReduce('group')
    provider._comm = Mock()
    provider._comm.allreduce_fusion.side_effect = lambda **kwargs: kwargs['input'] * 2
    matching = torch.ones(64, 1152, dtype=torch.bfloat16)
    assert provider.all_reduce_(matching)
    torch.testing.assert_close(matching, torch.full_like(matching, 2))
    workspace = provider._workspace

    incompatible = torch.ones(shape, dtype=dtype)
    residual = torch.full_like(incompatible, 0.5)
    weight = torch.ones(shape[-1], dtype=dtype)
    assert not provider.all_reduce_(incompatible)
    assert provider.fused_all_reduce_residual_rms_norm(incompatible, residual, weight, 1e-6) is None
    torch.testing.assert_close(incompatible, torch.ones_like(incompatible))
    torch.testing.assert_close(residual, torch.full_like(residual, 0.5))
    provider._comm.allreduce_fusion.assert_called_once()
    assert provider._workspace is workspace
    assert not provider._disabled

    assert provider.all_reduce_(matching)
    torch.testing.assert_close(matching, torch.full_like(matching, 4))
    provider._comm.create_allreduce_fusion_workspace.assert_called_once()
    assert provider._workspace is workspace
    workspace.destroy.assert_not_called()
    provider.close()
    workspace.destroy.assert_called_once()
