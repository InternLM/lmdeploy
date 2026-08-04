# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest

from lmdeploy.cli.serve import SubCliServe, _validate_lora_backend


def test_validate_lora_backend_rejects_turbomind():
    with pytest.raises(ValueError, match='LoRA adapters are only supported by the PyTorch backend'):
        _validate_lora_backend(['adapter=/path/to/lora'], 'turbomind')


def test_validate_lora_backend_allows_pytorch():
    _validate_lora_backend(['adapter=/path/to/lora'], 'pytorch')


def test_api_server_rejects_lora_adapters_after_backend_resolution(monkeypatch):
    import lmdeploy.archs

    monkeypatch.setattr(lmdeploy.archs, 'autoget_backend', lambda *args, **kwargs: 'turbomind')
    args = SimpleNamespace(max_batch_size=1,
                           device='cuda',
                           backend='turbomind',
                           model_path='model',
                           trust_remote_code=False,
                           adapters=['adapter=/path/to/lora'])

    with pytest.raises(ValueError, match='LoRA adapters are only supported by the PyTorch backend'):
        SubCliServe.api_server(args)
