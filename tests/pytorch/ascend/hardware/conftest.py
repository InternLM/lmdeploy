# Copyright (c) OpenMMLab. All rights reserved.
import pytest


@pytest.fixture(scope='session', autouse=True)
def ascend_runtime():
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.fail('Hardware validation requires an available Ascend NPU.')
    from lmdeploy.pytorch.backends.ascend.runtime import initialize

    initialize()
