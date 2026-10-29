# Copyright (c) OpenMMLab. All rights reserved.
import gc

import pytest


@pytest.fixture
def release_cuda_graph_cycles():
    yield
    # Inference-mode graphs retained by test-model cycles keep PyTorch 2.10's
    # default CUDA generator tensors in inference mode for the next test.
    gc.collect()
