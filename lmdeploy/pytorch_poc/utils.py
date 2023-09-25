# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import inspect
from inspect import Parameter, Signature
from typing import Dict, Sequence

import psutil
import torch


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def bind_sigature(input_names: str, args: Sequence, kwargs: Dict):
    """Bind args and kwargs to given input names."""
    kind = inspect._ParameterKind.POSITIONAL_OR_KEYWORD
    sig = Signature([Parameter(name, kind) for name in input_names])
    bind = sig.bind(*args, **kwargs)
    return bind.arguments
