# Copyright (c) OpenMMLab. All rights reserved.
"""Eager Execution Wrapper for Piecewise Mode.

In piecewise mode, certain operations (like attention) require eager execution. This wrapper temporarily disables graph
mode to execute eager versions of functions.
"""

from typing import Any, Callable

import torch.nn as nn
from torch.profiler import record_function


class EagerExecutionWrapper(nn.Module):
    """Wrapper to force eager execution.

    Features:
    - Wraps operations or modules to always execute in eager mode
    - Temporarily disables graph mode internally even when outer enable_graph_mode=True
    """

    def __init__(self, op_or_module: Callable, op_name: str = 'unknown'):
        super().__init__()
        self.op_or_module = op_or_module
        self.op_name = op_name

    @record_function('piecewise_eager_forward')
    def forward(self, *args, **kwargs) -> Any:
        return self.op_or_module(*args, **kwargs)

    def __repr__(self):
        return f'EagerExecutionWrapper({self.op_name})'
