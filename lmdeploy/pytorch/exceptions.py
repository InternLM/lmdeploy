# Copyright (c) OpenMMLab. All rights reserved.
"""Exceptions raised by the PyTorch engine."""


class WeightUpdateError(RuntimeError):
    """Raised when model weights are updated in an unsafe engine state."""
