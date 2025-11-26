# Copyright (c) OpenMMLab. All rights reserved.

from .base import SPEC_PROPOSERS
from .deepseek_mtp import DeepseekMTP


@SPEC_PROPOSERS.register_module(name='eagle')
class Eagle(DeepseekMTP):
    """Eagle."""
