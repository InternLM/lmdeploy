# Copyright (c) OpenMMLab. All rights reserved.

from .deepseek_mtp import DeepseekMTP  # noqa F401
from .eagle import Eagle  # noqa F401
from .eagle3 import Eagle3  # noqa F401
from .reject_sampler import RejectionSampler
from .spec_agent import SpecModelAgent

__all__ = ['RejectionSampler', 'SpecModelAgent']
