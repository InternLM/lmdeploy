# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.kernels.cuda import multinomial_sampling

from ..multinomial_sampling import MultinomialSamplingBuilder


class TritonMultinomialSamplingBuilder(MultinomialSamplingBuilder):
    """triton multinomial sampling builder."""

    def build():
        """build."""
        return multinomial_sampling
