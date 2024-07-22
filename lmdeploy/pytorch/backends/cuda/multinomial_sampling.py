# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.kernels.cuda import multinomial_sampling

from ..multinomial_sampling import MultinomialSamplingBuilder


class TritonMultinomialSamplingBuilder(MultinomialSamplingBuilder):

    def build():
        return multinomial_sampling
