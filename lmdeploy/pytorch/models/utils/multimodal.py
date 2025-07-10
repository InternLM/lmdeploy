# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from lmdeploy.pytorch.multimodal.data_type import MultiModalInputs

PreparedInputs = Tuple[List[int], MultiModalInputs]


class MultiModalMixin:

    def prepare_multimodal_input(self, input_ids, input_multimodals, **kwargs) -> PreparedInputs:
        """Prepare multimodals inputs."""
        raise NotImplementedError('prepare input not implemented.')
