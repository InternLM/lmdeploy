# Copyright (c) OpenMMLab. All rights reserved.
from .base_device_utils import BaseDeviceUtils


class CUDADeviceUtils(BaseDeviceUtils):

    device = 'cuda'

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        return step_context
