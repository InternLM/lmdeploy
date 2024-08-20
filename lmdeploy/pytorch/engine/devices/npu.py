# Copyright (c) OpenMMLab. All rights reserved.
from .ascend import ASCENDDeviceUtils
from .base_device_utils import BaseDeviceUtils


class NPUDeviceUtils(BaseDeviceUtils):

    device = 'npu'

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        return ASCENDDeviceUtils.update_step_context(step_context)
