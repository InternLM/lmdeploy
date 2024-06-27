# Copyright (c) OpenMMLab. All rights reserved.
from .dipu import DIPUDeviceUtils


class ASCENDDeviceUtils(DIPUDeviceUtils):

    device = 'ascend'

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        return DIPUDeviceUtils.update_step_context(step_context)
