# Copyright (c) OpenMMLab. All rights reserved.
from .base_device_utils import BaseDeviceUtils


class DIPUDeviceUtils(BaseDeviceUtils):

    device = 'dipu'

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        raise NotImplementedError('`update_step_context` of '
                                  f'<{cls}> not implemented.')
