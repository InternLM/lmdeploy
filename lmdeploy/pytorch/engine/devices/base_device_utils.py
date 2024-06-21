# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC


class BaseDeviceUtils(ABC):

    _sub_classes = dict()
    device = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        BaseDeviceUtils.register_builder(cls)

    @classmethod
    def register_builder(cls, sub_cls):
        """register builder."""
        if sub_cls not in BaseDeviceUtils._sub_classes:
            BaseDeviceUtils._sub_classes[sub_cls.device] = sub_cls

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        raise NotImplementedError('`update_step_context` of '
                                  f'<{cls}> not implemented.')
