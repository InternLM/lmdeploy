# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseChecker  # noqa: F401


def check_env_deeplink(device_type: str):
    """check Deeplink environment."""
    from .deeplink import DeeplinkChecker
    checker = DeeplinkChecker(device_type)
    checker.handle()


def try_import_deeplink(device_type: str):
    """check Deeplink environment."""
    check_env_deeplink(device_type)
