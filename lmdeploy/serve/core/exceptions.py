# Copyright (c) OpenMMLab. All rights reserved.
"""Exceptions for the serve module."""


class SafeRunException(Exception):
    """Exception raised by safe_run to avoid upper layer handling the original
    exception again.

    This exception wraps the original exception that occurred during safe_run execution.
    """
