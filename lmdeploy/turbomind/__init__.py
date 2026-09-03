# Copyright (c) OpenMMLab. All rights reserved.

import torch  # noqa: F401

_import_error = None

try:
    from . import _turbomind as _tm
except (ImportError, OSError) as error:
    _tm = None
    _import_error = error
else:
    from .turbomind import TurboMind


def is_available() -> bool:
    return _tm is not None
