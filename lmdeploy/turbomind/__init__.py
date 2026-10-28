# Copyright (c) OpenMMLab. All rights reserved.

import sys

import torch  # noqa: F401

_import_error = None

try:
    from . import _turbomind as _tm
except (ImportError, OSError) as error:
    _tm = None
    _import_error = error
else:
    # Expose the extension under its bare name so submodules that predate the
    # lazy-import design can `import _turbomind` without a second load.
    sys.modules.setdefault('_turbomind', _tm)
    from .turbomind import TurboMind as TurboMind


def is_available() -> bool:
    return _tm is not None
