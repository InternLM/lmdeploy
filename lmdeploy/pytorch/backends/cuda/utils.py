# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache


@lru_cache
def has_tilelang():
    try:
        import tilelang  # noqa: F401
        return True
    except Exception:
        return False
