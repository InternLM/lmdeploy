# Copyright (c) OpenMMLab. All rights reserved.
import threading


def singleton(cls):
    """Singleton decorator that preserves class type."""
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    class SingletonClass(cls):

        def __new__(cls_, *args, **kwargs):
            nonlocal _instance, _initialized
            with _lock:
                if _instance is None:
                    _instance = super().__new__(cls_)
            return _instance

        def __init__(self, *args, **kwargs):
            nonlocal _initialized
            with _lock:
                if not _initialized:
                    super().__init__(*args, **kwargs)
                    _initialized = True

    # Preserve original class metadata
    SingletonClass.__name__ = cls.__name__
    SingletonClass.__qualname__ = cls.__qualname__
    SingletonClass.__doc__ = cls.__doc__
    SingletonClass.__module__ = cls.__module__

    return SingletonClass
