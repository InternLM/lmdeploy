# Copyright (c) OpenMMLab. All rights reserved.


def bootstrap():
    import os
    import sys

    has_turbomind = False
    pwd = os.path.dirname(__file__)
    if os.path.exists(os.path.join(pwd, '..', 'lib')):
        has_turbomind = True
    if os.name == 'nt' and has_turbomind:
        if sys.version_info[:2] >= (3, 8):
            CUDA_PATH = os.getenv('CUDA_PATH')
            assert CUDA_PATH is not None, 'Can not find $env:CUDA_PATH'
            dll_path = os.path.join(CUDA_PATH, 'bin')
            print(f'Add dll path {dll_path}, please note cuda version '
                  'should >= 11.3 when compiled with cuda 11')
            os.add_dll_directory(dll_path)


bootstrap()

from .turbomind import TurboMind  # noqa: E402

__all__ = ['TurboMind']
