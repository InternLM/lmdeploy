# Copyright (c) OpenMMLab. All rights reserved.


def bootstrap():
    import os

    has_turbomind = False
    pwd = os.path.dirname(__file__)
    if os.path.exists(os.path.join(pwd, '..', 'lib')):
        has_turbomind = True
    if os.name == 'nt' and has_turbomind:
        dll_paths = []
        cuda_path = os.getenv('CUDA_PATH')
        if cuda_path is not None:
            dll_paths.append(os.path.join(cuda_path, 'bin'))
        try:
            import torch
            dll_paths.append(os.path.join(os.path.dirname(torch.__file__), 'lib'))
        except ImportError:
            pass
        conda_prefix = os.getenv('CONDA_PREFIX')
        if conda_prefix is not None:
            dll_paths.append(os.path.join(conda_prefix, 'Library', 'bin'))
        added = False
        for dll_path in dll_paths:
            if os.path.isdir(dll_path):
                print(f'Add dll path {dll_path}, please note cuda version '
                      'should >= 11.3 when compiled with cuda 11')
                os.add_dll_directory(dll_path)
                added = True
        if not added:
            print('Warning: no CUDA DLL directory found. Set CUDA_PATH or '
                  'install PyTorch with CUDA support, otherwise TurboMind '
                  'may fail to load.')


bootstrap()

from .turbomind import TurboMind, update_parallel_config  # noqa: E402

__all__ = ['TurboMind', 'update_parallel_config']
