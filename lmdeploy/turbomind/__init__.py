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
        # de-duplicate while preserving order; normalize Windows paths so
        # case and slash differences do not bypass de-duplication
        seen = set()
        deduped_dll_paths = []
        for dll_path in dll_paths:
            dll_path_key = os.path.normcase(os.path.normpath(dll_path))
            if dll_path_key in seen:
                continue
            seen.add(dll_path_key)
            deduped_dll_paths.append(dll_path)
        dll_paths = deduped_dll_paths
        _dll_dirs = []  # keep directory handles alive
        for dll_path in dll_paths:
            if os.path.isdir(dll_path):
                _dll_dirs.append(os.add_dll_directory(dll_path))
        if not _dll_dirs:
            print('Warning: no CUDA DLL directory found. Set CUDA_PATH or '
                  'install PyTorch with CUDA support, otherwise TurboMind '
                  'may fail to load.')


bootstrap()

from .turbomind import TurboMind, update_parallel_config  # noqa: E402

__all__ = ['TurboMind', 'update_parallel_config']
