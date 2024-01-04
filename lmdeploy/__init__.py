# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.api import client, pipeline, serve
from lmdeploy.messages import EngineGenerationConfig, GenerationConfig
from lmdeploy.tokenizer import Tokenizer

__all__ = [
    'pipeline', 'serve', 'client', 'Tokenizer', 'GenerationConfig',
    'EngineGenerationConfig'
]


def find_win_cuda_path(cuda_ver, start, stop, step):
    import os
    for i in range(start, stop, step):
        env_name = 'CUDA_PATH_V' + cuda_ver.replace('.x', '_' + str(i))
        cuda_path = os.getenv(env_name)
        if cuda_path is not None:
            return cuda_path
    return None


def bootstrap():
    import os
    import sys

    has_turbomind = False
    pwd = os.path.dirname(__file__)
    if os.path.exists(os.path.join(pwd, 'lib')):
        has_turbomind = True
    if os.name == 'nt' and has_turbomind:
        if sys.version_info[:2] >= (3, 8):
            from .version import cuda_ver
            cuda_path = None
            if cuda_ver == '11.x':
                cuda_path = find_win_cuda_path(cuda_ver, 8, 2, -1)
            elif cuda_ver == '12.x':
                cuda_path = find_win_cuda_path(cuda_ver, 1, 9, 1)
            print(f'Find cuda path: {cuda_path}')
            os.add_dll_directory(os.path.join(cuda_path, 'bin'))


bootstrap()
