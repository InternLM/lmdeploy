# Copyright (c) OpenMMLab. All rights reserved.
def bootstrap():
    import os
    import sys

    has_turbomind = False
    pwd = os.path.dirname(__file__)
    if os.path.exists(os.path.join(pwd, 'lib')):
        has_turbomind = True
    if os.name == 'nt' and has_turbomind:
        if sys.version_info[:2] >= (3, 8):
            CUDA_PATH = os.getenv('CUDA_PATH')
            os.add_dll_directory(os.path.join(CUDA_PATH, 'bin'))


bootstrap()
