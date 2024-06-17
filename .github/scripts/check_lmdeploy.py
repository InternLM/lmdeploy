# Copyright (c) MegFlow. All rights reserved.
import glob
import os

import fire


def check_module_init(root: str):
    """Check if a module has __init__.py file."""
    all_files = glob.glob(os.path.join(root, '**/*'), recursive=True)
    not_exist = []
    for d in all_files:
        if not os.path.isdir(d):
            continue
        if '__pycache__' in d:
            continue
        elif d.startswith('lmdeploy/bin'):
            continue
        elif d.startswith('lmdeploy/lib'):
            continue
        elif d.startswith('lmdeploy/serve/turbomind/triton_models'):
            continue
        elif d.startswith('lmdeploy/serve/turbomind/triton_python_backend'):
            continue
        init_file = os.path.join(d, '__init__.py')
        if not os.path.exists(init_file):
            not_exist.append(init_file)

    assert len(not_exist) == 0, f'Missing files: {not_exist}'


if __name__ == '__main__':
    fire.Fire()
