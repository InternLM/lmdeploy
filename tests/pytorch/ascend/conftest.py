# Copyright (c) OpenMMLab. All rights reserved.
"""Keep Ascend hardware validation out of the common unit-test collection."""
from pathlib import Path

import pytest

HARDWARE_DIR = Path(__file__).resolve().parent / 'hardware'


def pytest_addoption(parser):
    parser.addoption('--run-ascend-hardware',
                     action='store_true',
                     default=False,
                     help='Collect Ascend hardware validation; requires an initialized A3 environment.')


def pytest_configure(config):
    if config.getoption('--run-ascend-hardware'):
        return
    for arg in config.args:
        path = Path(arg.split('::', 1)[0]).resolve()
        if path == HARDWARE_DIR or HARDWARE_DIR in path.parents:
            raise pytest.UsageError('Ascend hardware tests require --run-ascend-hardware.')


def pytest_ignore_collect(collection_path, config):
    if collection_path == HARDWARE_DIR and not config.getoption('--run-ascend-hardware'):
        return True
