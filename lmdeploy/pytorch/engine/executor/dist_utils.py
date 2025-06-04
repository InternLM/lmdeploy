# Copyright (c) OpenMMLab. All rights reserved.
import os
import socket
from datetime import timedelta

import torch.distributed as dist

from lmdeploy.pytorch.backends.selector import get_backend


def find_available_port() -> bool:
    """Find available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port


def setup_master_addr(addr: str, port: str):
    """Setup master addr."""
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')

    if not isinstance(port, str):
        port = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    logger.info(f'MASTER_ADDR={addr}, MASTER_PORT={port}')


def init_dist_environ(rank: int, world_size: int):
    """Init environ."""
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)


def init_process_group(rank: int, world_size: int):
    """Init process group."""
    DIST_TIMEOUT = timedelta(days=35600)
    init_dist_environ(rank, world_size)
    os.environ.pop('TORCHELASTIC_USE_AGENT_STORE', None)

    ccl_backend = get_backend().ccl_backend()
    dist.init_process_group(backend=ccl_backend, rank=rank, world_size=world_size, timeout=DIST_TIMEOUT)
    assert dist.is_initialized()
