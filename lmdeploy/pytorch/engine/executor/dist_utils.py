# Copyright (c) OpenMMLab. All rights reserved.
import os
import socket
from datetime import timedelta

import torch.distributed as dist


def find_available_port(start_port: int = 0) -> int:
    """find available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', start_port))
            port = s.getsockname()[1]
            return port
        except socket.error as e:
            if start_port == 0:
                raise RuntimeError('Failed to find available port.') from e
            return find_available_port(0)


def setup_master_addr(addr: str, port: str):
    """setup master addr."""
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')

    if not isinstance(port, str):
        port = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    logger.info(f'MASTER_ADDR={addr}, MASTER_PORT={port}')


def init_dist_environ(rank: int, world_size: int):
    """init environ."""
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)


def init_process_group(rank: int, world_size: int):
    """init process group."""
    DIST_TIMEOUT = timedelta(days=35600)
    init_dist_environ(rank, world_size)
    os.environ.pop('TORCHELASTIC_USE_AGENT_STORE', None)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=DIST_TIMEOUT)
    assert dist.is_initialized()
