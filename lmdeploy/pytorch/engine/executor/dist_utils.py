# Copyright (c) OpenMMLab. All rights reserved.
import os
import socket
from datetime import timedelta

import torch.distributed as dist


def find_available_port() -> bool:
    """find available port."""
    port = 29500
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1


def setup_master_addr(addr: str, port: str):
    """setup master addr."""
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')

    if not isinstance(port, str):
        port = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    logger.info(f'MASTER_ADDR={addr}, MASTER_PORT={port}')


def init_dist_environ(rank: int, world_size: int, nproc_per_node: int):
    """init environ."""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank % nproc_per_node)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(nproc_per_node)


def init_process_group(rank: int, world_size: int, nproc_per_node: int):
    """init process group."""
    DIST_TIMEOUT = timedelta(days=35600)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=DIST_TIMEOUT)
    assert dist.is_initialized()
    init_dist_environ(rank, world_size, nproc_per_node)
