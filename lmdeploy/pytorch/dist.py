# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for parallel and distributed inference."""

import functools
import os

import torch
from torch.distributed import broadcast, broadcast_object_list, is_initialized


def get_local_rank():
    """Get local rank of current process.

    Assume environment variable ``LOCAL_RANK`` is properly set by some launcher.
    See: https://pytorch.org/docs/stable/elastic/run.html#environment-variables
    """  # noqa: E501

    return int(os.getenv('LOCAL_RANK', '0'))


def get_rank():
    """Get rank of current process.

    Assume environment variable ``RANK`` is properly set by some launcher.
    See: https://pytorch.org/docs/stable/elastic/run.html#environment-variables
    """  # noqa: E501

    return int(os.getenv('RANK', '0'))


def get_world_size():
    """Get rank of current process.

    Assume environment variable ``WORLD_SIZE`` is properly set by some launcher.
    See: https://pytorch.org/docs/stable/elastic/run.html#environment-variables
    """  # noqa: E501

    return int(os.getenv('WORLD_SIZE', '1'))


def master_only(func):
    """Decorator to run a function only on the master process."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_initialized():
            if get_rank() != 0:
                return None
        return func(*args, **kwargs)

    return wrapper


def master_only_and_broadcast_general(func):
    """Decorator to run a function only on the master process and broadcast the
    result to all processes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_initialized():
            if get_rank() == 0:
                result = [func(*args, **kwargs)]
            else:
                result = [None]
            broadcast_object_list(result, src=0)
            result = result[0]
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


def master_only_and_broadcast_tensor(func):
    """Decorator to run a function only on the master process and broadcast the
    result to all processes.

    Note: Require CUDA tensor.
    Note: Not really work because we don't know the shape aforehand,
          for cpu tensors, use master_only_and_broadcast_general
    """

    @functools.wraps(func)
    def wrapper(*args, size, dtype, **kwargs):
        if is_initialized():
            if get_rank() == 0:
                result = func(*args, **kwargs)
            else:
                result = torch.empty(size=size,
                                     dtype=dtype,
                                     device=get_local_rank())
            broadcast(result, src=0)
            # print(f'rank {get_rank()} received {result}')
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper
