# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for parallel and distributed inference."""

import functools
import os
import unittest

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
            print(f'rank {get_rank()} received {result}')
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


class SimpleTest(unittest.TestCase):

    @master_only
    def fake_input(self):
        print(f'Evaluate fake input 1 on {get_rank()}')
        return 'master only or none'

    @master_only_and_broadcast_general
    def fake_input21(self):
        print(f'Evaluate fake input 21 (str) on {get_rank()}')
        return 'master only and_broadcast'

    @master_only_and_broadcast_general
    def fake_input22(self):
        print(f'Evaluate fake input 22 (cpu tensor) on {get_rank()}')
        return torch.tensor([6, 66, 666])

    @master_only_and_broadcast_tensor
    def fake_input3(self):
        print(f'Evaluate fake input 3 (gpu tensor) on {get_rank()}')
        return torch.tensor([6, 66, 666]).cuda()

    def test(self):
        torch.distributed.init_process_group(backend='nccl')
        rank = get_rank()
        # unittest will discard --local_rank, thus set manually
        torch.cuda.set_device(rank)

        in1 = self.fake_input()
        in21 = self.fake_input21()
        in22 = self.fake_input22()
        in3 = self.fake_input3(dtype=torch.long, size=(1, 3))

        if rank == 0:
            self.assertEqual(in1, 'master only or none')
        else:
            self.assertEqual(in1, None)

        self.assertEqual(in21, 'master only and_broadcast')
        self.assertTrue(torch.allclose(in22, torch.tensor([6, 66, 666])))
        self.assertFalse(torch.allclose(in3.cpu(), torch.tensor([6, 6, 666])))
        self.assertTrue(torch.allclose(in3.cpu(), torch.tensor([6, 66, 666])))


if __name__ == '__main__':
    unittest.main()
