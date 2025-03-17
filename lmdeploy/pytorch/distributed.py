# Copyright (c) OpenMMLab. All rights reserved.
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from torch import distributed as dist
from torch.distributed import ReduceOp


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    tp: int = 1
    dp: int = 1
    tp_rank: int = 0
    world_cpu_group: dist.ProcessGroup = None
    tp_cpu_group: dist.ProcessGroup = None
    tp_gpu_group: dist.ProcessGroup = None
    dp_cpu_group: dist.ProcessGroup = None
    dp_gpu_group: dist.ProcessGroup = None

    @classmethod
    def get_world_size(cls, tp: int, dp: int):
        return tp * dp

    @classmethod
    def build(cls, rank: int = 0, tp: int = 1, dp: int = 1, ccl_backend: str = 'nccl'):
        """build dist context."""
        from datetime import timedelta
        timeout = timedelta(days=35600)

        world_size = cls.get_world_size(tp, dp)
        if world_size == 1:
            return DistContext()

        assert dist.is_initialized()
        # world(assume world group is gloo)
        world_cpu_group = dist.GroupMember.WORLD

        # tp
        tp_gpu_group = None
        tp_rank = rank % tp
        if tp > 1:
            tp_rank0 = rank // tp
            tp_ranks = list(range(tp_rank0, tp_rank0 + tp))
            tp_gpu_group = dist.new_group(ranks=tp_ranks, timeout=timeout, backend=ccl_backend)

        # dp
        dp_gpu_group = None
        if dp > 1 and rank % tp == 0:
            dp_ranks = list(range(0, world_size, tp))
            dp_gpu_group = dist.new_group(ranks=dp_ranks, timeout=timeout, backend=ccl_backend)

        context = DistContext(
            rank=rank,
            world_size=world_size,
            tp=tp,
            dp=dp,
            tp_rank=tp_rank,
            world_cpu_group=world_cpu_group,
            tp_cpu_group=None,
            tp_gpu_group=tp_gpu_group,
            dp_cpu_group=None,
            dp_gpu_group=dp_gpu_group,
        )
        return context


DefaultContext = DistContext.build()


class DistManager:
    """distributed context manager."""

    def __init__(self):
        self.t_local = threading.local()
        self.t_local.device_context = DefaultContext

    def current_context(self) -> DistContext:
        """get current context."""
        return getattr(self.t_local, 'device_context', DefaultContext)

    def set_context(self, context: DistContext):
        """set current context."""
        self.t_local.device_context = context

    @contextmanager
    def context(self, context: DistContext):
        """context manager."""
        origin_context = self.current_context()
        self.set_context(context)
        yield self
        self.set_context(origin_context)


_DIST_MANAGER: DistManager = None


def get_dist_manager():
    """get device manager."""
    global _DIST_MANAGER
    if _DIST_MANAGER is None:
        _DIST_MANAGER = DistManager()
    return _DIST_MANAGER


def get_world_rank():
    """get distributed world size and rank."""
    ctx = get_dist_manager().current_context()
    world_size = ctx.world_size
    rank = ctx.rank

    return world_size, rank


def _check_group_device(device: str):
    """check group device."""
    assert (device in ['cpu', 'gpu']), ('Expect process group device in ("cpu", "gpu"), '
                                        f'but get {device}.')


def get_process_group(device: str = None):
    """get process group."""
    ctx = get_dist_manager().current_context()
    if device is None:
        return dist.GroupMember.WORLD

    _check_group_device(device)

    if device == 'cpu':
        return ctx.world_cpu_group
    else:
        raise RuntimeError('gpu world group is not supported.')


def get_tp_group(device: str = 'gpu'):
    """get tp group."""
    ctx = get_dist_manager().current_context()

    _check_group_device(device)

    if device == 'cpu':
        return ctx.tp_cpu_group
    else:
        return ctx.tp_gpu_group


def get_dp_group(device: str = 'gpu'):
    """get dp group."""
    ctx = get_dist_manager().current_context()

    _check_group_device(device)

    if device == 'cpu':
        return ctx.dp_cpu_group
    else:
        return ctx.dp_gpu_group


def get_group(group_type: str, device: str):
    """get group."""
    if group_type == 'tp':
        return get_tp_group(device)
    elif group_type == 'dp':
        return get_dp_group(device)
    elif group_type in ['world', 'all']:
        return get_process_group(device)
    else:
        raise RuntimeError(f'Unknown group type: {group_type}')


def all_reduce(tensor, op=ReduceOp.SUM, group='tp', async_op=False):
    """all reduce."""
    if isinstance(group, str):
        group = get_group(group, 'gpu')
    dist.all_reduce(tensor, op, group, async_op)


def broadcast(tensor, src, group='tp', async_op=False):
    """broadcast."""
    if isinstance(group, str):
        group = get_group(group, 'gpu')
    dist.broadcast(tensor, src, group, async_op)
