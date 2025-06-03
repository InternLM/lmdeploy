# Copyright (c) OpenMMLab. All rights reserved.
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

from torch import distributed as dist
from torch.distributed import ReduceOp

from .config import DistConfig


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    tp: int = 1
    dp: int = 1
    ep: int = 1
    tp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0
    world_cpu_group: dist.ProcessGroup = None
    tp_cpu_group: dist.ProcessGroup = None
    tp_gpu_group: dist.ProcessGroup = None
    tp_gpu_groups: List[dist.ProcessGroup] = None
    dp_cpu_group: dist.ProcessGroup = None
    dp_gpu_group: dist.ProcessGroup = None
    ep_gpu_group: dist.ProcessGroup = None
    ep_gpu_groups: List[dist.ProcessGroup] = None
    dist_config: DistConfig = None

    @classmethod
    def build(cls, rank: int = 0, dist_config: DistConfig = None, ccl_backend: str = 'nccl'):
        """Build dist context."""
        from datetime import timedelta
        timeout = timedelta(days=35600)
        cpu_backend = 'gloo'

        if dist_config is None:
            dist_config = DistConfig()
        tp = dist_config.tp
        dp = dist_config.dp
        ep = dist_config.ep
        world_size = dist_config.world_size
        dp_rank = dist_config.dp_rank

        if world_size == 1:
            return DistContext(dist_config=dist_config)

        assert dist.is_initialized()
        # world(assume world group is gloo)
        world_cpu_group = dist.GroupMember.WORLD

        tp_rank = rank % tp

        # tp
        tp_gpu_group = None
        tp_gpu_groups = None
        tp_cpu_group = None
        tp_group_id = dp_rank // tp
        if tp > 1:
            # all tp groups should be created in all procs
            ranks = range(world_size)
            tp_gpu_groups = []
            tp_cpu_groups = []
            for start in range(0, world_size, tp):
                tp_ranks = ranks[start:start + tp]
                group = dist.new_group(ranks=tp_ranks, timeout=timeout, backend=ccl_backend)
                tp_gpu_groups.append(group)
                cpu_group = dist.new_group(ranks=tp_ranks, timeout=timeout, backend=cpu_backend)
                tp_cpu_groups.append(cpu_group)
            tp_gpu_group = tp_gpu_groups[tp_group_id]
            tp_cpu_group = tp_cpu_groups[tp_group_id]

        ep_rank = rank % ep
        ep_gpu_group = None
        ep_gpu_groups = None
        ep_group_id = dp_rank // ep
        if ep > 1:
            ranks = range(world_size)
            ep_gpu_groups = []
            for start in range(0, world_size, ep):
                ep_ranks = ranks[start:start + ep]
                group = dist.new_group(ranks=ep_ranks, timeout=timeout, backend=ccl_backend)
                ep_gpu_groups.append(group)
            ep_gpu_group = ep_gpu_groups[ep_group_id]

        dp_cpu_group = None
        if dp > 1:
            dp_cpu_group = dist.new_group(ranks=range(dp), timeout=timeout, backend=cpu_backend)

        context = DistContext(
            rank=rank,
            world_size=world_size,
            tp=tp,
            dp=dp,
            ep=ep,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            ep_rank=ep_rank,
            world_cpu_group=world_cpu_group,
            tp_cpu_group=tp_cpu_group,
            tp_gpu_group=tp_gpu_group,
            tp_gpu_groups=tp_gpu_groups,
            dp_cpu_group=dp_cpu_group,
            dp_gpu_group=None,
            ep_gpu_group=ep_gpu_group,
            ep_gpu_groups=ep_gpu_groups,
            dist_config=dist_config,
        )
        return context

    def close(self):
        """Close groups."""
        if not dist.is_initialized():
            return
        if self.tp_gpu_groups is not None:
            for group in self.tp_gpu_groups:
                dist.destroy_process_group(group)
        if self.ep_gpu_groups is not None:
            for group in self.ep_gpu_groups:
                dist.destroy_process_group(group)


DefaultContext = DistContext.build()


class DistManager:
    """Distributed context manager."""

    def __init__(self):
        self.t_local = threading.local()
        self.t_local.device_context = DefaultContext

    def current_context(self) -> DistContext:
        """Get current context."""
        return getattr(self.t_local, 'device_context', DefaultContext)

    def set_context(self, context: DistContext):
        """Set current context."""
        self.t_local.device_context = context

    @contextmanager
    def context(self, context: DistContext):
        """Context manager."""
        origin_context = self.current_context()
        self.set_context(context)
        yield self
        self.set_context(origin_context)


_DIST_MANAGER: DistManager = None


def get_dist_manager():
    """Get device manager."""
    global _DIST_MANAGER
    if _DIST_MANAGER is None:
        _DIST_MANAGER = DistManager()
    return _DIST_MANAGER


def get_world_rank():
    """Get distributed world size and rank."""
    ctx = get_dist_manager().current_context()
    world_size = ctx.world_size
    rank = ctx.rank

    return world_size, rank


def get_tp_world_rank():
    ctx = get_dist_manager().current_context()
    return ctx.tp, ctx.tp_rank


def get_dp_world_rank():
    ctx = get_dist_manager().current_context()
    return ctx.dp, ctx.dp_rank


def get_ep_world_rank():
    ctx = get_dist_manager().current_context()
    return ctx.ep, ctx.ep_rank


def _check_group_device(device: str):
    """Check group device."""
    assert (device in ['cpu', 'gpu']), ('Expect process group device in ("cpu", "gpu"), '
                                        f'but get {device}.')


def get_process_group(device: str = None):
    """Get process group."""
    ctx = get_dist_manager().current_context()
    if device is None:
        return dist.GroupMember.WORLD

    _check_group_device(device)

    if device == 'cpu':
        return ctx.world_cpu_group
    else:
        raise RuntimeError('gpu world group is not supported.')


def get_tp_group(device: str = 'gpu'):
    """Get tp group."""
    ctx = get_dist_manager().current_context()

    _check_group_device(device)

    if device == 'cpu':
        return ctx.tp_cpu_group
    else:
        return ctx.tp_gpu_group


def get_dp_group(device: str = 'gpu'):
    """Get dp group."""
    ctx = get_dist_manager().current_context()

    _check_group_device(device)

    if device == 'cpu':
        return ctx.dp_cpu_group
    else:
        return ctx.dp_gpu_group


def get_group(group_type: str, device: str):
    """Get group."""
    if group_type == 'tp':
        return get_tp_group(device)
    elif group_type == 'dp':
        return get_dp_group(device)
    elif group_type in ['world', 'all']:
        return get_process_group(device)
    else:
        raise RuntimeError(f'Unknown group type: {group_type}')


def all_reduce(tensor, op=ReduceOp.SUM, group='tp', async_op=False):
    """All reduce."""
    if isinstance(group, str):
        group = get_group(group, 'gpu')
    return dist.all_reduce(tensor, op, group, async_op)


def broadcast(tensor, src, group='tp', async_op=False):
    """broadcast."""
    if isinstance(group, str):
        group = get_group(group, 'gpu')
    return dist.broadcast(tensor, src, group, async_op)


def all_gather_object(object_list, obj, group='tp'):
    if isinstance(group, str):
        group = get_group(group, 'cpu')
    return dist.all_gather_object(object_list, obj, group=group)


def all_gather(tensor_list, tensor, group='tp', async_op=False):
    if isinstance(group, str):
        group = get_group(group, 'gpu')
    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)


def all_gather_into_tensor(output_tensor, input_tensor, group='tp', async_op=False):
    if isinstance(group, str):
        group = get_group(group, 'gpu')
    return dist.all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group='tp', async_op=False):
    """Reduce scatter."""
    if isinstance(group, str):
        group = get_group(group, 'gpu')
    return dist.reduce_scatter(output, input_list, op=op, group=group, async_op=async_op)
