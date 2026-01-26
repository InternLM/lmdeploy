# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import torch
from torch import distributed as dist
from torch.distributed import ProcessGroup, ReduceOp, Work  # noqa: F401

from lmdeploy.pytorch.utils import CtxMgrBase, singleton

from .config import DistConfig, TPMode


@dataclass
class DistGroup:
    """Distributed group."""
    rank: int = 0
    cpu_group: dist.ProcessGroup = None
    gpu_group: dist.ProcessGroup = None
    cpu_groups: List[dist.ProcessGroup] = None
    gpu_groups: List[dist.ProcessGroup] = None
    gpu_gather_group: dist.ProcessGroup = None

    def close(self):
        """Close groups."""
        if not dist.is_initialized():
            return
        if self.cpu_groups is not None:
            for group in self.cpu_groups:
                dist.destroy_process_group(group)
            self.cpu_groups = None
        if self.gpu_groups is not None:
            for group in self.gpu_groups:
                dist.destroy_process_group(group)
            self.gpu_groups = None


def _build_tp_group_impl(tp: int,
                         rank: int,
                         world_size: int,
                         timeout: timedelta,
                         cpu_backend: str = 'gloo',
                         ccl_backend: str = 'nccl',
                         attn_tp: int = 1,
                         tp_mode: TPMode = TPMode.DEFAULT):
    """Build tp group."""
    assert tp > 1
    tp_rank = rank % tp
    tp_group_id = rank // tp
    gather_group_id = (rank - tp_group_id * tp) % attn_tp
    ranks = range(world_size)
    tp_gpu_groups = []
    tp_cpu_groups = []
    gather_groups = []
    for start in range(0, world_size, tp):
        tp_ranks = ranks[start:start + tp]
        group = dist.new_group(ranks=tp_ranks, timeout=timeout, backend=ccl_backend)
        tp_gpu_groups.append(group)
        cpu_group = dist.new_group(ranks=tp_ranks, timeout=timeout, backend=cpu_backend)
        tp_cpu_groups.append(cpu_group)

        # create gather group
        if tp_mode == TPMode.DP_TP and attn_tp != tp:
            for g_start in range(start, start + attn_tp):
                g_ranks = ranks[g_start:(g_start + tp):attn_tp]
                gather_group = dist.new_group(ranks=g_ranks, timeout=timeout, backend=ccl_backend)
                gather_groups.append(gather_group)
    tp_gpu_group = tp_gpu_groups[tp_group_id]
    tp_cpu_group = tp_cpu_groups[tp_group_id]

    if tp_mode == TPMode.DP_TP:
        if attn_tp == tp:
            gather_group = tp_gpu_group
        else:
            gather_group = gather_groups[gather_group_id]
    else:
        gather_group = None
    return DistGroup(
        rank=tp_rank,
        cpu_group=tp_cpu_group,
        gpu_group=tp_gpu_group,
        cpu_groups=tp_cpu_groups,
        gpu_groups=tp_gpu_groups,
        gpu_gather_group=gather_group,
    )


def _build_attn_tp_group(context: 'DistContext',
                         timeout: timedelta,
                         cpu_backend: str = 'gloo',
                         ccl_backend: str = 'nccl'):
    """Build attention tp group."""
    dist_config = context.dist_config
    tp = dist_config.attn_tp
    # skip if tp == 1
    if tp == 1:
        context.attn_tp_group = DistGroup(rank=0)
        return

    dist_group = _build_tp_group_impl(
        tp,
        context.rank,
        dist_config.world_size,
        timeout=timeout,
        cpu_backend=cpu_backend,
        ccl_backend=ccl_backend,
        attn_tp=tp,
        tp_mode=TPMode.DEFAULT,
    )
    context.attn_tp_group = dist_group


def _build_mlp_tp_group(context: 'DistContext',
                        timeout: timedelta,
                        cpu_backend: str = 'gloo',
                        ccl_backend: str = 'nccl'):
    """Build mlp tp group."""
    dist_config = context.dist_config
    tp = dist_config.mlp_tp
    # skip if tp == 1
    if tp == 1:
        context.mlp_tp_group = DistGroup(rank=0)
        return

    # reuse attn tp group
    if tp == dist_config.attn_tp:
        context.mlp_tp_group = context.attn_tp_group
        return

    dist_group = _build_tp_group_impl(
        tp,
        context.rank,
        dist_config.world_size,
        timeout=timeout,
        cpu_backend=cpu_backend,
        ccl_backend=ccl_backend,
        attn_tp=dist_config.attn_tp,
        tp_mode=dist_config.mlp_tp_mode,
    )
    context.mlp_tp_group = dist_group


def _build_moe_tp_group(context: 'DistContext',
                        timeout: timedelta,
                        cpu_backend: str = 'gloo',
                        ccl_backend: str = 'nccl'):
    """Build moe tp group."""
    dist_config = context.dist_config
    tp = dist_config.moe_tp
    # skip if tp == 1
    if tp == 1:
        context.moe_tp_group = DistGroup(rank=0)
        return

    # reuse attn tp group
    if tp == dist_config.attn_tp:
        context.moe_tp_group = context.attn_tp_group
        return

    # reuse mlp tp group
    if tp == dist_config.mlp_tp:
        context.moe_tp_group = context.mlp_tp_group
        return

    dist_group = _build_tp_group_impl(
        tp,
        context.rank,
        dist_config.world_size,
        timeout=timeout,
        cpu_backend=cpu_backend,
        ccl_backend=ccl_backend,
        attn_tp=dist_config.attn_tp,
        tp_mode=dist_config.moe_tp_mode,
    )
    context.moe_tp_group = dist_group


def _build_tp_group(context: 'DistContext', timeout: timedelta, cpu_backend: str = 'gloo', ccl_backend: str = 'nccl'):
    """Build tp group."""
    _build_attn_tp_group(context, timeout, cpu_backend, ccl_backend)
    _build_mlp_tp_group(context, timeout, cpu_backend, ccl_backend)
    _build_moe_tp_group(context, timeout, cpu_backend, ccl_backend)
    context.tp_group = context.attn_tp_group


@dataclass
class DistContext:
    rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0

    tp_group: DistGroup = None
    attn_tp_group: DistGroup = None
    mlp_tp_group: DistGroup = None
    moe_tp_group: DistGroup = None

    cpu_group: dist.ProcessGroup = None
    ep_gpu_group: dist.ProcessGroup = None
    ep_gpu_groups: List[dist.ProcessGroup] = None
    dist_config: DistConfig = None

    @classmethod
    def _build_ep_group(cls, context: 'DistContext', timeout: timedelta, ccl_backend: str = 'nccl'):
        """Build ep group."""
        dist_config = context.dist_config
        ep = dist_config.ep
        if ep <= 1:
            return

        dp_rank = context.dp_rank
        world_size = dist_config.world_size
        ep_rank = context.rank % ep
        ep_group_id = dp_rank // ep
        ranks = range(world_size)
        ep_gpu_groups = []
        for start in range(0, world_size, ep):
            ep_ranks = ranks[start:start + ep]
            group = dist.new_group(ranks=ep_ranks, timeout=timeout, backend=ccl_backend)
            ep_gpu_groups.append(group)
        ep_gpu_group = ep_gpu_groups[ep_group_id]

        context.ep_rank = ep_rank
        context.ep_gpu_group = ep_gpu_group
        context.ep_gpu_groups = ep_gpu_groups

    @classmethod
    def build(cls, rank: int = 0, dist_config: DistConfig = None, ccl_backend: str = 'nccl'):
        """Build dist context."""
        timeout = timedelta(days=35600)
        cpu_backend = 'gloo'

        if dist_config is None:
            dist_config = DistConfig()

        dp_rank = dist_config.dp_rank
        world_size = dist_config.world_size
        context = DistContext(rank=rank,
                              dp_rank=dp_rank,
                              dist_config=dist_config,
                              attn_tp_group=DistGroup(rank=0),
                              mlp_tp_group=DistGroup(rank=0),
                              moe_tp_group=DistGroup(rank=0),
                              tp_group=DistGroup(rank=0))
        if world_size == 1:
            return context

        assert dist.is_initialized()

        # cpu group
        context.cpu_group = dist.new_group(ranks=list(range(world_size)), timeout=timeout, backend=cpu_backend)

        # tp
        _build_tp_group(context, timeout, cpu_backend=cpu_backend, ccl_backend=ccl_backend)

        # ep
        cls._build_ep_group(context, timeout, ccl_backend=ccl_backend)

        return context

    def close(self):
        """Close groups."""
        if not dist.is_initialized():
            return
        if self.attn_tp_group is not None:
            self.attn_tp_group.close()
        if self.mlp_tp_group is not None:
            self.mlp_tp_group.close()
        if self.moe_tp_group is not None:
            self.moe_tp_group.close()
        if self.ep_gpu_groups is not None:
            for group in self.ep_gpu_groups:
                dist.destroy_process_group(group)
            self.ep_gpu_groups = None


DefaultContext = DistContext.build()


@singleton
class DistManager(CtxMgrBase[DistContext]):
    """Distributed context manager."""

    def __init__(self):
        super().__init__(DefaultContext)

    def current_config(self) -> DistConfig:
        """Get current dist config."""
        return self.current_context().dist_config


def get_dist_manager():
    """Get device manager."""
    return DistManager()


def get_world_rank():
    """Get distributed world size and rank."""
    ctx = get_dist_manager().current_context()
    world_size = ctx.dist_config.world_size
    rank = ctx.rank

    return world_size, rank


def get_tp_world_rank(layer_type: Optional[str] = None):
    ctx = get_dist_manager().current_context()
    if layer_type is None:
        return ctx.dist_config.tp, ctx.tp_group.rank
    elif layer_type == 'attn':
        return ctx.dist_config.attn_tp, ctx.attn_tp_group.rank
    elif layer_type == 'mlp':
        return ctx.dist_config.mlp_tp, ctx.mlp_tp_group.rank
    elif layer_type == 'moe':
        return ctx.dist_config.moe_tp, ctx.moe_tp_group.rank
    else:
        raise RuntimeError(f'Unknown layer type: {layer_type}')


def get_dp_world_rank():
    ctx = get_dist_manager().current_context()
    return ctx.dist_config.dp, ctx.dp_rank


def get_ep_world_rank():
    ctx = get_dist_manager().current_context()
    return ctx.dist_config.ep, ctx.ep_rank


def _check_group_device(device: str):
    """Check group device."""
    assert (device in ['cpu', 'gpu']), ('Expect process group device in ("cpu", "gpu"), '
                                        f'but get {device}.')


def get_process_group(device: str = None):
    """Get process group."""
    return dist.GroupMember.WORLD


def get_dist_group(layer_type: str = 'attn'):
    """Get dist group."""
    ctx = get_dist_manager().current_context()
    if layer_type == 'attn':
        tp_group = ctx.attn_tp_group
    elif layer_type == 'mlp':
        tp_group = ctx.mlp_tp_group
    elif layer_type == 'moe':
        tp_group = ctx.moe_tp_group
    else:
        raise RuntimeError(f'Unknown layer type: {layer_type}')
    return tp_group


def get_tp_group(device: str = 'gpu', layer_type: str = 'attn'):
    """Get tp group."""
    _check_group_device(device)
    tp_group = get_dist_group(layer_type)

    if tp_group is None:
        return None

    if device == 'cpu':
        return tp_group.cpu_group
    else:
        return tp_group.gpu_group


def get_group(group_type: str, device: str):
    """Get group."""
    if group_type == 'tp':
        return get_tp_group(device)
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


def gather_by_tp_sizes(x: torch.Tensor,
                       tp_sizes: List[int],
                       group: Optional[dist.ProcessGroup] = None,
                       async_op: bool = False):
    """Gather input."""
    assert all(size >= 0 for size in tp_sizes), f'Invalid tp sizes: {tp_sizes}'
    shape = (*x.shape[:-2], sum(tp_sizes), *x.shape[-1:])
    new_x = x.new_empty(shape)
    split_new_x = list(new_x.split(tp_sizes, -2))
    handle = dist.all_gather(split_new_x, x, group=group, async_op=async_op)
    if async_op:
        return new_x, handle
    return new_x


def reduce_scatter_by_tp_sizes(out: torch.Tensor, rank: int, tp_sizes: List[int], group: dist.ProcessGroup):
    """Reduce scatter."""
    attn_tp = get_dist_manager().current_config().attn_tp
    outs = list(out.split(tp_sizes, -2))
    outs = [item for item in outs for _ in range(attn_tp)]
    out = outs[rank]
    dist.reduce_scatter(out, outs, group=group)
    return out
