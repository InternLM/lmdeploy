# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from lmdeploy.pytorch.disagg.config import EngineRole

if TYPE_CHECKING:
    from .agent import BaseModelAgent


class DefaultForwardInputsMaker:
    """Default forward inputs maker."""

    def __init__(self, model_agent: 'BaseModelAgent'):
        self._in_que = model_agent._in_que

    async def get(self):
        """get."""
        return await self._in_que.get()

    def step(self):
        """step."""
        # No-op for default maker
        pass


class DPForwardInputsMaker:
    """Dp forward inputs maker."""

    def __init__(self, model_agent: 'BaseModelAgent'):
        self.model_agent = model_agent
        self.dist_ctx = model_agent.dist_ctx
        self.model_config = model_agent.model_config
        self.cache_config = model_agent.cache_config
        self.inputs_strategy = model_agent.inputs_strategy
        self.device = model_agent.device
        self._in_que = model_agent._in_que

        # maker metas
        self._ready_event = torch.cuda.Event()
        self._ready_event.record()

    def _make_dummy_forward_inputs(self):
        """Make dummy forward inputs."""
        is_decoding = self.cache_config.role != EngineRole.Prefill
        dist_config = self.dist_ctx.dist_config
        batch_size = 2 if dist_config.enable_microbatch else 1
        batch_size = min(self.cache_config.max_batches, batch_size)
        model_inputs = self.inputs_strategy.make_dummy(batch_size,
                                                       is_decoding,
                                                       device=self.device,
                                                       vocab_size=self.model_config.vocab_size)
        forward_inputs = dict(inputs=model_inputs, )
        return forward_inputs

    async def _gather_has_inputs(self, has_inputs: bool = False):
        """Broadcast has inputs."""
        attn_tp_group = self.dist_ctx.attn_tp_group
        attn_tp = self.dist_ctx.dist_config.attn_tp
        if attn_tp == 1:
            return has_inputs

        group = attn_tp_group.cpu_group
        has_inputs = torch.tensor((int(has_inputs), ))
        handle = dist.all_reduce(has_inputs, op=dist.ReduceOp.SUM, group=group, async_op=True)
        future = handle.get_future()
        while not future.done():
            await asyncio.sleep(0)
        future.wait()
        return (has_inputs > 0).item()

    async def _get_inputs(self):
        # get local forward inputs
        try:
            forward_inputs = self._in_que.get_nowait()
        except asyncio.QueueEmpty:
            forward_inputs = None

        # async inputs around tp group
        has_inputs = await self._gather_has_inputs(forward_inputs is not None)
        if has_inputs and forward_inputs is None:
            forward_inputs = await self._in_que.get()

        return forward_inputs

    async def get(self):
        """get."""
        # # wait until has inputs or prev forward finish
        while self._in_que.qsize() == 0 and not self._ready_event.query():
            await asyncio.sleep(0.001)

        # try get inputs
        forward_inputs = await self._get_inputs()

        # make dummy inputs
        if forward_inputs is None:
            forward_inputs = self._make_dummy_forward_inputs()

        return forward_inputs

    def step(self):
        """step."""
        self._ready_event.wait()
        self._ready_event = torch.cuda.Event()
        self._ready_event.record()


def build_inputs_maker(model_agent: 'BaseModelAgent'):
    """Build inputs maker."""
    dist_config = model_agent.dist_ctx.dist_config
    if dist_config.dp > 1:
        return DPForwardInputsMaker(model_agent)
    else:
        return DefaultForwardInputsMaker(model_agent)
