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
        self.misc_config = model_agent.misc_config
        self.inputs_strategy = model_agent.inputs_strategy
        self.device = model_agent.device
        self._in_que = model_agent._in_que

        # maker metas
        self._next_inputs = None
        self._is_decoding = False
        self._ready_event = torch.cuda.Event()
        self._attn_tp_cpu_group = self.dist_ctx.attn_tp_group.cpu_group

        # timeout to wait for inputs
        # if any rank has no inputs, all ranks would wait for this timeout
        # so it is very important to balance the inputs between ranks
        from lmdeploy.pytorch import envs
        self.base_timeout = envs.dp_input_timeout
        self.timeout = self.base_timeout

    def _make_dummy_forward_inputs(self):
        """Make dummy forward inputs."""
        is_decoding = self._is_decoding
        loop_count = self.misc_config.prefill_interval if is_decoding else 1
        dist_config = self.dist_ctx.dist_config
        batch_size = 2 if dist_config.enable_microbatch else 1
        batch_size = min(self.cache_config.max_batches, batch_size)
        model_inputs = self.inputs_strategy.make_dummy(batch_size,
                                                       is_decoding,
                                                       device=self.device,
                                                       vocab_size=self.model_config.vocab_size)
        forward_inputs = dict(
            inputs=model_inputs,
            loop_count=loop_count,
            is_dummy=True,
            sync_long_context=False,
        )
        return forward_inputs

    def _update_is_decoding(self, forward_inputs):
        """Update is decoding."""
        model_inputs = forward_inputs['inputs']
        assert model_inputs.is_decoding == self._is_decoding
        if self.cache_config.role != EngineRole.Prefill:
            self._is_decoding = not self._is_decoding

        if self.cache_config.role == EngineRole.Decode:
            # set timeout for next inputs
            # next inputs is ~self._is_decoding
            # and prefill is rarely happened in decoding engine
            if self._is_decoding:
                self.timeout = max(0.02, self.base_timeout / 2)
            else:
                self.timeout = self.base_timeout

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
        if self.model_agent._pre_in_que.qsize() > 0:
            forward_inputs = await self._in_que.get()
        else:
            try:
                forward_inputs = await asyncio.wait_for(self._in_que.get(), timeout=self.timeout)
            except asyncio.TimeoutError:
                forward_inputs = None

        has_inputs = await self._gather_has_inputs(forward_inputs is not None)

        # try get inputs
        if has_inputs and forward_inputs is None:
            forward_inputs = await self._in_que.get()

        return forward_inputs

    async def _try_get_inputs(self):
        """Try get inputs."""

        # initialize output
        forward_inputs = None
        need_dummy = True

        # get inputs from in_que. Rank 1 will not gather if rank 0 does not read inputs.
        forward_inputs = await self._get_inputs()

        if forward_inputs is not None:
            model_inputs = forward_inputs['inputs']
            if model_inputs.is_decoding != self._is_decoding:
                self._next_inputs = forward_inputs
            else:
                need_dummy = False

        return forward_inputs, need_dummy

    async def get(self):
        """get."""
        if self._next_inputs is not None:
            forward_inputs = self._next_inputs
            self._next_inputs = None
            self._update_is_decoding(forward_inputs)
            return forward_inputs

        # wait until has inputs or prev forward finish
        while self._in_que.qsize() == 0 and not self._ready_event.query():
            await asyncio.sleep(0.001)

        # try get inputs
        forward_inputs, need_dummy = await self._try_get_inputs()

        # make dummy inputs
        if need_dummy:
            forward_inputs = self._make_dummy_forward_inputs()

        self._update_is_decoding(forward_inputs)

        return forward_inputs

    def step(self):
        """step."""
        self._ready_event = torch.cuda.Event()
        self._ready_event.record()


def build_inputs_maker(model_agent: 'BaseModelAgent'):
    """Build inputs maker."""
    dist_config = model_agent.dist_ctx.dist_config
    if dist_config.dp > 1:
        return DPForwardInputsMaker(model_agent)
    else:
        return DefaultForwardInputsMaker(model_agent)
