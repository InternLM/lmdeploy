# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from lmdeploy.pytorch.disagg.config import EngineRole

if TYPE_CHECKING:
    from .agent import BaseModelAgent


# Polling is only used inside the worker actor while a real input is in the
# CPU-side preprocess queue but has not reached the CUDA-ready queue yet.
_PREPROCESS_POLL_INTERVAL = 0.001

# Ray actor delivery of forward_async may lag behind the background forward
# loop by a few event-loop turns. Yield briefly before falling back to dummy
# inputs so just-scheduled real inputs can reach _pre_in_que.
_FORWARD_RPC_YIELD_TURNS = 2


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
    """DP forward inputs maker.

    DP workers must enter a forward step even when a rank has no local request,
    so this maker creates dummy inputs as a fallback. Real inputs arrive in two
    stages: ``set_forward_inputs`` enqueues CPU-side data into ``_pre_in_que``,
    then the preprocess task moves CUDA-ready data to ``_in_que``. A non-empty
    ``_pre_in_que`` is therefore pending real work and should delay dummy
    fallback, except while the model agent is sleeping because those queued
    items may be stale across sleep/wakeup.
    """

    def __init__(self, model_agent: 'BaseModelAgent'):
        self.model_agent = model_agent
        self.dist_ctx = model_agent.dist_ctx
        self.model_config = model_agent.model_config
        self.cache_config = model_agent.cache_config
        self.inputs_strategy = model_agent.inputs_strategy
        self.device = model_agent.device
        self._pre_in_que = model_agent._pre_in_que
        self._in_que = model_agent._in_que

        # maker metas
        self._ready_event = torch.cuda.Event()
        self._ready_event.record()

        # other
        self.make_dummy_meta = model_agent.make_dummy_meta

    def _make_dummy_forward_inputs(self):
        """Make dummy forward inputs."""
        is_decoding = self.cache_config.role != EngineRole.Prefill
        dist_config = self.dist_ctx.dist_config
        batch_size = 2 if dist_config.enable_microbatch else 1
        batch_size = min(self.cache_config.max_batches, batch_size)
        model_inputs = self.inputs_strategy.make_dummy(batch_size,
                                                       is_decoding,
                                                       device=self.device,
                                                       vocab_size=self.model_config.vocab_size,
                                                       meta=self.make_dummy_meta)
        extra_inputs = self.inputs_strategy.make_dummy_extra_inputs(model_inputs, meta=self.make_dummy_meta)
        return_logits = self.model_agent.spec_agent.is_enabled()
        forward_inputs = dict(inputs=model_inputs, extra_inputs=extra_inputs, return_logits=return_logits)
        return forward_inputs

    def _is_sleeping(self):
        """Whether the model agent is entering sleep."""
        state = getattr(self.model_agent, 'state', None)
        return bool(getattr(state, 'is_sleeping', False))

    def _has_pending_real_inputs(self):
        """Whether real inputs are waiting for preprocessing or ready.

        ``_pre_in_que`` matters even when ``_in_que`` is empty: H2D transfer and
        lightweight preprocessing happen between those queues, and replacing
        that pending real input with a dummy would waste a DP forward step.
        """
        if self._is_sleeping():
            return False
        return self._in_que.qsize() > 0 or self._pre_in_que.qsize() > 0

    async def _wait_preprocessed_real_inputs(self):
        """Wait until queued real inputs are ready for forward.

        Keep checking the sleep state while waiting so sleep/wakeup can ignore stale queued inputs and let the normal
        dummy/sleep metadata path run.
        """
        while not self._is_sleeping() and self._in_que.qsize() == 0 and self._pre_in_que.qsize() > 0:
            await asyncio.sleep(_PREPROCESS_POLL_INTERVAL)

    async def _yield_to_forward_rpc(self):
        """Let scheduled worker RPCs enqueue inputs before dummy fallback.

        The scheduler may have already issued ``worker.forward_async.remote``,
        while the worker actor has not yet executed ``set_forward_inputs``.
        A few zero-time yields reduce dummy forwards caused by this async
        delivery gap without making the loop block when no real work exists.
        """
        if self._is_sleeping():
            return
        for _ in range(_FORWARD_RPC_YIELD_TURNS):
            if self._has_pending_real_inputs():
                return
            await asyncio.sleep(0)

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
        if self._is_sleeping():
            return None

        await self._wait_preprocessed_real_inputs()

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
        while not self._has_pending_real_inputs() and not self._ready_event.query():
            await asyncio.sleep(_PREPROCESS_POLL_INTERVAL)

        await self._yield_to_forward_rpc()

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
