# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import List

import torch

from lmdeploy.utils import get_logger

from .cache_engine import EncoderCacheEngine

logger = get_logger('lmdeploy')


def _try_to_cuda(data, non_blocking: bool = True):
    """Recursively traverses a data structure and moves all torch.Tensors to
    the configured device."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to('cuda', non_blocking=non_blocking)
    if isinstance(data, list):
        return [_try_to_cuda(item, non_blocking) for item in data]
    if isinstance(data, tuple):
        return tuple(_try_to_cuda(item, non_blocking) for item in data)
    if isinstance(data, dict):
        return {key: _try_to_cuda(value, non_blocking) for key, value in data.items()}
    return data


def _try_to_cpu(data):
    """Recursively traverses a data structure and moves all torch.Tensors to
    the CPU."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu()
    if isinstance(data, list):
        return [_try_to_cpu(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_try_to_cpu(item) for item in data)
    if isinstance(data, dict):
        return {key: _try_to_cpu(value) for key, value in data.items()}
    return data


class BaseModelAgent:

    def __init__(self, model):

        # PreProcessor -> h2d loop
        self._pre_in_que = asyncio.Queue()
        # h2d loop -> forward loop
        self._in_que = asyncio.Queue()
        # forward loop -> d2h loop
        self._out_que = asyncio.Queue()
        # d2h loop -> PostProcessor
        self._post_proc_que = asyncio.Queue()

        # backpressure signal between h2d loop <-> forward loop
        self.has_inputs = asyncio.Event()

        # CUDA streams
        self.in_stream = torch.cuda.Stream()
        self.out_stream = torch.cuda.Stream()
        self.forward_stream = torch.cuda.Stream()

        self.model = model
        self.device = 'cuda'

    async def make_batch(self):
        # TODO: fix for multi-batch
        requests = []

        req = await self._pre_in_que.get()
        requests.append(req)

        return requests[0]

    async def async_model_forward(self):
        """Model forward."""
        while True:
            # wait for inputs
            session_id, forward_inputs = await self._in_que.get()
            print(f'get session_id: {session_id}')
            print(f'get forward inputs from _in_que: {forward_inputs}')
            self.next_inputs = None

            with torch.cuda.stream(self.forward_stream):
                feats, allocated_blocks = self._forward_impl(forward_inputs)

                # event for async fetch outputs
                event = torch.cuda.Event()
                event.record()

                # put inside out_que
                out = dict(
                    session_id=session_id,
                    feats=feats,
                    block_ids=allocated_blocks,
                    event=event,
                )
                self._out_que.put_nowait(out)

            # reset events, for h2d prepare the next round inputs
            self.has_inputs.set()

    async def h2d_loop(self):
        """Host to device loop.

        preprocess inputs and put them into in_que. copy inputs to device in a different stream.
        """
        while True:
            await self.has_inputs.wait()

            session_id, forward_inputs = await self.make_batch()
            print(f'check forward_inputs: {forward_inputs}')

            # use a different stream to copy h2d
            with torch.cuda.stream(self.in_stream):
                forward_inputs = _try_to_cuda(forward_inputs)

            # put inputs inside in_que, reset has_inputs
            self._in_que.put_nowait((session_id, forward_inputs))
            self.has_inputs.clear()

    async def d2h_loop(self):
        """Device to host loop.

        copy outputs from device to host. put outputs into post processing queue.
        """
        while True:
            out = await self._out_que.get()

            # check event periodically
            event = out.pop('event')
            while not event.query():
                await asyncio.sleep(0.001)

            # use a different stream to copy d2h
            with torch.cuda.stream(self.out_stream):
                out = _try_to_cpu(out)

            self._post_proc_que.put_nowait(out)

    def start_loop(self):
        """Start event loop."""
        event_loop = asyncio.get_event_loop()

        # set for the first batch
        self.has_inputs.set()

        # forward task
        logger.info('Create task MultiModal ModelAgent ForwardLoop.')
        self._forward_task = event_loop.create_task(self.async_model_forward(), name='ModelAgentForwardLoop')

        # preprocess inputs task
        logger.info('Create task MultiModal ModelAgent Preprocess.')
        self._preprocess_task = event_loop.create_task(self.h2d_loop(), name='ModelAgentPreprocess')

        # postprocess outputs task
        logger.info('Create task MultiModal ModelAgent Postprocess.')
        self._postprocess_task = event_loop.create_task(self.d2h_loop(), name='ModelAgentPostprocess')

        loop_tasks: list[asyncio.Task] = [self._forward_task, self._preprocess_task, self._postprocess_task]

        # binding done callback
        self._add_loop_tasks_done_callback(loop_tasks)

    @staticmethod
    def _add_loop_tasks_done_callback(tasks: List[asyncio.Task]):
        """Add loop tasks done callback."""

        def __task_callback(task: asyncio.Task) -> None:
            """Raise exception on finish."""
            task_name = task.get_name()
            try:
                task.result()
            except asyncio.CancelledError:
                logger.debug(f'Task <{task_name}> cancelled.')
                return
            except Exception:
                logger.exception(f'Task <{task_name}> failed')
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()

        for task in tasks:
            task.add_done_callback(__task_callback)

    def build_cache_engine(self):
        cache_engine = EncoderCacheEngine()
        self.cache_engine = cache_engine

    def _forward_impl(self, inputs):
        """Model forward implementation."""
        feats = self.model.forward(inputs)

        # put feat into encoder cache
        feats = feats[0]  # FIXME
        num_required_blocks = feats.shape[0] // 256
        if len(self.cache_engine.free_blocks) < num_required_blocks:
            raise RuntimeError('Not enough free blocks in cache engine')
        allocated_blocks = self.cache_engine.free_blocks[:num_required_blocks]

        # move into dedicated mm cache pool
        # TODO: we dont want copy, better to just write into that memory region
        # but current transformers get_image_features() returns a new tensor, seems no way to achieve this
        for i in range(num_required_blocks):
            src_chunk = feats[i * 256:(i + 1) * 256, :]
            dst_block_id = allocated_blocks[i]
            self.cache_engine.gpu_cache[dst_block_id].copy_(src_chunk)
        print(f'=> allocated blocks: {allocated_blocks}')

        return feats, allocated_blocks

    def init(self):
        self.build_cache_engine()

    def close(self):
        self.cache_engine = None
        self.model = None
        torch.cuda.empty_cache()


def build_model_agent(model):
    """Build model agent."""
    model_agent = BaseModelAgent(model=model, )
    return model_agent
