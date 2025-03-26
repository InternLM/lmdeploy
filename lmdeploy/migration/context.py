import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple

import torch
import zmq
import zmq.asyncio

from . import _migration_c


class RDMAContext:
    def __init__(
        self,
        dev_name: str,
        meta_endpoint: str,
        ib_port: int = 1,
        link_type: str = "Ethernet",
    ):
        self._rdma_context_c = _migration_c.rdma_context()

        self.meta_agent = zmq.asyncio.Context()
        self.meta_send = self.meta_agent.socket(zmq.PUSH)
        self.meta_recv = self.meta_agent.socket(zmq.PULL)
        self.meta_recv.bind(f"tcp://{meta_endpoint}")

        self.init_rdma_context(dev_name, ib_port, link_type)

    async def connect(self, endpoint: str):
        # bind tcp
        self.meta_send.connect(f"tcp://{endpoint}")
        exchange_info = self._rdma_context_c.exchange_info()
        await self.meta_send.send_pyobj(exchange_info)
        remote_info = await self.meta_recv.recv_pyobj()
        await self.construct(remote_info)

    def init_rdma_context(
        self,
        dev_name: str,
        ib_port: int = 1,
        link_type: str = "Ethernet",
    ) -> int:
        return self._rdma_context_c.init_rdma_context(dev_name, ib_port, link_type)

    def register_torch(self, mr_key, t: torch.Tensor):
        self._rdma_context_c.register_memory(
            mr_key, t.data_ptr(), t.numel() * t.itemsize
        )

    async def construct(self, info):
        # - qp init -> rts -> rtr
        self._rdma_context_c.connect(info)
        self._rdma_context_c.launch_cq_future()

    @torch.no_grad()
    async def r_rdma_async_batch_handler(self):
        while True:
            mr_key, offset, length = await self.meta_recv.recv_pyobj()
            offset = torch.tensor(offset, dtype=torch.int64, device="cuda")
            _migration_c.gather(
                self.memory_pool[mr_key].data_ptr(),
                self.memory_pool["buffer"].data_ptr(),
                length,
                offset.data_ptr(),
                offset.numel(),
            )
            torch.cuda.synchronize()
            await self.meta_send.send_pyobj("done")

    @torch.no_grad()
    async def r_rdma_async_batch(
        self,
        mr_key: str,
        target_offset: List[int],
        source_offset: List[int],
        length: List[int],
        callback=None,
    ):
        # Step 1. Send request to get the buffer.
        await self.meta_send.send_pyobj([mr_key, target_offset, length])
        total_length = length * len(target_offset)

        # Step 2. Recv the buffer tensor
        await self.meta_recv.recv_pyobj()

        # Step 3. Read the Buffer
        await self.r_rdma_async("buffer", 0, 0, total_length)

        # # Step 4. Tensor Scatter
        source_offset = torch.tensor(source_offset, dtype=torch.int64, device="cuda")
        _migration_c.scatter(
            self.memory_pool[mr_key].data_ptr(),
            self.memory_pool["buffer"].data_ptr(),
            length,
            source_offset.data_ptr(),
            source_offset.numel(),
        )
        torch.cuda.synchronize()

    async def batch_r_rdma_async(
        self, mr_key, target_offset, source_offset, length, callback=None
    ):
        rdma_call_back = None
        if callback is None:
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            def _default_callback(code):
                loop.call_soon_threadsafe(future.set_result, code)

            rdma_call_back = _default_callback
        else:
            rdma_call_back = callback

        self._rdma_context_c.batch_r_rdma_async(
            target_offset,
            source_offset,
            length,
            mr_key,
            rdma_call_back,
        )

        if callback is None:
            await future

    async def r_rdma_async(
        self, mr_key, target_offset, source_offset, length, callback=None
    ):
        rdma_call_back = None
        if callback is None:
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            def _default_callback(code):
                loop.call_soon_threadsafe(future.set_result, code)

            rdma_call_back = _default_callback
        else:
            rdma_call_back = callback

        self._rdma_context_c.r_rdma_async(
            target_offset,
            source_offset,
            length,
            mr_key,
            rdma_call_back,
        )

        if callback is None:
            await future
