import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple

import torch
import zmq
import zmq.asyncio

from . import _migration_c
from .config import ExchangeInfo, MemoryRegionInfo, RDMAInfo


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
        self.remote_memory_pool: Dict[str, MemoryRegionInfo] = {}
        self.memory_pool: Dict[str, torch.Tensor] = {}

    async def connect(self, endpoint: str):
        # bind tcp
        self.meta_send.connect(f"tcp://{endpoint}")
        local_info = self.dump_info()
        await self.meta_send.send_pyobj(local_info)
        remote_info = await self.meta_recv.recv_pyobj()
        await self.construct(remote_info)

    def dump_info(self):
        return ExchangeInfo(
            mr_info={mr_key: self.get_mr_info(mr_key) for mr_key in self.memory_pool},
            rdma_info=self.get_local_info(),
        )

    def get_local_info(self):
        return RDMAInfo._from_migration_c(self._rdma_context_c.get_local_rdma_info())

    def init_rdma_context(
        self,
        dev_name: str,
        ib_port: int = 1,
        link_type: str = "Ethernet",
    ) -> int:
        return self._rdma_context_c.init_rdma_context(dev_name, ib_port, link_type)

    def register_mr(self, mr_key, length: int, device="cpu"):
        t = torch.zeros(
            (length,), dtype=torch.uint8, requires_grad=False, device=device
        )
        self._rdma_context_c.register_memory_region(mr_key, t.data_ptr(), length)
        self.memory_pool[mr_key] = t

    def register_remote_mr(self, mr_key, mr_info: MemoryRegionInfo):
        self.remote_memory_pool[mr_key] = mr_info

    def register_torch(self, mr_key, t: torch.Tensor):
        self._rdma_context_c.register_memory_region(
            mr_key, t.data_ptr(), t.numel() * t.itemsize
        )
        self.memory_pool[mr_key] = t
        return self.get_mr_info(mr_key)

    async def construct(self, info: ExchangeInfo):
        # - qp init -> rts -> rtr
        _remote_rdma_info_c = info.rdma_info._to_migration_c()
        self._rdma_context_c.modify_qp_to_rtsr(_remote_rdma_info_c)
        self._rdma_context_c.launch_cq_future()

        # register remote mr
        for mr_key, mr_info in info.mr_info.items():
            self.register_remote_mr(mr_key, mr_info)

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
            self.remote_memory_pool[mr_key].addr + target_offset,
            self.memory_pool[mr_key].data_ptr() + source_offset,
            length,
            mr_key,
            self.remote_memory_pool[mr_key].r_key,
            rdma_call_back,
        )

        if callback is None:
            await future

    def get_mr_info(self, mr_key) -> MemoryRegionInfo:
        return MemoryRegionInfo(
            addr=self.memory_pool[mr_key].data_ptr(),
            offset=self.memory_pool[mr_key].storage_offset(),
            r_key=self._rdma_context_c.get_r_key(mr_key),
        )

    def get_remote_mr_info(self, mr_key) -> MemoryRegionInfo:
        return self.remote_memory_pool[mr_key]

    def stop(self):
        self._rdma_context_c.stop_cq_future()
