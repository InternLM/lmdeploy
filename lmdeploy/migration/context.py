import json
import time
import asyncio

from typing import Dict, Tuple

import torch

from .config import MemoryRegionInfo, RDMAInfo
from . import _migration_c


class RDMAContext:

    def __init__(self,
                 dev_name: str,
                 ib_port: int = 1,
                 link_type: str = "Ethernet"):
        self._rdma_context_c = _migration_c.rdma_context()
        self.init_rdma_context(dev_name, ib_port, link_type)
        self.remote_memory_pool: Dict[str, MemoryRegionInfo] = {}
        self.memory_pool: Dict[str, torch.Tensor] = {}

    def init_rdma_context(self,
                          dev_name: str,
                          ib_port: int = 1,
                          link_type: str = "Ethernet") -> int:
        return self._rdma_context_c.init_rdma_context(dev_name, ib_port,
                                                      link_type)

    def register_mr(self, mr_key, length: int, device="cpu"):
        t = torch.zeros((length, ), dtype=torch.uint8, requires_grad=False)
        self._rdma_context_c.register_memory_region(mr_key, t.data_ptr(),
                                                    length)
        self.memory_pool[mr_key] = t

    def register_torch(self, mr_key, t: torch.Tensor):
        self._rdma_context_c.register_memory_region(mr_key, t.data_ptr(),
                                                    t.numel() * t.itemsize)
        self.memory_pool[mr_key] = t

    def construct(self, info: RDMAInfo):
        remote_rdma_info = _migration_c.rdma_info(info.qpn, info.gid[0],
                                              info.gid[1], info.gidx, info.lid,
                                              info.psn, info.mtu)
        self._rdma_context_c.modify_qp_to_rtsr(remote_rdma_info)
        self._rdma_context_c.launch_cq_future()

    def get_local_info(self):
        local_info = self._rdma_context_c.get_local_rdma_info()
        return RDMAInfo(gid=local_info.get_gid(),
                        gidx=local_info.gidx,
                        lid=local_info.lid,
                        qpn=local_info.qpn,
                        psn=local_info.psn,
                        mtu=local_info.mtu)

    async def r_rdma_async(self,
                           mr_key,
                           target_offset,
                           source_offset,
                           length,
                           callback=None):
        print(self.remote_memory_pool[mr_key].addr + target_offset)
        print(self.remote_memory_pool[mr_key].r_key)

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
            self.memory_pool[mr_key].data_ptr() + source_offset, length,
            mr_key, self.remote_memory_pool[mr_key].r_key, rdma_call_back)

        if callback is None:
            await future

    def get_mr_info(self, mr_key) -> MemoryRegionInfo:
        return MemoryRegionInfo(
            addr=self.memory_pool[mr_key].data_ptr(),
            offset=self.memory_pool[mr_key].storage_offset(),
            r_key=self._rdma_context_c.get_r_key(mr_key))

    def get_remote_mr_info(self, mr_key) -> MemoryRegionInfo:
        return self.remote_memory_pool[mr_key]

    def register_remote_mr(self, mr_key, mr_info: MemoryRegionInfo):
        self.remote_memory_pool[mr_key] = mr_info
