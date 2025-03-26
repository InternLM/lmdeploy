import asyncio

from . import _migration_c


class RDMAContext:
    def __init__(
        self,
        dev_name: str,
        ib_port: int = 1,
        link_type: str = "Ethernet",
    ):
        self._rdma_context_c = _migration_c.rdma_context()
        self.init_rdma_context(dev_name, ib_port, link_type)

    @property
    async def local_info(self):
        return self._rdma_context_c.exchange_info()

    def init_rdma_context(
        self,
        dev_name: str,
        ib_port: int = 1,
        link_type: str = "Ethernet",
    ) -> int:
        return self._rdma_context_c.init_rdma_context(dev_name, ib_port, link_type)

    async def connect(self, exchange_info):
        self._rdma_context_c.connect(exchange_info)
        self._rdma_context_c.launch_cq_future()

    def register_memory(self, mr_key, addr: int, length: int):
        self._rdma_context_c.register_memory(mr_key, addr, length)

    async def batch_r_rdma_async(
        self,
        mr_key,
        target_offset,
        source_offset,
        length,
    ):
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _callback(code):
            loop.call_soon_threadsafe(future.set_result, code)

        self._rdma_context_c.batch_r_rdma_async(
            target_offset,
            source_offset,
            length,
            mr_key,
            _callback,
        )

        await future

    async def r_rdma_async(
        self,
        mr_key,
        target_offset,
        source_offset,
        length,
    ):
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _callback(code):
            loop.call_soon_threadsafe(future.set_result, code)

        self._rdma_context_c.r_rdma_async(
            target_offset,
            source_offset,
            length,
            mr_key,
            _callback,
        )

        await future
