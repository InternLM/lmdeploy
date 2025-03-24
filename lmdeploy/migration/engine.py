import asyncio

from typing import Dict, List, Tuple

import torch
import zmq

from .config import ExchangeInfo, RDMAInfo
from .context import RDMAContext


class TransferEngine:

    def __init__(self):
        self.links: Dict[int, RDMAContext] = {}

    def init_link(
        self,
        session_id: int,
        dev_name: str,
        metadata_endpoint,
        ib_port=1,
        link_type: str = "Ethernet",
    ) -> RDMAContext:
        if session_id in self.links:
            raise KeyError(f"session_id {session_id} already in links")
        self.links[session_id] = RDMAContext(
            dev_name=dev_name,
            meta_endpoint=metadata_endpoint,
            ib_port=ib_port,
            link_type=link_type,
        )
        return self.links[session_id]

    def register_mr(self, session_id, mr_key, length, device="cpu"):
        if session_id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[session_id].register_mr(mr_key, length, device=device)

    def register_torch(self, session_id, mr_key, t):
        if session_id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        return self.links[session_id].register_torch(mr_key, t)

    async def construct(self, id, local_info: RDMAInfo):
        if id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        await self.links[id].construct(local_info)

    def get_local_info(self, session_id: int) -> RDMAInfo:
        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")
        local_info = self.links[session_id].get_local_info()
        return local_info

    async def r_rdma_async(
        self, session_id, mr_key, target_offset, source_offset, length
    ):
        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")
        await self.links[session_id].r_rdma_async(
            mr_key, target_offset, source_offset, length
        )

    async def buffered_send_tensor(
        self,
        session_id: int,
        tensor: torch.Tensor,
        send_indices: List[int],
        remote_host: str,
        remote_port: int,
        local_port: int,
    ):
        """
        Sender gather tensor into a buffer tensor based on send_indices, then sent rdma infos through tcp to receiver.
        Receiver can read it after have those info.
        """

        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")

        #
        # Gather tensors based on indices
        #
        send_index_tensor = torch.tensor(
            send_indices, dtype=torch.int64, device=tensor.device
        )
        # Reshape and expand the indices to match tensor's dimensions
        expend_send_index = send_index_tensor.view(
            -1, *([1] * (tensor.dim() - 1))
        ).expand(-1, *tensor.shape[1:])
        # Gather the elements along dim=0
        buffer_tensor = torch.gather(tensor, dim=0, index=expend_send_index)

        #
        # Register the gather tensor on MR
        #
        rdma_link = self.links[session_id]
        mr_key = str(buffer_tensor.data_ptr())
        rdma_link.register_torch(mr_key, buffer_tensor)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        #
        # Tcp send meta
        #
        zmq_ctx = zmq.Context(2)
        send_socket = zmq_ctx.socket(zmq.PUSH)
        send_socket.connect(f"tcp://{remote_host}:{remote_port}")
        recv_socket = zmq_ctx.socket(zmq.PULL)
        recv_socket.bind(f"tcp://*:{local_port}")

        local_rdma_info = rdma_link.get_local_info()
        local_mr_info = rdma_link.get_mr_info(mr_key)
        send_socket.send_pyobj([local_rdma_info, local_mr_info])
        remote_rdma_info, remote_mr_info = recv_socket.recv_pyobj()
        rdma_link.construct(remote_rdma_info)

        future.set_result(0)
        return future

    async def buffered_receive_tensor(
        self,
        session_id: int,
        out_tensor: torch.Tensor,
        receiver_indices: List[int],
        remote_host: str,
        remote_port: int,
        local_port: int,
    ):
        """
        Receiver read the remote buffer tensor to local buffer tensor, then scatter it to out_tensor.
        """

        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")

        #
        # Regist the buffer tensor on MR
        #
        rdma_link = self.links[session_id]
        buffer_shape = (len(receiver_indices),) + out_tensor.shape[1:]
        buffer_tensor = torch.zeros(
            buffer_shape, device=out_tensor.device, dtype=out_tensor.dtype
        )
        mr_key = str(buffer_tensor.data_ptr())
        rdma_link.register_torch(mr_key, buffer_tensor)

        #
        # tcp exchange meta
        #
        zmq_ctx = zmq.Context(2)
        send_socket = zmq_ctx.socket(zmq.PUSH)
        send_socket.connect(f"tcp://{remote_host}:{remote_port}")
        recv_socket = zmq_ctx.socket(zmq.PULL)
        recv_socket.bind(f"tcp://*:{local_port}")

        local_rdma_info = rdma_link.get_local_info()
        local_mr_info = rdma_link.get_mr_info(mr_key)
        send_socket.send_pyobj([local_rdma_info, local_mr_info])
        remote_rdma_info, remote_mr_info = recv_socket.recv_pyobj()
        rdma_link.register_remote_mr(mr_key, remote_mr_info)
        rdma_link.construct(remote_rdma_info)

        #
        # Do scatter callback once read finish
        #
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _scatter_callback(code):
            if code == 0:
                #
                # Scatter tensors based on indices
                #
                receive_index_tensor = torch.tensor(
                    receiver_indices, dtype=torch.int64, device=out_tensor.device
                )
                # Reshape and expand the indices to match tensor's dimensions
                expend_receive_index = receive_index_tensor.view(
                    -1, *([1] * (buffer_tensor.dim() - 1))
                ).expand(-1, *buffer_tensor.shape[1:])
                # Scatter the elements along dim=0
                out_tensor.scatter_(
                    dim=0, index=expend_receive_index, src=buffer_tensor
                )
                # Success, we should run call back
                loop.call_soon_threadsafe(future.set_result, code)
            else:
                loop.call_soon_threadsafe(future.set_exception, code)

        read_len = buffer_tensor.numel() * buffer_tensor.itemsize
        await rdma_link.r_rdma_async(
            mr_key,
            remote_mr_info.offset,
            local_mr_info.offset,
            read_len,
            _scatter_callback,
        )
        return future

    def stop_link(self, session_id: int):
        if session_id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[session_id].stop_link()
        del self.links[session_id]
