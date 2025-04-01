import asyncio

import time
from typing import Dict, List, Tuple

import torch

import zmq

from slime.transport.rdma_endpoint import RDMAContext


class TransferEngine:

    def __init__(self, dev_name: str, ib_port: int = 1, link_type: str = "Ethernet"):
        self.dev_name = dev_name
        self.ib_port = ib_port
        self.link_type = link_type
        self.links: Dict[int, RDMAContext] = {}
        self.link_exchange_sockets: Dict[int, Tuple] = {}
        self.link_buffer_mr_key: Dict[int, str] = {}

    def init_link(
        self,
        session_id: int,
        mem_pool_tensor: torch.Tensor,
        remote_host: str,
        remote_port: int,
        local_port: int,
    ):
        if session_id in self.links:
            raise KeyError(f"session_id {session_id} already in links")
        rdma_link = RDMAContext(
            dev_name=self.dev_name, ib_port=self.ib_port, link_type=self.link_type
        )
        self.links[session_id] = rdma_link

        mr_key = str(mem_pool_tensor)
        self.link_buffer_mr_key[session_id] = mr_key
        rdma_link.register_torch(mr_key, mem_pool_tensor)

        zmq_ctx = zmq.Context(2)
        send_socket = zmq_ctx.socket(zmq.PUSH)
        send_socket.connect(f"tcp://{remote_host}:{remote_port}")
        recv_socket = zmq_ctx.socket(zmq.PULL)
        recv_socket.bind(f"tcp://*:{local_port}")
        self.link_exchange_sockets[session_id] = (send_socket, recv_socket)

        #
        # Tcp exchange meta
        #
        send_socket, recv_socket = self.link_exchange_sockets[session_id]
        local_rdma_info = rdma_link.get_local_info()
        local_mr_info = rdma_link.get_mr_info(mr_key)
        send_socket.send_pyobj([local_rdma_info, local_mr_info])
        remote_rdma_info, remote_mr_info = recv_socket.recv_pyobj()
        rdma_link.construct(remote_rdma_info)
        rdma_link.register_remote_mr(mr_key, remote_mr_info)

    def stop_link(self, session_id: int):
        if session_id not in self.links:
            raise KeyError(f"session_id {id} not in links")
        self.links[session_id].stop_link()
        del self.links[session_id]

    async def buffered_send_tensor(
        self, session_id: int, tensor: torch.Tensor, send_indices: List[int]
    ):
        """
        Sender gather tensor into a buffer tensor based on send_indices, then sent rdma infos through tcp to receiver.
        Receiver can read it after have those info.
        """

        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")

        start_time = time.time()
        #
        # Put tensors to buffer
        #
        rdma_link = self.links[session_id]
        mr_key = self.link_buffer_mr_key[session_id]
        buffer_tensor = rdma_link.get_mem_pool_tensor(mr_key)
        buffer_tensor = buffer_tensor.view(-1, *tensor.shape[1:])
        scatter_index_tensor = torch.arange(
            len(send_indices), device=buffer_tensor.device
        )
        expand_index_tensor = scatter_index_tensor.view(
            -1, *([1] * (buffer_tensor.dim() - 1))
        ).expand(-1, *buffer_tensor.shape[1:])

        buffer_tensor.scatter_(0, expand_index_tensor, tensor[send_indices])
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        print(f"Gather duration time = {duration} s")

        #
        # Tcp send ready to send
        #
        send_socket, recv_socket = self.link_exchange_sockets[session_id]
        send_socket.send_pyobj("READY")

        # TODO: whether receive ACK based on need

    async def buffered_receive_tensor(
        self, session_id: int, out_tensor: torch.Tensor, receiver_indices: List[int]
    ):
        """
        Receiver read the remote buffer tensor to local buffer tensor, then scatter it to out_tensor.
        """

        if session_id not in self.links:
            raise KeyError(f"session_id {session_id} not in links")

        rdma_link = self.links[session_id]

        mr_key = self.link_buffer_mr_key[session_id]
        buffer_tensor = rdma_link.get_mem_pool_tensor(mr_key)
        buffer_tensor = buffer_tensor.view(-1, *out_tensor.shape[1:])
        local_mr_info = rdma_link.get_mr_info(mr_key)
        remote_mr_info = rdma_link.get_remote_mr_info(mr_key)

        start_time = time.time()
        send_socket, recv_socket = self.link_exchange_sockets[session_id]
        ready_sign = recv_socket.recv_pyobj()

        local_mr_info = rdma_link.get_mr_info(mr_key)
        remote_mr_info = rdma_link.get_remote_mr_info(mr_key)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Prepare rdma meta takes {duration} s")

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
                start_time = time.time()
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
                torch.cuda.synchronize()
                # Success, we should run call back
                end_time = time.time()
                duration = end_time - start_time
                print(f"Scatter takes {duration} s")
                loop.call_soon_threadsafe(future.set_result, code)
            else:
                loop.call_soon_threadsafe(future.set_exception, code)

        read_len = buffer_tensor.numel() * buffer_tensor.itemsize
        start_time = time.time()
        await rdma_link.r_rdma_async(
            mr_key,
            remote_mr_info.offset,
            local_mr_info.offset,
            read_len,
            _scatter_callback,
        )
        await future
        end_time = time.time()
        duration = end_time - start_time
        total_data_bytes = buffer_tensor.numel() * buffer_tensor.itemsize
        total_data_gb = total_data_bytes / (1e9)
        bandwidth = (total_data_gb) / (duration)
        print(
            f"Measure only the r_rdma_async data size = {total_data_gb} GB, total time = {duration} s, {bandwidth=} GB/s"
        )
