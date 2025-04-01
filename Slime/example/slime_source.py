import asyncio
import time

import zmq

import torch
from slime import _slime_c


async def test_r_rdma_async():
    zmq_ctx = zmq.Context(2)
    send_socket = zmq_ctx.socket(zmq.PUSH)
    send_socket.connect("tcp://localhost:2121")
    recv_socket = zmq_ctx.socket(zmq.PULL)
    recv_socket.bind("tcp://localhost:1212")

    x = torch.zeros([5, 5], device="cuda")

    ctx = _slime_c.rdma_context()

    # Init RDMA
    ctx.init_rdma_context("mlx5_bond_0", 1, "Ethernet")

    mr_key = "source_tensor"
    # Init Memory Region
    ctx.register_memory_region(mr_key, x.data_ptr(), x.numel() * x.itemsize)

    # memory key
    local_rkey = ctx.get_r_key(mr_key)
    # rdma info
    local_rdma_info = ctx.get_local_rdma_info()

    # exchange RDMA Info
    send_socket.send_pyobj([
        local_rdma_info.get_gid(), local_rdma_info.gidx, local_rdma_info.lid,
        local_rdma_info.qpn, local_rdma_info.psn, local_rdma_info.mtu,
        x.data_ptr(), local_rkey
    ])
    gid, gidx, lid, qpn, psn, mtu, data_ptr, rkey = recv_socket.recv_pyobj()
    remote_rdma_info = _slime_c.rdma_info(qpn, gid[0], gid[1], gidx, lid, psn,
                                          mtu)
    remote_rdma_info.log()
    ctx.modify_qp_to_rtsr(remote_rdma_info)

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _callback(code):
        print(f"Callback has been successfully called, {code=}")
        loop.call_soon_threadsafe(future.set_result, code)
        #future.set_result("Callback success")
        print(f"Callback after set future")

    ctx.r_rdma_async(data_ptr, x.data_ptr(), 12, mr_key, rkey, _callback)
    ctx.launch_cq_future()
    #ctx.cq_poll_handle()
    future_result = await future
    print(f"{future_result=}")
    ctx.stop_cq_future()

async def test_batch_r_rdma_async():
    zmq_ctx = zmq.Context(2)
    send_socket = zmq_ctx.socket(zmq.PUSH)
    send_socket.connect("tcp://localhost:2121")
    recv_socket = zmq_ctx.socket(zmq.PULL)
    recv_socket.bind("tcp://localhost:1212")

    batch_size = 2
    x = [torch.zeros([5, 5], device="cuda") * i for i in range(batch_size)]

    ctx = _slime_c.rdma_context()

    # Init RDMA
    ctx.init_rdma_context("mlx5_bond_0", 1, "Ethernet")

    
    remote_data_ptrs = []
    test_trans_lengths = []
    source_data_ptrs = []
    source_mr_keys = []
    remote_rkeys = []

    # Init Memory Region
    for i in range(len(x)):
        mr_key = "source_tensor_" + str(i)
        tensor_len = x[i].numel() * x[i].itemsize
        ctx.register_memory_region(mr_key, x[i].data_ptr(), tensor_len)

        # memory key
        local_rkey = ctx.get_r_key(mr_key)
        
        # exchange data memory info
        send_socket.send_pyobj([x[i].data_ptr(), local_rkey])
        data_ptr, rkey = recv_socket.recv_pyobj()
        remote_data_ptrs.append(data_ptr)
        test_len = min(12 + i * 4, tensor_len)
        source_mr_keys.append(mr_key)
        test_trans_lengths.append(test_len)
        source_data_ptrs.append(x[i].data_ptr())
        remote_rkeys.append(rkey)

    # rdma info
    local_rdma_info = ctx.get_local_rdma_info()
    
    # exchange RDMA Info
    send_socket.send_pyobj([
        local_rdma_info.get_gid(), local_rdma_info.gidx, local_rdma_info.lid,
        local_rdma_info.qpn, local_rdma_info.psn, local_rdma_info.mtu
    ])
    gid, gidx, lid, qpn, psn, mtu = recv_socket.recv_pyobj()
    remote_rdma_info = _slime_c.rdma_info(qpn, gid[0], gid[1], gidx, lid, psn, mtu)
    remote_rdma_info.log()
    ctx.modify_qp_to_rtsr(remote_rdma_info)

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _callback(code):
        print(f"Callback has been successfully called, {code=}")
        loop.call_soon_threadsafe(future.set_result, code)
        #future.set_result("Callback success")
        print(f"Callback after set future")

    ctx.batch_r_rdma_async(remote_data_ptrs, source_data_ptrs, test_trans_lengths,
                           source_mr_keys, remote_rkeys, _callback)
    ctx.launch_cq_future()
    #ctx.cq_poll_handle()
    future_result = await future
    print(f"{future_result=}")
    ctx.stop_cq_future()

if __name__ == "__main__":
    #asyncio.run(test_r_rdma_async())
    asyncio.run(test_batch_r_rdma_async())