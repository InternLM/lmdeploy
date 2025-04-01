import time

import zmq

import torch
from slime import _slime_c

def test_r_rdma_async():
    zmq_ctx = zmq.Context(2)
    send_socket = zmq_ctx.socket(zmq.PUSH)
    send_socket.connect("tcp://localhost:1212")
    recv_socket = zmq_ctx.socket(zmq.PULL)
    recv_socket.bind("tcp://localhost:2121")

    x = torch.ones([5, 5], device="cuda")

    ctx = _slime_c.rdma_context()

    # Init RDMA
    ctx.init_rdma_context("mlx5_bond_0", 1, "Ethernet")

    mr_key = "target_tensor"
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
    remote_rdma_info = _slime_c.rdma_info(qpn, gid[0], gid[1], gidx, lid, psn, mtu)
    remote_rdma_info.log()

    ctx.modify_qp_to_rtsr(remote_rdma_info)

    time.sleep(10)


def test_batch_r_rdma_async():
    zmq_ctx = zmq.Context(2)
    send_socket = zmq_ctx.socket(zmq.PUSH)
    send_socket.connect("tcp://localhost:1212")
    recv_socket = zmq_ctx.socket(zmq.PULL)
    recv_socket.bind("tcp://localhost:2121")

    batch_size = 2
    x = [torch.ones([5, 5], device="cuda") * i for i in range(batch_size)]

    ctx = _slime_c.rdma_context()

    # Init RDMA
    ctx.init_rdma_context("mlx5_bond_0", 1, "Ethernet")

    # Init Memory Region
    for i in range(len(x)):
        mr_key = "source_tensor_" + str(i)
        
        ctx.register_memory_region(mr_key, x[i].data_ptr(), x[i].numel() * x[i].itemsize)

        # memory key
        local_rkey = ctx.get_r_key(mr_key)

        # exchange data memory info
        send_socket.send_pyobj([x[i].data_ptr(), local_rkey])
        data_ptr, rkey = recv_socket.recv_pyobj()
    
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

    time.sleep(10)


if __name__ == '__main__':
    #test_r_rdma_async()
    test_batch_r_rdma_async()