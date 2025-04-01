import asyncio

import zmq
import torch

from slime import RDMAEndpoint

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="send", type=str)

parser.add_argument("--device-name", default="mlx5_bond_0", type=str)

parser.add_argument("--remote-host", default="localhost", type=str)
parser.add_argument("--local-host", default="localhost", type=str)
parser.add_argument("--remote-port", default=8000, type=int)
parser.add_argument("--local-port", default=8001, type=int)

args = parser.parse_args()

if __name__ == "__main__":
    zmq_ctx = zmq.Context(2)
    zmq_send = zmq_ctx.socket(zmq.PUSH)
    zmq_recv = zmq_ctx.socket(zmq.PULL)
    zmq_send.connect(f"tcp://{args.remote_host}:{args.remote_port}")
    zmq_recv.bind(f"tcp://{args.local_host}:{args.local_port}")

    endpoint = RDMAEndpoint(device_name=args.device_name)
    zmq_send.send_string(endpoint.local_endpoint_info)
    remote_endpoint_info = zmq_recv.recv_string()
    endpoint.connect_to(remote_endpoint_info)

    if args.mode == "send":
        ones = torch.ones([16], dtype=torch.uint8)
        endpoint.register_memory_region("buffer", ones.data_ptr(), 16)
        asyncio.run(endpoint.send_async("buffer", 0, 8))
    elif args.mode == "recv":
        zeros = torch.zeros([16], dtype=torch.uint8)
        print(f"before recv: {zeros}")
        endpoint.register_memory_region("buffer", zeros.data_ptr(), 16)
        asyncio.run(endpoint.recv_async("buffer", 8, 8))
        print(f"after recv: {zeros}")
        assert torch.all(zeros[8:] == 1)
        assert torch.all(zeros[:8] == 0)

