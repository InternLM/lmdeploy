# Slime Transfer Engine

A Peer to Peer RDMA Transfer Engine.

## Usage

### RDMA READ

- Details in [p2p.py](example/p2p.py)

```python
devices = avaliable_nic()
assert devices, "No RDMA devices."

# Initialize RDMA endpoint
initiator = RDMAEndpoint(device_name=devices[0], ib_port=1, link_type="Ethernet")
# Register local GPU memory with RDMA subsystem
local_tensor = torch.tensor(...)
initiator.register_memory_region("buffer", local_tensor...)

# Initialize target endpoint on different NIC
target = RDMAEndpoint(device_name=devices[-1], ib_port=1, link_type="Ethernet")
# Register target's GPU memory
remote_tensor = torch.tensor(...)
target.register_memory_region("buffer", remote_tensor...)

# Establish bidirectional RDMA connection:
# 1. Target connects to initiator's endpoint information
# 2. Initiator connects to target's endpoint information
# Note: Real-world scenarios typically use out-of-band exchange (e.g., via TCP)
target.connect_to(initiator.local_endpoint_info)
initiator.connect_to(target.local_endpoint_info)

# Execute asynchronous batch read operation:
asyncio.run(initiator.async_read_batch("buffer", [0], [8], 8))
```

### SendRecv

- Details in [sendrecv.py](example/sendrecv.py)

#### Sender
``` python
# RDMA init and RDMA Connect just like RDMA Read
...

# RDMA Send
ones = torch.ones([16], dtype=torch.uint8)
endpoint.register_memory_region("buffer", ones.data_ptr(), 16)
asyncio.run(endpoint.send_async("buffer", 0, 8))
```

#### Receiver

``` python
# RDMA init and RDMA Connect just like RDMA Read
...

# RDMA Recv
zeros = torch.zeros([16], dtype=torch.uint8)
endpoint.register_memory_region("buffer", zeros.data_ptr(), 16)
asyncio.run(endpoint.recv_async("buffer", 8, 8))
```

## Build

``` bash
# on CentOS
sudo yum install cppzmq-devel gflags-devel  cmake 

# on Ubuntu
sudo apt install libzmq-dev libgflags-dev cmake

# build from source
mkdir build; cd build
cmake -DBUILD_BENCH=ON -DBUILD_PYTHON=ON ..; make
```

## Benchmark

``` bash
# Target
./bench/transfer_bench                \
  --remote-endpoint=10.130.8.138:8000 \
  --local-endpoint=10.130.8.139:8000  \
  --device-name="mlx5_bond_0"         \
  --mode target                       \
  --block-size=2048000                \
  --batch-size=160

# Initiator
./bench/transfer_bench                \
  --remote-endpoint=10.130.8.139:8000 \ 
  --local-endpoint=10.130.8.138:8000  \ 
  --device-name="mlx5_bond_0"         \
  --mode initiator                    \
  --block-size=16384                  \
  --batch-size=16                     \
  --duration 10
```

### Cross node performance

- H800 with NIC ("mlx5_bond_0"), RoCE v2.

| Batch Size | Block Size (Bytes) | Total Trips | Total Transferred (MiB) | Duration (s) | Average Latency (ms/trip) | Throughput (MiB/s) |
|-----------|-------------------|-------------|-------------------------|-------------|---------------------------|--------------------|
| 160       | 8,192             | 59,391      | 74,238                  | 10.0001     | 0.168377                  | 7,423.8            |
| 160       | 16,384            | 51,144      | 127,860                 | 10.0002     | 0.195530                  | 12,785.8           |
| 160       | 32,768            | 36,614      | 183,070                 | 10.0002     | 0.273124                  | 18,306.7           |
| 160       | 65,536            | 21,021      | 210,210                 | 10.0003     | 0.475729                  | 21,020.4           |
| 160       | 128,000           | 11,419      | 223,027                 | 10.0006     | 0.875789                  | 22,301.3           |
| 160       | 256,000           | 5,839       | 228,085                 | 10.0015     | 1.712880                  | 22,805.2           |
| 160       | 512,000           | 2,956       | 230,937                 | 10.0010     | 3.383300                  | 23,091.3           |
| 160       | 1,024,000         | 1,486       | 232,187                 | 10.0006     | 6.729860                  | 23,217.4           |
| 160       | 2,048,000         | 742         | 231,875                 | 10.0010     | 13.478400                 | 23,185.2           |

