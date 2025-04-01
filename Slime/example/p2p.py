import asyncio  # For asynchronous operations

import torch  # For GPU tensor management

from slime import avaliable_nic, RDMAEndpoint  # RDMA endpoint management


devices = avaliable_nic()
assert devices, "No RDMA devices."

# Initialize RDMA endpoint on NIC 'mlx5_bond_1' port 1 using Ethernet transport
initiator = RDMAEndpoint(device_name=devices[0], ib_port=1, link_type="Ethernet")
# Create a zero-initialized CUDA tensor on GPU 0 as local buffer
local_tensor = torch.zeros([16], device="cuda:0", dtype=torch.uint8)
# Register local GPU memory with RDMA subsystem
initiator.register_memory_region(
    mr_identifier="buffer",
    virtual_address=local_tensor.data_ptr(),
    length_bytes=local_tensor.numel() * local_tensor.itemsize,
)


# Initialize target endpoint on different NIC
target = RDMAEndpoint(device_name=devices[-1], ib_port=1, link_type="Ethernet")

# Create a one-initialized CUDA tensor on GPU 1 as remote buffer
remote_tensor = torch.ones([16], device="cuda:1", dtype=torch.uint8)
# Register target's GPU memory
target.register_memory_region(
    mr_identifier="buffer",
    virtual_address=remote_tensor.data_ptr(),
    length_bytes=remote_tensor.numel() * remote_tensor.itemsize,
)

# Establish bidirectional RDMA connection:
# 1. Target connects to initiator's endpoint information
# 2. Initiator connects to target's endpoint information
# Note: Real-world scenarios typically use out-of-band exchange (e.g., via TCP)
target.connect_to(initiator.local_endpoint_info)
initiator.connect_to(target.local_endpoint_info)

# Execute asynchronous batch read operation:
# - Read 8 bytes from target's "buffer" at offset 0
# - Write to initiator's "buffer" at offset 0
# - asyncio.run() executes the async operation synchronously for demonstration
asyncio.run(
    initiator.read_batch_async(
        mr_key="buffer",
        target_offset=[0],
        source_offset=[8],  # Write to start of local buffer
        length=8,
    )
)

# Verify data transfer:
# - Local tensor should now contain data from remote tensor's first 8 elements
# - Remote tensor remains unchanged (RDMA read is non-destructive)

assert torch.all(local_tensor[:8] == 0)
assert torch.all(local_tensor[8:] == 1)
print("Local tensor after RDMA read:", local_tensor)
