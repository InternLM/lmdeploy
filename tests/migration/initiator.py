import asyncio

import time

import torch

from lmdeploy.migration.engine import TransferEngine
from torch.profiler import profile, ProfilerActivity, record_function


engine = TransferEngine()
link = engine.init_link(
    0,
    dev_name="mlx5_bond_1",
    metadata_endpoint="10.130.8.139:7001",
    ib_port=1,
    link_type="Ethernet",
)


async def main():
    layer_num = 64 * 80
    block_size = 256
    k = torch.zeros(
        [layer_num, 10, block_size, 1, 128], dtype=torch.half, device="cuda"
    )
    v = torch.zeros(
        [layer_num, 10, block_size, 1, 128], dtype=torch.half, device="cuda"
    )
    buffer = torch.zeros([1024 * 1024 * 1024], dtype=torch.half, device="cuda")
    link.register_torch("k", k)
    link.register_torch("v", v)
    link.register_torch("buffer", buffer)
    await link.connect("10.130.8.139:7000")
    length = [block_size * 1 * 128] * layer_num
    offset = [lid * 10 * block_size * 1 * 128 for lid in range(layer_num)]
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_source"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:
    begin = time.time()
    # for l, o in zip(length, offset):
    #     await link.r_rdma_async("k", o, o, l)
    for step in range(10):
        await link.r_rdma_async_batch("k", offset, offset, length)
    #     prof.step()
    end = time.time()
    print(f"bw: {sum(length) * 10 / (end - begin) / 1e9}GBps")
    link.stop()
    link.meta_send.close()
    link.meta_recv.close()


if __name__ == "__main__":
    asyncio.run(main())
