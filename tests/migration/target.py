import asyncio

import time

import torch

from lmdeploy.migration import _migration_c
from lmdeploy.migration.engine import TransferEngine
from torch.profiler import profile, ProfilerActivity, record_function


engine = TransferEngine()
link = engine.init_link(
    0,
    dev_name="mlx5_bond_0",
    metadata_endpoint="10.130.8.139:7000",
    ib_port=1,
    link_type="Ethernet",
)


async def handler():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_target"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        while True:
            mr_key, offset, length = await link.meta_recv.recv_pyobj()
            offset = torch.tensor(offset, dtype=torch.int64, device="cuda")
            _migration_c.gather(
                link.memory_pool[mr_key].data_ptr(),
                link.memory_pool["buffer"].data_ptr(),
                length,
                offset.data_ptr(),
                offset.numel(),
            )
            await link.meta_send.send_pyobj("done")
            prof.step()


async def main():
    layer_num = 16000
    block_size = 256
    k = torch.ones([layer_num, 10, block_size, 1, 128], dtype=torch.half, device="cuda")
    v = torch.ones([layer_num, 10, block_size, 1, 128], dtype=torch.half, device="cuda")
    buffer = torch.ones([1024 * 1024 * 1024], dtype=torch.half, device="cuda")
    link.register_torch("k", k)
    link.register_torch("v", v)
    link.register_torch("buffer", buffer)
    await link.connect("10.130.8.139:7001")

    loop = asyncio.get_event_loop()
    loop.create_task(handler())

    # 保持事件循环运行
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
