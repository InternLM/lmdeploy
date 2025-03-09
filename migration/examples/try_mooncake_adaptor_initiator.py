import time
import torch
from lmdeploy.pytorch.mooncake_lmdeploy_adaptor import mooncake_lmdeploy_adaptor

a = mooncake_lmdeploy_adaptor()
a.initialize("10.130.8.139:2379", "10.130.8.139:8001")

x = torch.ones([1000 * 1000 * 1000 * 8]).cuda().half()
data_ptr = x.data_ptr()
print(data_ptr)

a.register_local_memory(data_ptr, 0, x.numel() * x.dtype.itemsize + 2)

begin = time.time()
a.transport_batch("10.130.8.139:8000", data_ptr, [0], [x.numel() * x.dtype.itemsize], [0])
print(time.time() - begin)

print(x.sum())
