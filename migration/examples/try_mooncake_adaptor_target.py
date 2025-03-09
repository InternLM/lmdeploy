import time
import torch
from lmdeploy.pytorch.mooncake_lmdeploy_adaptor import mooncake_lmdeploy_adaptor

a = mooncake_lmdeploy_adaptor()
a.initialize("10.130.8.139:2379", "10.130.8.139:8000")

x = torch.zeros([1000, 1000, 1000, 8], dtype=torch.half, device="cuda")

a.register_local_memory(x.data_ptr(), 0, x.numel() * x.dtype.itemsize + 2)
print(x.data_ptr())

while True:
    time.sleep(15)