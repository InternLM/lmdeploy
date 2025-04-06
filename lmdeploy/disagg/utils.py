from dlslime import available_nic


def find_best_rdma_device(rank):
    devices = available_nic()
    return devices[rank % len(devices)]