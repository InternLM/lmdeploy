from typing import List, Tuple

from dlslime import avaliable_nic, RDMAEndpoint

from lmdeploy.disagg.messages import RemoteEngineConfig


def find_best_rdma_device(rank):
    devices = avaliable_nic()
    return devices[rank % len(devices)]


class EngineEndpoint:
    def __init__(
        self,
        device_name,
        remote_engine_config: RemoteEngineConfig,
        mr_info: List[Tuple[str, int, int]],  # mr_key, addr, length
        ib_port=1,
        link_type: str = "Ethernet",
    ):
        self.remote_engine_config = remote_engine_config
        self.rdma_endpoint = RDMAEndpoint(device_name, ib_port, link_type)

        for mr in mr_info:
            self.rdma_endpoint.register_memory_region(*mr)

    @property
    def local_endpoint_info(self):
        return self.rdma_endpoint.local_endpoint_info
