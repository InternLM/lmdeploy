from typing import Tuple, List
from pydantic import BaseModel


class MemoryRegionInfo(BaseModel):
    addr: int
    offset: int
    r_key: int 

class RDMAInfo(BaseModel):
    gid: Tuple[int, int]
    gidx: int
    lid: int
    qpn: int
    psn: int
    mtu: int


class ExchangeInfo(BaseModel):
    rdma_info: RDMAInfo
    mr_info: List[MemoryRegionInfo]
