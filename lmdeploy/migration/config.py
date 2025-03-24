from typing import Dict, Tuple

from pydantic import BaseModel

from . import _migration_c


class RDMAInfo(BaseModel):
    gid: Tuple[int, int]
    gidx: int
    lid: int
    qpn: int
    psn: int
    mtu: int

    @classmethod
    def _from_migration_c(cls, _rdma_info_c: _migration_c.rdma_info):
        return cls(
            gid=_rdma_info_c.get_gid(),
            gidx=_rdma_info_c.gidx,
            lid=_rdma_info_c.lid,
            qpn=_rdma_info_c.qpn,
            psn=_rdma_info_c.psn,
            mtu=_rdma_info_c.mtu,
        )

    def _to_migration_c(self):
        return _migration_c.rdma_info(
            self.qpn, self.gid[0], self.gid[1], self.gidx, self.lid, self.psn, self.mtu
        )


class MemoryRegionInfo(BaseModel):
    addr: int
    offset: int
    r_key: int


class ExchangeInfo(BaseModel):
    rdma_info: RDMAInfo
    mr_info: Dict[str, MemoryRegionInfo]
