from lmdeploy.messages import EngineRole
from typing import List
from dataclasses import dataclass


@dataclass
class EngineSnapshot:
    # role
    role: EngineRole

    # server config
    endpoint: str

    # parallel config
    dp_size: int
    tp_size: int
    pp_size: int

    # cache config
    block_size: int
    total_gpu_blocks: int

    # transferEngine config
    segment_ids: List[str]

    def verify(self):
        assert self.dp_size * self.tp_size * self.pp_size == len(self.segment_ids)
    

@dataclass
class EngineSnapshotManager:
    engine_snapshots: List[EngineSnapshot]

