# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from lmdeploy.pytorch_poc.block import LogicalTokenBlock


class MessageStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAP_OUT = enum.auto()
    STOPPED = enum.auto()
    ENDED = enum.auto()
    FINISHED = enum.auto()
    ABORTED = enum.auto()


@dataclass
class SchedulerSession:
    session_id: int
    logical_blocks: Sequence[LogicalTokenBlock] = field(default_factory=list)
    block_table: Dict = field(default_factory=dict)
    status: MessageStatus = MessageStatus.WAITING
    arrive_time: float = 0.0
    history_length: int = 0


@dataclass
class SchedulerMessage:
    token_ids: Sequence
    session_id: int
    status: MessageStatus = MessageStatus.WAITING
    meta: Any = None
