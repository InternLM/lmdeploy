import enum
from typing import Sequence, Dict, Any
from dataclasses import dataclass, field


class MessageStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    STOPPED = enum.auto()
    ENDED = enum.auto()
    FINISHED = enum.auto()


@dataclass
class SchedulerSession:
    session_id: int
    block_table: Dict = field(default_factory=dict)


@dataclass
class SchedulerMessage:
    token_ids: Sequence
    session_id: int
    status: MessageStatus = MessageStatus.WAITING
    meta: Any = None
