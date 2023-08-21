# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass
from typing import Any, Sequence

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


class SchedulerSession:

    def __init__(self, session_id: int, arrive_time: float = 0.0) -> None:
        self.session_id = session_id
        self.logical_blocks: Sequence[LogicalTokenBlock] = []
        self.block_table = {}
        self.status: MessageStatus = MessageStatus.WAITING
        self.arrive_time: float = arrive_time
        self.history_length: int = 0
        self.token_ids: Sequence = []
        self.history_token_ids = []

    def append_tokens(self, num_tokens: int, block_size: int):

        if len(self.logical_blocks) == 0:
            remain_num_tokens = num_tokens
            next_block_id = 0
        else:
            last_block = self.logical_blocks[-1]
            num_empty_slots = last_block.get_num_empty_slots()
            num_append_slots = min(num_tokens, num_empty_slots)
            last_block.append_tokens(num_append_slots)
            remain_num_tokens = num_tokens - num_append_slots
            next_block_id = last_block.block_id + 1

        for block_id_offset, msg_offset in enumerate(
                range(0, remain_num_tokens, block_size)):
            num_tokens = min(remain_num_tokens - msg_offset, block_size)
            logical_block = LogicalTokenBlock(next_block_id + block_id_offset,
                                              block_size)
            logical_block.append_tokens(num_tokens=num_tokens)
            self.logical_blocks.append(logical_block)


@dataclass
class SchedulerMessage:
    token_ids: Sequence
    session_id: int
    status: MessageStatus = MessageStatus.WAITING
    meta: Any = None
