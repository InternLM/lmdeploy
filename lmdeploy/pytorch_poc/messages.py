# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass, field
from typing import Any, List, Sequence

from lmdeploy.pytorch_poc.block import LogicalTokenBlock


class SamplingParam:
    """Sampling parameter."""

    def __init__(
        self,
        top_p: float = 0.8,
        top_k: int = None,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        random_seed: int = None,
        stop_words: List[int] = None,
        bad_words: List[int] = None,
    ):
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.ignore_eos = ignore_eos
        self.random_seed = random_seed
        self.stop_words = stop_words
        self.bad_words = bad_words


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
    """Scheduler session."""

    def __init__(self, session_id: int, arrive_time: float = 0.0) -> None:
        self.session_id = session_id
        self.logical_blocks: Sequence[LogicalTokenBlock] = []
        self.block_table = {}
        self.status: MessageStatus = MessageStatus.WAITING
        self.arrive_time: float = arrive_time
        self.history_length: int = 0

    def append_tokens(self, num_tokens: int, block_size: int):
        """Append new tokens, update logical blocks.

        Args:
            num_tokens (int): Number of tokens.
            block_size (int): Size of block.
        """
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
    """Scheduler message."""

    token_ids: Sequence
    session_id: int
    status: MessageStatus = MessageStatus.WAITING
    max_request_output_len: int = 0
    request_output_len: int = 0
    meta: Any = None
    req_id: int = 0
    sampling_param: SamplingParam = field(default_factory=SamplingParam)
