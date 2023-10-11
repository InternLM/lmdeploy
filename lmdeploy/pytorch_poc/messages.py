# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor

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
    ABORTED = enum.auto()


_SEQ_COUNT = 0


def _new_msg_id():
    """get a new message id."""
    global _SEQ_COUNT
    seq_id = _SEQ_COUNT
    _SEQ_COUNT += 1
    return seq_id


class SchedulerSession:
    """Scheduler session."""

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        self.messages: Dict[SchedulerSequence] = dict()

    def add_sequence(
            self,
            token_ids: Tensor,
            max_output_len: int = 512,
            sampling_param: SamplingParam = None) -> 'SchedulerSequence':
        """Add a new message."""
        if sampling_param is None:
            sampling_param = SamplingParam()

        seq = SchedulerSequence(seq_id=_new_msg_id(),
                                token_ids=token_ids,
                                session=self,
                                status=MessageStatus.WAITING,
                                remain_output_len=max_output_len,
                                sampling_param=sampling_param,
                                arrive_time=time.time())
        self.messages[seq.seq_id] = seq
        return seq

    def fork_sequence(
            self,
            token_ids: Tensor,
            seq: 'SchedulerSequence',
            max_output_len: int = 512,
            sampling_param: SamplingParam = None) -> 'SchedulerSequence':
        """Fork a new message from exist message."""
        if sampling_param is None:
            sampling_param = deepcopy(seq.sampling_param)
        assert seq.session == self

        new_msg = SchedulerSequence(
            seq_id=_new_msg_id(),
            token_ids=token_ids,
            session=self,
            history_token_ids=seq.history_token_ids.clone(),
            status=seq.status,
            remain_output_len=max_output_len,
            logical_blocks=deepcopy(seq.logical_blocks),
            sampling_param=sampling_param,
            arrive_time=time.time(),
            meta=deepcopy(seq.meta))

        self.messages[new_msg.seq_id] = new_msg
        return new_msg


@dataclass
class SchedulerSequence:
    """Scheduler message."""
    seq_id: int
    token_ids: Tensor
    session: SchedulerSession
    history_token_ids: Tensor = field(default=torch.empty(0, dtype=torch.long))
    remain_output_len: int = 0
    sampling_param: SamplingParam = field(default_factory=SamplingParam)
    status: MessageStatus = MessageStatus.WAITING
    logical_blocks: Sequence[LogicalTokenBlock] = field(default_factory=list)
    arrive_time: float = 0.0
    meta: Any = None

    @property
    def history_len(self) -> int:
        """get history length."""
        return len(self.history_token_ids)

    @property
    def session_id(self) -> int:
        """get session id."""
        return len(self.session.session_id)

    def num_logical_tokens(self) -> int:
        if len(self.logical_blocks) == 0:
            return 0
        else:
            return sum(block.num_tokens for block in self.logical_blocks)

    def num_required_tokens(self) -> int:
        num_all_tokens = len(self.token_ids) + self.history_len
        num_logical_tokens = self.num_logical_tokens()
        return num_all_tokens - num_logical_tokens

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

    def update_token_ids(self, token_ids: Tensor):
        """Update token ids, old token ids will be added to history."""
        self.history_token_ids = torch.cat([self.history_token_ids, token_ids])
        self.token_ids = token_ids
        self.arrive_time = time.time()
