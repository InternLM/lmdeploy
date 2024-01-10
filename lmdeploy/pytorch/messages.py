# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch import Tensor

from lmdeploy.messages import EngineGenerationConfig

from .block import LogicalTokenBlocks


class SamplingParam:
    """Sampling parameter."""

    def __init__(
        self,
        top_p: float = 1.0,
        top_k: int = 1,
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

    @classmethod
    def from_gen_config(self, gen_config: EngineGenerationConfig):
        """from gen config."""

        return SamplingParam(top_p=gen_config.top_p,
                             top_k=gen_config.top_k,
                             temperature=gen_config.temperature,
                             repetition_penalty=gen_config.repetition_penalty,
                             ignore_eos=gen_config.ignore_eos,
                             random_seed=gen_config.random_seed,
                             stop_words=gen_config.stop_words,
                             bad_words=gen_config.bad_words)


class MessageStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
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

    def __init__(self, session_id: int, block_size: int) -> None:
        self.session_id = session_id
        self.block_size = block_size
        self.status: MessageStatus = MessageStatus.RUNNING
        self.sequences: Dict[int, SchedulerSequence] = dict()

    def add_sequence(self,
                     token_ids: Tensor,
                     max_output_len: int = 512,
                     sampling_param: SamplingParam = None,
                     adapter_name: str = None) -> 'SchedulerSequence':
        """Add a new message."""
        if not isinstance(token_ids, Tensor):
            token_ids = torch.tensor(token_ids)
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)
        if sampling_param is None:
            sampling_param = SamplingParam()

        seq = SchedulerSequence(seq_id=_new_msg_id(),
                                token_ids=token_ids,
                                session=self,
                                block_size=self.block_size,
                                status=MessageStatus.WAITING,
                                remain_output_len=max_output_len,
                                sampling_param=sampling_param,
                                adapter_name=adapter_name,
                                arrive_time=time.time())
        self.sequences[seq.seq_id] = seq
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
        if not isinstance(token_ids, Tensor):
            token_ids = torch.tensor(token_ids)
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)
        assert seq.session == self

        new_msg = SchedulerSequence(
            seq_id=_new_msg_id(),
            token_ids=token_ids,
            session=self,
            block_size=self.block_size,
            history_token_ids=seq.history_token_ids.copy(),
            remain_output_len=max_output_len,
            sampling_param=sampling_param,
            status=seq.status,
            logical_blocks=seq.logical_blocks.clone(),
            adapter_name=seq.adapter_name,
            arrive_time=time.time(),
            meta=deepcopy(seq.meta))

        self.sequences[new_msg.seq_id] = new_msg
        return new_msg


@dataclass
class SchedulerSequence:
    """Scheduler message."""
    seq_id: int
    token_ids: Tensor
    session: SchedulerSession
    block_size: int
    history_token_ids: list = field(default_factory=list)
    remain_output_len: int = 0
    sampling_param: SamplingParam = field(default_factory=SamplingParam)
    status: MessageStatus = MessageStatus.WAITING
    logical_blocks: LogicalTokenBlocks = None
    sender_id: int = -1
    req_id: int = -1
    adapter_name: str = None
    arrive_time: float = 0.0
    meta: Any = None

    def __post_init__(self):
        self.logical_blocks = self.logical_blocks or LogicalTokenBlocks(
            self.block_size)

    @property
    def history_len(self) -> int:
        """get history length."""
        return len(self.history_token_ids)

    @property
    def session_id(self) -> int:
        """get session id."""
        return self.session.session_id

    def num_logical_tokens(self) -> int:
        """num logitcal tokens."""
        return self.logical_blocks.num_tokens()

    def num_all_tokens(self) -> int:
        """num all tokens."""
        return len(self.token_ids) + self.history_len

    def num_required_tokens(self) -> int:
        """num required tokens."""
        num_all_tokens = self.num_all_tokens()
        num_logical_tokens = self.num_logical_tokens()
        return num_all_tokens - num_logical_tokens

    def num_required_blocks(self) -> int:
        """num required blocks."""
        return self.logical_blocks.num_required_blocks(
            self.num_required_tokens())

    def update_token_ids(self, token_ids: Tensor, update_history: bool = True):
        """Update token ids, old token ids will be added to history."""
        if update_history:
            self.history_token_ids += self.token_ids.tolist()
        if not isinstance(token_ids, Tensor):
            token_ids = self.token_ids.new_tensor(token_ids)
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)
        self.token_ids = token_ids
        self.arrive_time = time.time()

    def set_step(self, step: int):
        """set step."""
        assert step <= self.history_len
        history_token_ids = torch.tensor(self.history_token_ids,
                                         dtype=torch.long)
        new_history_ids = self.history_token_ids[:step]
        new_token_ids = torch.cat([history_token_ids[step:], self.token_ids])
        self.history_token_ids = new_history_ids
        self.token_ids = new_token_ids

        self.logical_blocks.reshape_by_tokens(step)
