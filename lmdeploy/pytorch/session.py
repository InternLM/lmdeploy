# Copyright (c) OpenMMLab. All rights reserved.
import logging

import torch
from transformers.generation.utils import ModelOutput

logger = logging.getLogger(__name__)


class BasicSessionManager:
    """Basic session manager without history."""

    def prepend_history(self, input_ids):
        return input_ids

    def add_to_history(self, output):
        pass


class BasicSessionManagerWithHistory:
    """Basic session manager with chat history.

    Args:
        max_session_len (int): Maximum number of tokens allowed for all chat sessions.
        reduce_size (int): Number of tokens to be trimmed when reaching maximum
            session length. Default: 256.
        start_ids (list[int]): Sequences of ids at the start of the chat session.
        sep_ids (list[int]): Sequences of ids separating chat sessions.
    """   # noqa: E501
    bs = 1

    def __init__(self,
                 max_session_len=2048,
                 reduce_size=256,
                 start_ids=[1],
                 sep_ids=[13]) -> None:

        self.start_ids = torch.tensor(start_ids, dtype=torch.long)
        self.sep_ids = torch.tensor(sep_ids, dtype=torch.long)

        assert self.start_ids.ndim == 1
        assert self.sep_ids.ndim == 1

        self.max_session_len = max(len(start_ids), max_session_len)
        self.reduce_size = min(reduce_size, max_session_len - len(start_ids))

        assert self.max_session_len > self.reduce_size

        self.new_session()

    def new_session(self):
        self.history_ids = self.start_ids.repeat(self.bs, 1)

    def prepend_history(self, input_ids: torch.Tensor):
        """Prepend history ids to input ids and trim if over-length."""

        input_ids = input_ids.to(self.history_ids.device).long()
        sep_ids = self.sep_ids.to(self.history_ids.device).long().repeat(1, 1)
        input_ids = torch.cat([self.history_ids, sep_ids, input_ids], dim=1)

        if input_ids.shape[1] > self.max_session_len:
            input_ids = input_ids[:,
                                  (self.reduce_size - self.max_session_len):]
            input_ids[:, :len(self.start_ids)] = self.start_ids.repeat(
                self.bs, 1)
        return input_ids

    def add_to_history(self, output):
        """Save history output ids.

        Note:
            Output returned by HuggingFace generator contains both input
            and output ids.
        """

        if isinstance(output, ModelOutput):
            self.history_ids = output.sequences
        elif isinstance(output, torch.Tensor):
            self.history_ids = output
        else:
            raise ValueError(f'Unknown output type {type(output)}')
