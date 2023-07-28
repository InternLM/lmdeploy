# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers.generation.utils import GenerateOutput


class BasicSessionManager:
    """Basic session manager without history."""

    def prepend_history(self, input_ids):
        return input_ids

    def add_to_history(self, output):
        pass


class BasicSessionManagerWithHistory:
    """Basic session manager with chat history."""
    bs = 1

    def __init__(self,
                 max_history=2048,
                 reduce_size=256,
                 start_ids=[1],
                 sep_ids=[13]) -> None:
        assert max_history > reduce_size

        self.start_ids = torch.tensor(start_ids, dtype=torch.long)
        self.sep_ids = torch.tensor(sep_ids, dtype=torch.long)

        assert self.start_ids.ndim == 1
        assert self.sep_ids.ndim == 1

        self.max_history = max(len(start_ids), max_history)
        self.reduce_size = min(reduce_size, max_history - len(start_ids))

        self.new_session()

    def new_session(self):
        self.history_ids = self.start_ids.repeat(self.bs, 1)

    def prepend_history(self, input_ids: torch.Tensor):
        """Prepend history ids to input ids and trim if overlength."""

        input_ids = input_ids.to(self.history_ids.device).long()
        sep_ids = self.sep_ids.to(self.history_ids.device).long().repeat(1, 1)
        input_ids = torch.cat([self.history_ids, sep_ids, input_ids], dim=1)

        if input_ids.shape[1] > self.max_history:
            input_ids = input_ids[:, (self.reduce_size - self.max_history):]
            input_ids[:, :len(self.start_ids)] = self.start_ids.repeat(
                self.bs, 1)
        return input_ids

    def add_to_history(self, output):
        """Save history output ids."""

        # output returned by generator contains both input and output
        if isinstance(output, GenerateOutput):
            self.history_ids = output.sequences
        elif isinstance(output, torch.Tensor):
            self.history_ids = output
        else:
            raise ValueError(f'Unknown output type {type(output)}')
