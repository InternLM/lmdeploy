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

    def __init__(self, max_seq_len=48, reduce_size=12, bos_id=1) -> None:
        assert max_seq_len > reduce_size

        self.bos_id = bos_id
        self.max_seq_len = max_seq_len
        self.reduce_size = reduce_size
        self.new_session()

    def new_session(self):
        self.history_ids = torch.tensor([[self.bos_id]], dtype=torch.long)

    def prepend_history(self, input_ids: torch.Tensor):
        """Prepend history ids to input ids and trim if overlength."""

        input_ids = input_ids.to(self.history_ids.device).long()
        input_ids = torch.cat([self.history_ids, input_ids], dim=1)

        if input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, (self.reduce_size - self.max_seq_len):]
            input_ids[:, 0] = self.bos_id
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
