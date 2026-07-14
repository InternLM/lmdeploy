# Copyright (c) OpenMMLab. All rights reserved.
"""Data types for model agent.

Extracted from agent.py to reduce module size.
"""
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from lmdeploy.pytorch.strategies.base.model_agent import ExtraOutputs


@dataclass
class BatchedLogProbs:
    vals: torch.Tensor
    indices: torch.Tensor

    def to_cpu(self):
        """To cpu."""
        return BatchedLogProbs(vals=self.vals.cpu().detach(), indices=self.indices.cpu().detach())

    def to_numpy(self):
        """To numpy."""
        if self.vals.dtype == torch.bfloat16:
            np_vals = self.vals
        else:
            np_vals = self.vals.detach().numpy()
        return BatchedLogProbs(vals=np_vals, indices=self.indices.detach().numpy())

    def to_tensor(self):
        """To tensor."""
        if isinstance(self.vals, torch.Tensor):
            vals = self.vals
        else:
            vals = torch.from_numpy(self.vals)
        return BatchedLogProbs(vals=vals, indices=torch.from_numpy(self.indices))


@dataclass
class BatchedOutputs:
    next_token_ids: torch.Tensor
    stopped: torch.Tensor
    stop_pos: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    model_metas: list[dict[str, Any]] = None
    logprobs: BatchedLogProbs | None = None
    new_token_timestamp: int = 0
    extra_outputs: ExtraOutputs | None = None
    all_routed_experts: torch.Tensor | None = None

    def to_cpu(self):
        """To cpu."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach()
            elif hasattr(v, 'to_cpu'):
                v = v.to_cpu()
            out[k] = v
        return BatchedOutputs(**out)

    def to_numpy(self):
        """To numpy."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor) and v.dtype != torch.bfloat16:
                v = v.detach().numpy()
            elif hasattr(v, 'to_numpy'):
                v = v.to_numpy()
            out[k] = v
        return BatchedOutputs(**out)

    def to_tensor(self):
        """To tensor."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif hasattr(v, 'to_tensor'):
                v = v.to_tensor()
            out[k] = v
        return BatchedOutputs(**out)