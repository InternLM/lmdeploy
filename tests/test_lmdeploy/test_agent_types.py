# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for agent_types module."""

import numpy as np
import torch

import pytest

try:
    from lmdeploy.pytorch.engine.model_agent.agent_types import BatchedLogProbs, BatchedOutputs
except ImportError:
    pytest.skip("Missing dependencies", allow_module_level=True)


class TestBatchedLogProbs:

    def test_init(self):
        """Test basic initialization."""
        vals = torch.tensor([0.1, 0.2, 0.3])
        indices = torch.tensor([1, 2, 3])
        blp = BatchedLogProbs(vals=vals, indices=indices)
        assert blp.vals is vals
        assert blp.indices is indices

    def test_to_numpy_bf16(self):
        """Test to_numpy with bfloat16."""
        vals = torch.tensor([0.1, 0.2], dtype=torch.bfloat16)
        indices = torch.tensor([1, 2])
        blp = BatchedLogProbs(vals=vals, indices=indices)
        np_blp = blp.to_numpy()
        assert isinstance(np_blp.vals, torch.Tensor)
        assert isinstance(np_blp.indices, np.ndarray)

    def test_to_tensor_numpy(self):
        """Test to_tensor from numpy."""
        vals_np = np.array([0.1, 0.2])
        indices_np = np.array([1, 2])
        blp = BatchedLogProbs(vals=vals_np, indices=indices_np)
        tensor_blp = blp.to_tensor()
        assert isinstance(tensor_blp.vals, torch.Tensor)
        assert isinstance(tensor_blp.indices, torch.Tensor)

    def test_to_tensor_torch(self):
        """Test to_tensor from torch tensor."""
        vals = torch.tensor([0.1, 0.2])
        indices = torch.tensor([1, 2])
        blp = BatchedLogProbs(vals=vals, indices=indices)
        tensor_blp = blp.to_tensor()
        assert tensor_blp.vals is vals
        assert tensor_blp.indices is indices


class TestBatchedOutputs:

    def test_init_defaults(self):
        """Test initialization with default values."""
        next_tokens = torch.tensor([1, 2, 3])
        stopped = torch.tensor([False, True, False])
        outputs = BatchedOutputs(next_token_ids=next_tokens, stopped=stopped)
        assert outputs.next_token_ids is next_tokens
        assert outputs.stopped is stopped
        assert outputs.stop_pos is None
        assert outputs.logprobs is None
        assert outputs.extra_outputs is None

    def test_to_numpy(self):
        """Test to_numpy conversion."""
        next_tokens = torch.tensor([1, 2, 3])
        stopped = torch.tensor([False, True, False])
        outputs = BatchedOutputs(next_token_ids=next_tokens, stopped=stopped)
        np_outputs = outputs.to_numpy()
        assert isinstance(np_outputs.next_token_ids, np.ndarray)
        assert isinstance(np_outputs.stopped, np.ndarray)

    def test_to_tensor(self):
        """Test to_tensor from numpy."""
        next_tokens = np.array([1, 2, 3])
        stopped = np.array([False, True, False])
        outputs = BatchedOutputs(next_token_ids=next_tokens, stopped=stopped)
        tensor_outputs = outputs.to_tensor()
        assert isinstance(tensor_outputs.next_token_ids, torch.Tensor)
        assert isinstance(tensor_outputs.stopped, torch.Tensor)

    def test_with_logprobs(self):
        """Test outputs with logprobs attached."""
        next_tokens = torch.tensor([1])
        stopped = torch.tensor([False])
        logprobs = BatchedLogProbs(vals=torch.tensor([0.1]), indices=torch.tensor([1]))
        outputs = BatchedOutputs(
            next_token_ids=next_tokens,
            stopped=stopped,
            logprobs=logprobs,
        )
        assert outputs.logprobs is logprobs
        assert float(outputs.logprobs.vals[0]) == 0.1

    def test_logits_roundtrip(self):
        """Test logits field roundtrip through to_cpu/to_tensor."""
        next_tokens = torch.tensor([1])
        stopped = torch.tensor([False])
        logits = torch.tensor([[0.1, 0.2, 0.3]])
        outputs = BatchedOutputs(
            next_token_ids=next_tokens,
            stopped=stopped,
            logits=logits,
        )
        assert outputs.logits is logits

    def test_new_token_timestamp(self):
        """Test new_token_timestamp field."""
        outputs = BatchedOutputs(
            next_token_ids=torch.tensor([1]),
            stopped=torch.tensor([False]),
            new_token_timestamp=1234567890,
        )
        assert outputs.new_token_timestamp == 1234567890