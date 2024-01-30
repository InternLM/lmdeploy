import pytest
import torch

from lmdeploy.pytorch.tools.layout_convert import (batch_tensor,
                                                   continuous_tensor)


class TestContinuous:

    @pytest.fixture
    def batched_tensor(self):
        yield torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8], [9, 10, 0, 0,
                                                               0]])

    @pytest.fixture
    def seq_len(self):
        yield torch.tensor([3, 5, 2])

    @pytest.fixture
    def conti_tensor(self):
        yield torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    def test_conti_tensor(self, batched_tensor, seq_len, conti_tensor):
        conti_out = continuous_tensor(batched_tensor, seq_len)
        torch.testing.assert_close(conti_out, conti_tensor)

        batched_out = batch_tensor(conti_tensor, seq_len)
        torch.testing.assert_close(batched_out, batched_tensor)
