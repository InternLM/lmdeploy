import pytest
import torch

from lmdeploy.pytorch.kernels import multinomial_sampling


class TestMultinomialSampling:

    @pytest.fixture
    def num_tokens(self, request):
        yield request.param

    @pytest.fixture
    def select_ids(self, request):
        yield request.param

    @pytest.fixture
    def batch_size(self, select_ids):
        yield len(select_ids)

    @pytest.fixture
    def dtype(self, request):
        yield request.param

    @pytest.fixture
    def scores(self, num_tokens, batch_size, select_ids, dtype):
        ret = torch.zeros(batch_size, num_tokens).cuda()
        batch_ids = torch.arange(batch_size).cuda()
        ret[batch_ids, select_ids] = 1
        ret = ret.to(dtype)
        yield ret

    @pytest.fixture
    def seeds(self, batch_size):
        yield torch.randint(1000, 2000, (batch_size, )).cuda()

    @pytest.fixture
    def offsets(self, batch_size):
        yield torch.randint(1000, 2000, (batch_size, )).cuda()

    @pytest.fixture
    def indices(self, scores):
        num_tokens = scores.size(1)
        ret = [torch.randperm(num_tokens) for _ in scores]
        ret = torch.stack(ret, 0).cuda()
        yield ret

    @pytest.fixture
    def gt(self, batch_size, select_ids, indices):
        batch_ids = torch.arange(batch_size).cuda()
        yield indices[batch_ids, select_ids]

    @pytest.mark.parametrize('dtype',
                             [torch.float32, torch.half, torch.bfloat16])
    @pytest.mark.parametrize(['num_tokens', 'select_ids'], [
        (8, (4, 2) * 30),
        (2000, (500, 1500)),
    ],
                             indirect=True)
    def test_multinomial_sampling(self, scores, seeds, offsets, indices, gt):
        output = multinomial_sampling(scores, seeds, offsets, indices)
        torch.testing.assert_close(output, gt)
