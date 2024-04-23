import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from lmdeploy.pytorch.kernels.mbgmv import mbgmv_a, mbgmv_b


class TestMBGMV:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def head_size(self):
        yield 64

    @pytest.fixture
    def out_head_size(self):
        yield 32

    @pytest.fixture
    def batch_size(self):
        yield 8

    @pytest.fixture
    def ranks(self):
        yield torch.tensor([2, 4]).cuda()

    @pytest.fixture
    def input(self, batch_size, head_size, dtype):
        x = torch.rand(batch_size, head_size, dtype=dtype).cuda()
        x -= 0.5
        yield x

    @pytest.fixture
    def adapter_ids(self, batch_size, ranks):
        num_ranks = len(ranks)
        ret = torch.randint(0, num_ranks, (batch_size, )).cuda()
        yield ret

    @pytest.fixture
    def scaling(self, ranks):
        yield torch.arange(ranks.size(0)).cuda() + 1

    @pytest.fixture
    def lora_a(self, ranks, head_size, dtype):
        out = []
        for rank in ranks:
            w = torch.rand(head_size, rank, dtype=dtype).cuda()
            w -= 0.5
            out.append(w)
        yield out

    @pytest.fixture
    def lora_b(self, ranks, out_head_size, dtype):
        out = []
        for rank in ranks:
            w = torch.rand(rank, out_head_size, dtype=dtype).cuda()
            w -= 0.5
            out.append(w)
        yield out

    @pytest.fixture
    def page_table(self, ranks):
        total_ranks = sum(ranks)
        index = torch.randperm(total_ranks)
        index = index.split(ranks.tolist())
        yield pad_sequence(index, batch_first=True).cuda()

    @pytest.fixture
    def rank_offset(self, page_table, head_size):
        yield page_table * head_size

    @pytest.fixture
    def paged_lora_a(self, lora_a, ranks, page_table, head_size, dtype):
        num_pages = sum(ranks)
        cache = torch.empty(num_pages, head_size, dtype=dtype).cuda()
        for index, r, w in zip(page_table, ranks, lora_a):
            cache[index[:r]] = w.t()
        yield cache

    @pytest.fixture
    def paged_lora_b(self, lora_b, ranks, page_table, head_size, out_head_size,
                     dtype):
        num_pages = sum(ranks)
        cache = torch.empty(num_pages, head_size, dtype=dtype).cuda()
        for index, r, w in zip(page_table, ranks, lora_b):
            cache[index[:r], :out_head_size] = w
        yield cache

    @pytest.fixture
    def gt(self, input, adapter_ids, lora_a, lora_b, scaling):
        out = []
        for inp, r_id in zip(input, adapter_ids):
            inp = inp.unsqueeze(0)
            l_a = lora_a[r_id]
            l_b = lora_b[r_id]
            s = scaling[r_id]
            out.append(inp @ l_a @ l_b * s)

        yield torch.cat(out)

    def test_mbgmv(self, input, paged_lora_a, paged_lora_b, out_head_size,
                   adapter_ids, scaling, rank_offset, ranks, gt):
        max_rank = rank_offset.size(-1)

        xa = mbgmv_a(input,
                     paged_lora_a,
                     adapter_ids=adapter_ids,
                     rank_offset=rank_offset,
                     ranks=ranks,
                     max_rank=max_rank)

        output = mbgmv_b(xa,
                         paged_lora_b[..., :out_head_size],
                         adapter_ids=adapter_ids,
                         scaling=scaling,
                         rank_offset=rank_offset,
                         ranks=ranks,
                         max_rank=max_rank)
        torch.testing.assert_close(gt, output, atol=2e-3, rtol=1e-5)
