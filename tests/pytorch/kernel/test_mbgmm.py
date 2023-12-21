import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from lmdeploy.pytorch.kernels.mbgmm import mbgmm_a, mbgmm_b


class TestMBGMM:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def head_size(self):
        yield 32

    @pytest.fixture
    def out_head_size(self):
        yield 16

    @pytest.fixture
    def seq_lens(self):
        yield torch.tensor([2, 4, 6, 8]).cuda()

    @pytest.fixture
    def ranks(self):
        yield torch.tensor([2, 4]).cuda()

    @pytest.fixture
    def start_loc(self, seq_lens):
        yield seq_lens.cumsum(0) - seq_lens

    @pytest.fixture
    def input(self, seq_lens, head_size, dtype):
        total_len = seq_lens.sum()
        yield torch.rand(total_len, head_size, dtype=dtype).cuda()

    @pytest.fixture
    def adapter_ids(self, seq_lens, ranks):
        num_ranks = len(ranks)
        num_seqs = len(seq_lens)
        ret = torch.randint(0, num_ranks, (num_seqs, )).cuda()
        yield ret

    @pytest.fixture
    def lora_a(self, ranks, head_size, dtype):
        out = []
        for rank in ranks:
            w = torch.rand(head_size, rank, dtype=dtype).cuda()
            out.append(w)
        yield out

    @pytest.fixture
    def lora_b(self, ranks, out_head_size, dtype):
        out = []
        for rank in ranks:
            w = torch.rand(rank, out_head_size, dtype=dtype).cuda()
            out.append(w)
        yield out

    @pytest.fixture
    def page_table(self, ranks):
        total_ranks = sum(ranks)
        index = torch.randperm(total_ranks)
        index = index.split(ranks.tolist())
        yield pad_sequence(index, batch_first=True).cuda()

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
    def gt(self, input, start_loc, seq_lens, adapter_ids, lora_a, lora_b):
        out = []
        for loc, s_len, r_id in zip(start_loc, seq_lens, adapter_ids):
            inp = input[loc:loc + s_len]
            l_a = lora_a[r_id]
            l_b = lora_b[r_id]
            out.append(inp @ l_a @ l_b)

        yield torch.cat(out)

    def test_mbgmm(self, input, paged_lora_a, paged_lora_b, out_head_size,
                   start_loc, seq_lens, adapter_ids, page_table, ranks, gt):
        max_seq_len = max(seq_lens).item()
        max_rank = page_table.size(-1)

        xa = mbgmm_a(input,
                     paged_lora_a,
                     b_start_loc=start_loc,
                     b_seq_lens=seq_lens,
                     b_adapter_ids=adapter_ids,
                     rank_page_table=page_table,
                     ranks=ranks,
                     max_seq_len=max_seq_len,
                     max_rank=max_rank)

        output = mbgmm_b(xa,
                         paged_lora_b[..., :out_head_size],
                         b_start_loc=start_loc,
                         b_seq_lens=seq_lens,
                         b_adapter_ids=adapter_ids,
                         rank_page_table=page_table,
                         ranks=ranks,
                         max_seq_len=max_seq_len,
                         max_rank=max_rank)

        torch.testing.assert_close(gt, output)
