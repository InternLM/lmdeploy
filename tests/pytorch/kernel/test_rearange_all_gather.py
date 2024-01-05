import pytest
import torch

from lmdeploy.pytorch.kernels.rearange_all_gather import rearange_all_gather


class TestRearangeAllGather:

    @pytest.fixture
    def seq_lens(self, request):
        yield torch.tensor(request.param, device='cuda')

    @pytest.fixture
    def start_loc(self, seq_lens):
        yield seq_lens.cumsum(0) - seq_lens

    @pytest.fixture
    def ranks(self):
        yield torch.tensor([4, 8]).cuda()

    @pytest.fixture
    def adapter_ids(self, seq_lens, ranks):
        num_ranks = len(ranks)
        num_seqs = len(seq_lens)
        ret = torch.randint(0, num_ranks, (num_seqs, )).cuda()
        yield ret

    @pytest.fixture
    def world_size(self):
        yield 2

    @pytest.fixture
    def input(self, seq_lens, ranks):
        max_rank = max(ranks)
        total_len = seq_lens.sum()
        yield torch.rand(total_len, max_rank).cuda()

    @pytest.fixture
    def rank_per_input(self, seq_lens, ranks, adapter_ids):
        token_adapter_ids = [
            torch.full((slen, ), ada_id)
            for slen, ada_id in zip(seq_lens, adapter_ids)
        ]
        token_adapter_ids = torch.cat(token_adapter_ids).cuda()
        yield ranks[token_adapter_ids]

    @pytest.fixture
    def valid_mask(self, rank_per_input, seq_lens, ranks):
        max_rank = max(ranks)
        total_len = seq_lens.sum()
        mask = torch.zeros(total_len, max_rank).to(bool)
        for r, m in zip(rank_per_input, mask):
            m[:r] = True
        yield mask.cuda()

    @pytest.fixture
    def gt(self, input, rank_per_input, ranks, world_size):
        max_rank = max(ranks)
        pranks = rank_per_input // world_size
        pmax_rank = max_rank // world_size
        output = torch.empty_like(input)
        for pr, inp, out in zip(pranks, input, output):
            pindex = torch.arange(pr).cuda()
            index = [pindex + ws * pmax_rank for ws in range(world_size)]
            index = torch.cat(index)
            out[:index.size(0)] = inp[index]
        yield output

    @pytest.mark.parametrize('seq_lens', [[30, 50, 70, 90], [1, 1, 1, 1]],
                             indirect=True)
    def test_gather(self, input, start_loc, seq_lens, adapter_ids, ranks,
                    world_size, gt, valid_mask):
        max_seq_len = max(seq_lens)
        output = rearange_all_gather(input,
                                     start_loc,
                                     seq_lens,
                                     adapter_ids,
                                     ranks,
                                     world_size,
                                     max_seq_len=max_seq_len)
        output = output.where(valid_mask, output.new_tensor(0))
        gt = gt.where(valid_mask, gt.new_tensor(0))
        torch.testing.assert_close(output, gt)
