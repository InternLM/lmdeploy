import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.fused_lora import fused_lora


class TestFusedLoRA:

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
    def seq_lens(self, request):
        yield torch.tensor(request.param).cuda()

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
        ret = torch.arange(0, num_seqs) % num_ranks
        ret = ret.cuda()
        yield ret

    @pytest.fixture
    def scaling(self, ranks):
        yield torch.arange(ranks.size(0)).cuda() + 1

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
    def fused_lora_a(self, lora_a):
        yield torch.cat(lora_a, dim=1).t().contiguous()

    @pytest.fixture
    def fused_lora_b(self, lora_b):
        yield torch.cat(lora_b, dim=0).contiguous()

    @pytest.fixture
    def gt(self, input, start_loc, seq_lens, adapter_ids, lora_a, lora_b,
           scaling):
        out = []
        for loc, s_len, r_id in zip(start_loc, seq_lens, adapter_ids):
            inp = input[loc:loc + s_len]
            l_a = lora_a[r_id]
            l_b = lora_b[r_id]
            s = scaling[r_id]
            out.append(inp @ l_a @ l_b * s)

        yield torch.cat(out)

    @pytest.mark.parametrize('seq_lens', [
        (2, 4, 6, 8),
        (1, 1, 1, 1),
    ],
                             indirect=True)
    def test_fused_lora(self, input, fused_lora_a, fused_lora_b, start_loc,
                        seq_lens, adapter_ids, scaling, ranks, gt):
        max_seq_len = max(seq_lens).item()
        max_rank = max(ranks).item()
        rank_offset = ranks.cumsum(0) - ranks

        output = fused_lora(
            input,
            fused_lora_a,
            fused_lora_b,
            scaling=scaling,
            rank_start=rank_offset,
            ranks=ranks,
            seq_start=start_loc,
            seq_lens=seq_lens,
            adapter_ids=adapter_ids,
            max_rank=max_rank,
            max_seqlen=max_seq_len,
        )

        torch.testing.assert_close(gt, output)
