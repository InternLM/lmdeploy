import pytest
import torch


class TestBitonicTopk:

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def k(self):
        yield 2048

    @pytest.fixture
    def kv_seqlens(self, device):
        ret = [1024, 2048, 4096, 4096 + 133]
        ret = torch.tensor(ret, dtype=torch.int32, device=device)
        yield ret

    @pytest.fixture
    def batch_size(self, kv_seqlens):
        return kv_seqlens.numel()

    @pytest.fixture
    def max_kv_len(self, kv_seqlens):
        return kv_seqlens.max().item()

    @pytest.fixture
    def scores(self, batch_size, max_kv_len, device):
        yield torch.randn((batch_size, max_kv_len), device=device)

    @pytest.fixture
    def gt(self, scores, kv_seqlens, k):
        batch_size, _ = scores.shape
        topk_indices = torch.empty((batch_size, k), dtype=torch.int32, device=scores.device)
        topk_indices.fill_(-1)

        for i in range(batch_size):
            seqlen = kv_seqlens[i].item()
            tmp_k = min(seqlen, k)
            _, topk_indices[i, :seqlen] = torch.topk(scores[i, :seqlen], tmp_k, largest=True, sorted=True)
        return topk_indices

    def test_bitonic_topk(self, scores, kv_seqlens, k, gt):
        from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
        out = bitonic_topk(scores, kv_seqlens=kv_seqlens, k=k, fill=-1)
        gt[gt < 0] = 0
        out[out < 0] = 0
        gt_score = torch.gather(scores, 1, gt.to(torch.int64))
        out_score = torch.gather(scores, 1, out.to(torch.int64))
        torch.testing.assert_close(gt_score, out_score)
