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
    def q_seqlens(self, device):
        ret = [4, 16, 1, 32]
        ret = torch.tensor(ret, dtype=torch.int32, device=device)
        yield ret

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
    def scores(self, q_seqlens, max_kv_len, device):
        num_tokens = q_seqlens.sum().item()
        yield torch.randn((num_tokens, max_kv_len), device=device)

    @pytest.fixture
    def gt(self, scores, q_seqlens, kv_seqlens, k):
        batch_size = kv_seqlens.numel()
        num_tokens, _ = scores.shape
        topk_indices = torch.empty((num_tokens, k), dtype=torch.int32, device=scores.device)
        topk_indices.fill_(-1)

        start = 0
        for i in range(batch_size):
            q_seqlen = q_seqlens[i].item()
            seqlen = kv_seqlens[i].item()
            tmp_k = min(seqlen, k)
            end = start + q_seqlen
            _, topk_indices[start:end, :seqlen] = torch.topk(scores[start:end, :seqlen],
                                                             tmp_k,
                                                             largest=True,
                                                             sorted=True)
            start = end
        return topk_indices

    def test_bitonic_topk(self, scores, q_seqlens, kv_seqlens, k, gt):
        from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
        out = bitonic_topk(scores, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens, k=k, fill=-1, sorted=True)
        gt[gt < 0] = 0
        out[out < 0] = 0
        gt_score = torch.gather(scores, 1, gt.to(torch.int64))
        out_score = torch.gather(scores, 1, out.to(torch.int64))
        torch.testing.assert_close(gt_score, out_score)
