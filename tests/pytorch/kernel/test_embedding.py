import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from lmdeploy.pytorch.distributed import DefaultContext
from lmdeploy.pytorch.nn import ParallelEmbedding


def parrall_emb(rank: int, world_size: int, vocab_size: int, feat_size: int, padding_idx: int, dtype: torch.dtype,
                x: torch.Tensor, weight: torch.Tensor):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    gpu_group = dist.new_group(ranks=list(range(world_size)), backend='nccl')

    DefaultContext.mlp_tp_group.rank = rank
    DefaultContext.dist_config.mlp_tp = world_size
    DefaultContext.mlp_tp_group.gpu_group = gpu_group

    model = ParallelEmbedding(vocab_size=vocab_size,
                              hidden_size=feat_size,
                              padding_idx=None,
                              dtype=dtype,
                              is_tp=True,
                              device=torch.device(type='cuda', index=rank))
    token_emb = nn.Embedding(vocab_size,
                             feat_size,
                             padding_idx=padding_idx,
                             dtype=dtype,
                             device=torch.device(type='cuda', index=rank))

    weight = weight.to(torch.device(type='cuda', index=rank))
    token_emb.weight.data.copy_(weight)
    model.weight_loader(model.weight, weight)

    input = x.to(torch.device(type='cuda', index=rank))

    with torch.inference_mode():
        out = model(input)
        gt = token_emb(input)

    torch.testing.assert_close(out, gt)

    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


class TestEmbedding:

    @pytest.fixture
    def vocab_size(self, request):
        yield request.param

    @pytest.fixture
    def feat_size(self, request):
        yield request.param

    @pytest.fixture
    def padding_idx(self, request):
        yield request.param

    @pytest.fixture
    def dtype(self, request):
        yield request.param

    @pytest.fixture
    def tp(self, request):
        yield request.param

    @pytest.fixture
    def seqlen(self, request):
        yield request.param

    @pytest.fixture
    def weight(self, vocab_size, feat_size, dtype):
        yield torch.rand(vocab_size, feat_size, dtype=dtype)

    @pytest.fixture
    def x(self, seqlen, vocab_size):
        yield torch.randint(low=0, high=vocab_size, size=(seqlen, ), dtype=torch.int32)

    @pytest.mark.parametrize('vocab_size', [65576, 65533, 3333], indirect=True)
    @pytest.mark.parametrize('feat_size', [4096, 768], indirect=True)
    @pytest.mark.parametrize('padding_idx', [None], indirect=True)
    @pytest.mark.parametrize('seqlen', [1024, 1011, 128], indirect=True)
    @pytest.mark.parametrize('tp', [2], indirect=True)
    @pytest.mark.parametrize('dtype', [torch.bfloat16], indirect=True)
    def test_embedding(self, vocab_size, feat_size, padding_idx, seqlen, tp, dtype, x, weight):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

        world_size = tp
        processes = []
        mp.set_start_method('spawn', force=True)

        for rank in range(world_size):
            p = mp.Process(target=parrall_emb,
                           args=(rank, world_size, vocab_size, feat_size, padding_idx, dtype, x, weight))
            processes.append(p)
            p.start()
            time.sleep((0.5))

        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
