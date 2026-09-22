# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.backends.cuda.attention.sparse_mla import FlashMLAImpl, FlashMLASparseImpl
from lmdeploy.pytorch.models import glm_moe_dsa


@pytest.mark.parametrize('enabled,length,sparse', [(False, 2048, False), (False, 2049, True), (True, 2048, True)])
def test_prefill_dispatch(monkeypatch, enabled, length, sparse):
    monkeypatch.setattr(envs, 'mla_disable_dense_prefill', enabled)
    impl = object.__new__(FlashMLASparseImpl)
    impl.mla_index_topk = 2048
    dense = Mock()
    monkeypatch.setattr(FlashMLAImpl, '_forward_prefill', dense)
    cache = torch.empty(1, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, dtype=torch.int32)
    meta = SimpleNamespace(max_kv_seqlen=length)
    impl._flatten_prefill_kv_cache = Mock(return_value=(cache, None))
    impl._prefill_sparse = Mock()
    output = impl._forward_prefill(cache, cache, None, meta, indices)
    assert output is (impl._prefill_sparse.return_value if sparse else dense.return_value)
    assert dense.call_count == int(not sparse)
    assert impl._flatten_prefill_kv_cache.call_count == impl._prefill_sparse.call_count == int(sparse)


@pytest.mark.parametrize('enabled,backend,layer,skip', [
    (False, 'flashmla', 4, True), (True, 'flashmla', 4, False),
    (False, 'tilelang', 4, False), (False, 'flashmla', 5, False),
])
def test_glm_indexer_requires_indices(monkeypatch, enabled, backend, layer, skip):
    monkeypatch.setattr(envs, 'mla_disable_dense_prefill', enabled)
    monkeypatch.setattr(envs, 'sparse_mla_backend', backend)
    builder = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(glm_moe_dsa, 'IndexerTopKFP8', builder)
    config = SimpleNamespace(hidden_size=64, index_n_heads=2, index_head_dim=32,
                             qk_rope_head_dim=16, index_topk=2048, q_lora_rank=32, num_hidden_layers=5)
    glm_moe_dsa.GlmMoeDsaIndexer(config, layer_idx=layer, dtype=torch.bfloat16, device='cpu')
    assert builder.call_args.kwargs['allow_short_prefill_scoring_skip'] is skip
