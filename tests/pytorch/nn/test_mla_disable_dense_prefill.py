# Copyright (c) OpenMMLab. All rights reserved.
"""CPU dispatch tests; kernels are stubbed, not numerical GPU validation."""
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.backends.cuda.attention.sparse_mla import FlashMLAImpl, FlashMLASparseImpl


@pytest.mark.parametrize('value, expected', [(None, False), ('0', False), ('1', True)])
def test_mla_disable_dense_prefill_env(monkeypatch, value, expected):
    monkeypatch.delenv('LMDEPLOY_MLA_DISABLE_DENSE_PREFILL', raising=False)
    if value is not None:
        monkeypatch.setenv('LMDEPLOY_MLA_DISABLE_DENSE_PREFILL', value)
    namespace = runpy.run_path(str(Path(envs.__file__)))
    assert namespace['mla_disable_dense_prefill'] is expected
    if value is not None:
        assert namespace['get_all_envs']()['LMDEPLOY_MLA_DISABLE_DENSE_PREFILL'] == value


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('length', [1, 2048, 2049])
def test_prefill_dispatch(monkeypatch, enabled, length):
    monkeypatch.setattr(envs, 'mla_disable_dense_prefill', enabled)
    impl = object.__new__(FlashMLASparseImpl)
    impl.mla_index_topk = 2048
    calls = []
    output = torch.empty(1)
    query = torch.empty(1, dtype=torch.bfloat16)
    cache = torch.empty(1, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, dtype=torch.int32)
    meta = SimpleNamespace(max_kv_seqlen=length)

    def dense(self, *args, **kwargs):
        calls.append('dense')
        assert kwargs['nsa_indices'] is None
        return output

    def flatten(*args, **kwargs):
        calls.append('flatten')
        assert kwargs['out_dtype'] == torch.bfloat16
        return cache, None

    def sparse(q, kv, ids, metadata):
        calls.append('sparse')
        assert q is query and kv is cache and ids is indices and metadata is meta
        return output

    monkeypatch.setattr(FlashMLAImpl, '_forward_prefill', dense)
    impl._flatten_prefill_kv_cache = flatten
    impl._prefill_sparse = sparse
    use_sparse = enabled or length > impl.mla_index_topk
    assert impl._forward_prefill(query, cache, None, meta,
                                 indices if use_sparse else None) is output
    assert calls == (['flatten', 'sparse'] if use_sparse else ['dense'])
    if use_sparse:
        with pytest.raises(RuntimeError, match='requires DSA top-k indices'):
            impl._forward_prefill(query, cache, None, meta)


@pytest.mark.parametrize('backend', ['flashmla', 'tilelang'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('layer_idx', [4, 5])
def test_indexer_builder_controls_short_prefill_skip(monkeypatch, backend, enabled, layer_idx):
    from lmdeploy.pytorch.models import glm_moe_dsa

    monkeypatch.setattr(envs, 'sparse_mla_backend', backend)
    monkeypatch.setattr(envs, 'mla_disable_dense_prefill', enabled)
    captured = {}

    def build_indexer(*args, **kwargs):
        captured.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(glm_moe_dsa, 'IndexerTopKFP8', build_indexer)
    config = SimpleNamespace(hidden_size=64, index_n_heads=2, index_head_dim=32,
                             qk_rope_head_dim=16, index_topk=2048, q_lora_rank=32,
                             num_hidden_layers=5)
    glm_moe_dsa.GlmMoeDsaIndexer(config, layer_idx=layer_idx, dtype=torch.bfloat16, device='cpu')
    assert captured['allow_short_prefill_scoring_skip'] == (
        layer_idx < 5 and backend != 'tilelang' and not enabled)
