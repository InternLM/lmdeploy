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


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_decode_keeps_existing_flashmla_dispatch(monkeypatch, enabled, dtype):
    monkeypatch.setattr(envs, 'mla_disable_dense_prefill', enabled)
    impl = object.__new__(FlashMLASparseImpl)
    calls = []
    impl._decoding_sparse_bf16 = lambda *args: calls.append('bf16')
    impl._decoding_sparse_fp8 = lambda *args: calls.append('fp8')
    cache = torch.empty(1, dtype=dtype)
    impl._forward_decoding(torch.empty(1), cache, None, torch.zeros(1, dtype=torch.int32))
    assert calls == (['bf16'] if dtype == torch.bfloat16 else ['fp8'])
    with pytest.raises(RuntimeError, match='requires DSA top-k indices'):
        impl._forward_decoding(torch.empty(1), cache, None)


def test_bf16_prefill_and_decode_share_flashmla_entrypoint():
    impl = object.__new__(FlashMLASparseImpl)
    impl.scale = 0.125
    impl._get_max_q_seqlen = lambda *args: 1
    mapped = torch.zeros(1, 1, 1, dtype=torch.int32)
    impl.index_mapper = SimpleNamespace(
        map_flat_prefill=lambda *args: mapped,
        map_strided_decode=lambda *args: mapped)
    calls = []
    output = torch.empty(1, 64, 512, dtype=torch.bfloat16)

    def flashmla(q, kv, ids, sm_scale):
        calls.append((q, kv, ids, sm_scale))
        return (output, )

    impl.flash_mla_sparse_fwd = flashmla
    query = torch.empty(1, 64, 576, dtype=torch.bfloat16)
    cache = torch.empty(1, 64, 1, 576, dtype=torch.bfloat16)
    ids = torch.zeros(1, 1, dtype=torch.int32)
    meta = SimpleNamespace(q_seqlens=torch.tensor([1]), cu_seqlens_k=torch.tensor([0, 1]),
                           block_offsets=torch.tensor([[0]]))
    prefill = impl._prefill_sparse(query, cache.flatten(0, 1), ids, meta)
    decode = impl._decoding_sparse_bf16(query, cache, ids, meta)
    assert len(calls) == 2
    assert all(call[0] is query and call[3] == impl.scale for call in calls)
    assert prefill.data_ptr() == decode.data_ptr() == output.data_ptr()


@pytest.mark.parametrize('model_name', ['deepseek', 'glm'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('layer_idx', [4, 5])
def test_indexer_builder_controls_short_prefill_skip(monkeypatch, model_name, enabled, layer_idx):
    from lmdeploy.pytorch.models import deepseek_v32, glm_moe_dsa

    module, cls = ((deepseek_v32, deepseek_v32.Indexer) if model_name == 'deepseek'
                   else (glm_moe_dsa, glm_moe_dsa.GlmMoeDsaIndexer))
    monkeypatch.setattr(envs, 'mla_disable_dense_prefill', enabled)
    captured = {}

    def build_indexer(*args, **kwargs):
        captured.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(module, 'IndexerTopKFP8', build_indexer)
    config = SimpleNamespace(hidden_size=64, index_n_heads=2, index_head_dim=32,
                             qk_rope_head_dim=16, index_topk=2048, q_lora_rank=32,
                             num_hidden_layers=5)
    cls(config, layer_idx=layer_idx, dtype=torch.bfloat16, device='cpu')
    assert captured['allow_short_prefill_scoring_skip'] == (
        layer_idx < 5 and (model_name == 'deepseek' or not enabled))
