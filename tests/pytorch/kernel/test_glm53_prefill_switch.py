# Copyright (c) OpenMMLab. All rights reserved.
"""Regression coverage for GLM53's model-specific dense/sparse prefill
dispatch."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Route(Exception):
    pass


class _Tensor:

    def size(self, dim):
        return 8 if dim == 0 else 64

    def __getitem__(self, key):
        return self


class _Attention:
    kv_lora_rank = 512
    index_topk = 2048

    def _kpool_indices(self, *args, **kwargs):
        self.return_indices = kwargs['return_indices']
        self.topk_buffer = kwargs['topk_indices_buffer']
        return 'causal_indices' if self.return_indices else None

    def _forward_prefill_mha(self, *args):
        raise _Route('dense')

    def _absorbed_query(self, *args):
        return _Tensor()

    @property
    def decode_attn_fwd(self):
        return self

    def forward(self, *args, **kwargs):
        assert kwargs['nsa_indices'] == 'causal_indices'
        raise _Route('sparse')


@pytest.fixture
def forward():
    # Compile the actual production method without importing CUDA-only optional dependencies.
    import lmdeploy
    source = Path(lmdeploy.__file__).parent / 'pytorch/models/glm5_next.py'
    tree = ast.parse(source.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Glm5NextSparseAttention')
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_forward_attention')
    envs = SimpleNamespace(mla_disable_dense_prefill=False)
    scope = dict(envs=envs)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    return scope['_forward_attention'], envs


@pytest.mark.parametrize('disable_dense', [False, True])
@pytest.mark.parametrize('length', [128, 1024, 2048, 2049, 4096])
@pytest.mark.parametrize('has_mtp_buffer', [False, True])
def test_prefill_route_and_required_indices(forward, disable_dense, length, has_mtp_buffer):
    method, envs = forward
    envs.mla_disable_dense_prefill = disable_dense
    owner = _Attention()
    buffer = object() if has_mtp_buffer else None
    expected = 'sparse' if disable_dense or length > 2048 else 'dense'
    with pytest.raises(_Route, match=expected):
        method(owner, None, None, _Tensor(), _Tensor(), None, past_key_value=[_Tensor(), _Tensor()],
               attn_metadata=SimpleNamespace(is_decoding=False, max_kv_seqlen=length),
               tail_k=None, tail_score=None, state_ids=None, topk_indices_buffer=buffer)
    assert owner.return_indices == (expected == 'sparse')
    assert owner.topk_buffer is buffer


@pytest.mark.parametrize('disable_dense', [False, True])
def test_decode_is_unaffected(forward, disable_dense):
    method, envs = forward
    envs.mla_disable_dense_prefill = disable_dense
    owner = _Attention()
    with pytest.raises(_Route, match='sparse'):
        method(owner, None, None, _Tensor(), _Tensor(), None, past_key_value=[_Tensor(), _Tensor()],
               attn_metadata=SimpleNamespace(is_decoding=True, max_kv_seqlen=128),
               tail_k=None, tail_score=None, state_ids=None)
    assert owner.return_indices
