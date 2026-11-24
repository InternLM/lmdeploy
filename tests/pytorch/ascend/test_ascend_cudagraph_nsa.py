# Copyright (c) 2026, DeepLink. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.ascend import graph_runner as ascend_cudagraph


def _graph_meta(max_batches=4, query_len=1, sparse=True):
    return SimpleNamespace(
        max_batchs=max_batches,
        max_tokens=max_batches * query_len,
        num_blocks=3,
        device=torch.device('cpu'),
        is_ssm=False,
        use_mrope=False,
        vocab_size=128,
        mla_index_topk=2048 if sparse else None,
        input_buffers={},
    )


def _model(model_type='glm_moe_dsa'):
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type))


@pytest.mark.parametrize('query_len', [1, 3])
def test_graph_buffers_use_canonical_sequence_metadata(query_len):
    graph_meta = _graph_meta(query_len=query_len)
    buffers = ascend_cudagraph.make_buffers_cudagraph(_model(), graph_meta)

    assert buffers['q_seqlens'].tolist() == [query_len] * 4
    assert buffers['cu_seqlens_q'].tolist() == [step * query_len for step in range(5)]
    assert buffers['kv_seqlens'].tolist() == [1, 1, 1, 1]
    assert 'cu_seqlens_q_cpu' not in buffers
    assert 'kv_seqlens_cpu' not in buffers
    assert 'attention_mask' not in buffers


def test_non_dsa_graph_uses_same_metadata_contract():
    buffers = ascend_cudagraph.make_buffers_cudagraph(_model('deepseek_v2'), _graph_meta(sparse=False))

    assert not any(name.startswith('nsa_') for name in buffers)
    assert (
        set(
            (
                'q_seqlens',
                'cu_seqlens_q',
                'cu_seqlens_q_cpu',
                'kv_seqlens',
                'kv_seqlens_cpu',
            )
        )
        <= buffers.keys()
    )
    assert 'attention_mask' in buffers


def test_fill_glm_dsa_graph_buffers_pads_and_rebinds_metadata(monkeypatch):
    graph_meta = _graph_meta()
    graph_meta.input_buffers = ascend_cudagraph.make_buffers_cudagraph(_model(), graph_meta)
    moe_metadata = SimpleNamespace(x_active_mask=None)
    context = SimpleNamespace(moe_metadata=moe_metadata)
    manager = SimpleNamespace(current_context=lambda: context)
    monkeypatch.setattr(ascend_cudagraph, 'get_step_ctx_manager', lambda: manager)

    metadata = SimpleNamespace(
        block_offsets=torch.tensor([[7, 8], [9, 10]], dtype=torch.int32),
        kv_seqlens=torch.tensor([5, 7], dtype=torch.int32),
        kv_seqlens_cpu=None,
        kv_start_indices=torch.tensor([4, 6], dtype=torch.int32),
        q_start_loc=None,
        cache_seqlens=None,
        is_multi_token_decoding=False,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_q_cpu=None,
    )
    inputs = ascend_cudagraph.fill_buffers_cudagraph(
        _model(),
        graph_meta,
        input_ids=torch.tensor([[11, 12]], dtype=torch.int32),
        position_ids=torch.tensor([[4, 6]], dtype=torch.int32),
        past_key_values=[],
        attn_metadata=metadata,
        inputs_embeds=None,
    )

    assert metadata.q_seqlens is graph_meta.input_buffers['q_seqlens']
    assert metadata.cu_seqlens_q is graph_meta.input_buffers['cu_seqlens_q']
    assert metadata.cu_seqlens_q_cpu is None
    assert metadata.kv_seqlens is graph_meta.input_buffers['kv_seqlens']
    assert metadata.kv_seqlens_cpu is None
    assert metadata.q_seqlens.tolist() == [1, 1, 1, 1]
    assert metadata.cu_seqlens_q.tolist() == [0, 1, 2, 3, 4]
    assert metadata.kv_seqlens.tolist() == [5, 7, 0, 0]
    assert inputs['attn_metadata'] is metadata


def test_sparse_graph_replay_skips_cpu_metadata_update():
    class _Graph:
        replayed = False

        def replay(self):
            self.replayed = True

        def update(self, **kwargs):
            pytest.fail('sparse graph must not update CPU attention metadata')

    output = object()
    graph = _Graph()
    runner = object.__new__(ascend_cudagraph.AscendSingleGraphRunner)
    runner._graph = graph
    runner.meta = SimpleNamespace(
        mla_index_topk=2048,
        input_buffers={},
        output_buffers=output,
    )
    runner.model = SimpleNamespace(
        fill_buffers_cudagraph=lambda *args, **kwargs: None,
        update_context_cudagraph=lambda *args, **kwargs: None,
        get_outputs_cudagraph=lambda buffers, **kwargs: buffers,
    )
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: object())
    runner.is_mla = True

    actual = runner.forward()

    assert graph.replayed
    assert actual is output
