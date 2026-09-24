import pytest

from lmdeploy.pytorch.strategies.ar_spec.cudagraph import ARSpecCudagraphStrategy


def test_arspec_cudagraph_uses_single_token_graph_for_all_methods():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='qwen3_5_mtp')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=8) == 8


def test_arspec_cudagraph_uses_same_allocation_for_full_spec_capture():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='qwen3_5_mtp')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=40) == 40


def test_dspark_cudagraph_keeps_target_and_draft_query_widths_distinct():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=5, method='dspark')

    # One live request padded to a four-request bucket. The target verifies
    # N+1 rows while the sample-from-anchor draft queries only N rows.
    assert strategy.get_max_tokens(batch_size=4, origin_batch_size=1,
                                   num_tokens=6) == 24
    assert strategy.get_max_tokens(batch_size=4, origin_batch_size=1,
                                   num_tokens=5) == 20


def test_arspec_cudagraph_keeps_full_spec_capture_for_eagle3():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='eagle3')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=8) == 8
    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=40) == 40


def test_cudagraph_step_metadata_plan_owns_single_token_capture_buffers(monkeypatch):
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.models.utils import cudagraph as cudagraph_mod
    from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta, CudaGraphMixin

    step_context = SimpleNamespace(model_config=SimpleNamespace(sliding_window=4096))
    monkeypatch.setattr(cudagraph_mod, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: step_context))

    class DummyPlan:

        def __init__(self):
            self.max_seqlen_q_calls = []

        def make_cudagraph_buffers(self, graph_meta, input_buffers, step_context):
            self.max_seqlen_q_calls.append(graph_meta.decode_query_len)
            return SimpleNamespace(attention_buffers=(torch.zeros(4, dtype=torch.int32), ))

        def fill_cudagraph_buffers(self, graph_meta, input_buffers, step_context, buffers, attn_metadata):
            self.max_seqlen_q_calls.append(graph_meta.decode_query_len)
            assert len(buffers.attention_buffers) == 1

    class DummyCudaGraphModel(CudaGraphMixin):
        pass

    model = DummyCudaGraphModel()
    plan = DummyPlan()
    graph_meta = CudaGraphMeta(
        max_batchs=8,
        max_tokens=8,
        num_blocks=1,
        is_decoding=True,
        device=torch.device('cpu'),
        input_buffers={},
        output_buffers={},
        use_fa3_decoding=True,
        decode_query_len=1,
        step_meta_plan=plan,
    )
    input_ids = torch.zeros((1, 8), dtype=torch.long)
    position_ids = torch.zeros_like(input_ids)
    attn_metadata = SimpleNamespace(
        q_seqlens=torch.ones(8, dtype=torch.long),
        block_offsets=torch.zeros((8, 1), dtype=torch.long),
        q_start_loc=torch.arange(8, dtype=torch.long),
        kv_seqlens=torch.ones(8, dtype=torch.long),
    )

    graph_meta.input_buffers = model.make_buffers_cudagraph(
        graph_meta,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=[],
        attn_metadata=attn_metadata,
    )
    model.fill_buffers_cudagraph(
        graph_meta,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=[],
        attn_metadata=attn_metadata,
        inputs_embeds=None,
    )

    assert plan.max_seqlen_q_calls == [1, 1]


def test_cuda_graph_key_separates_query_len_without_target_hidden_size(monkeypatch):
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.backends.cuda.graph_runner import runner as cuda_graph_runner
    from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin

    context = SimpleNamespace(
        global_is_decoding=lambda: True,
        target_hidden_states=torch.zeros((1, 8, 16)),
    )
    runner = cuda_graph_runner.CUDAGraphRunner.__new__(cuda_graph_runner.CUDAGraphRunner)
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: context)
    runner.model = CudaGraphMixin()
    runner.get_meta = lambda: SimpleNamespace(padding_batch_size=None)
    runner._get_capture_tokens = lambda batch_size: batch_size

    step_context = SimpleNamespace(enable_microbatch=False)
    step_ctx_mgr = SimpleNamespace(current_context=lambda: step_context)
    monkeypatch.setattr(cuda_graph_runner, 'get_step_ctx_manager', lambda: step_ctx_mgr)

    def make_attn_metadata(batch_size: int, query_len: int):
        return SimpleNamespace(
            q_seqlens=torch.full((batch_size, ), query_len, dtype=torch.long),
            q_start_loc=torch.arange(batch_size, dtype=torch.long) * query_len,
        )

    input_ids_qlen4 = torch.zeros((1, 32), dtype=torch.long)
    input_ids_qlen1 = torch.zeros((1, 8), dtype=torch.long)
    attn_metadata_qlen4 = make_attn_metadata(batch_size=8, query_len=4)
    attn_metadata_qlen1 = make_attn_metadata(batch_size=8, query_len=1)

    key_qlen4 = runner.get_graph_key(
        input_ids=input_ids_qlen4,
        position_ids=torch.zeros_like(input_ids_qlen4),
        past_key_values=[],
        attn_metadata=attn_metadata_qlen4,
        inputs_embeds=None,
    )
    key_qlen1 = runner.get_graph_key(
        input_ids=input_ids_qlen1,
        position_ids=torch.zeros_like(input_ids_qlen1),
        past_key_values=[],
        attn_metadata=attn_metadata_qlen1,
        inputs_embeds=None,
    )
    assert key_qlen4 != key_qlen1

    context.target_hidden_states = torch.zeros((1, 8, 32))
    key_hidden32 = runner.get_graph_key(
        input_ids=input_ids_qlen4,
        position_ids=torch.zeros_like(input_ids_qlen4),
        past_key_values=[],
        attn_metadata=attn_metadata_qlen4,
        inputs_embeds=None,
    )
    assert key_hidden32 == key_qlen4


def test_cuda_graph_key_separates_dsa_seed_and_reuse(monkeypatch):
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.backends.cuda.graph_runner import runner as cuda_graph_runner
    from lmdeploy.pytorch.models.glm_moe_dsa_mtp import GlmMoeDsaMTPModel

    runner = cuda_graph_runner.CUDAGraphRunner.__new__(cuda_graph_runner.CUDAGraphRunner)
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: SimpleNamespace(global_is_decoding=lambda: True))
    runner.model = GlmMoeDsaMTPModel.__new__(GlmMoeDsaMTPModel)
    runner.get_meta = lambda: SimpleNamespace(padding_batch_size=None)
    runner._get_capture_tokens = lambda batch_size: batch_size
    monkeypatch.setattr(cuda_graph_runner, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(enable_microbatch=False)))

    input_ids = torch.zeros((1, 8), dtype=torch.long)
    kwargs = dict(
        input_ids=input_ids,
        position_ids=torch.zeros_like(input_ids),
        past_key_values=[],
        attn_metadata=SimpleNamespace(q_seqlens=torch.ones(8, dtype=torch.long)),
        inputs_embeds=None,
    )

    seed_key = runner.get_graph_key(**kwargs, skip_topk=False)
    reuse_key = runner.get_graph_key(**kwargs, skip_topk=True)

    assert seed_key != reuse_key
    assert seed_key[-1] is False
    assert reuse_key[-1] is True


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_graph_capture_advances_state_once_without_snapshot_sync(monkeypatch, device):
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.backends.cuda.graph_runner.full_graph import CUDASingleGraphRunner

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')

    # Slot 4 is larger than the layer dimension: using axis 1 for anonymous
    # SSM state fails. Include duplicate ids and graph-padding sentinels.
    ids = torch.tensor([4, 1, -1, 1], device=device)
    caches = {'state_0': torch.arange(60).reshape(5, 2, 6).float(),
              'v4': torch.arange(60, dtype=torch.uint8).reshape(2, 5, 6)}
    caches = {name: cache.to(device) for name, cache in caches.items()}
    before = {name: cache.clone() for name, cache in caches.items()}
    context = SimpleNamespace(named_state_caches=caches, model_config=SimpleNamespace(
        state_cache_specs=[SimpleNamespace(name='v4', layer_ids=(0, 1))]))
    runner = CUDASingleGraphRunner.__new__(CUDASingleGraphRunner)
    runner._use_graph = device == 'cuda'
    runner._pool = torch.cuda.graph_pool_handle() if runner._use_graph else None
    runner._ctx_mgr = SimpleNamespace(current_context=lambda: context)
    runner.meta = SimpleNamespace(step_meta_plan=None)
    runner.model = SimpleNamespace(
        make_buffers_cudagraph=lambda *a, **kw: {'state_ids': ids},
        make_output_buffers=lambda output: output,
        get_outputs_cudagraph=lambda output, **kw: output)
    runner._bind_inputs = lambda **kw: {}

    def forward():
        for name, cache in caches.items():
            for slot in (1, 4):
                cache.select(1 if name == 'v4' else 0, slot).add_(1)
        return torch.ones(1, device=device)

    def no_dynamic_selection(*args, **kwargs):
        raise AssertionError('warmup snapshot must not synchronize for dynamic selection')

    monkeypatch.setattr(torch, 'unique', no_dynamic_selection)
    monkeypatch.setattr(torch, 'nonzero', no_dynamic_selection)
    monkeypatch.setattr(torch.Tensor, '__getitem__', no_dynamic_selection)
    runner._model_forward = forward
    if runner._use_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            runner.capture()
        torch.cuda.current_stream().wait_stream(stream)
    else:
        runner.capture()
    for invocation in (1, 2):
        if invocation == 2:
            runner.forward()
        for name, cache in caches.items():
            expected = before[name].clone()
            for slot in (1, 4):
                expected.select(1 if name == 'v4' else 0, slot).add_(invocation)
            torch.testing.assert_close(cache, expected)
