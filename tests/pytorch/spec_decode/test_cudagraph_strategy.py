from lmdeploy.pytorch.strategies.ar_spec.cudagraph import ARSpecCudagraphStrategy


def test_arspec_cudagraph_uses_single_token_graph_for_all_methods():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='qwen3_5_mtp')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=8) == 8


def test_arspec_cudagraph_uses_same_allocation_for_full_spec_capture():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='qwen3_5_mtp')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=40) == 40


def test_arspec_cudagraph_keeps_full_spec_capture_for_eagle3():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='eagle3')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=8) == 8
    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=40) == 40


def test_cudagraph_fa3_metadata_uses_single_query_len_for_single_token_capture():
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta, CudaGraphMixin

    class DummyCudaGraphModel(CudaGraphMixin):

        def __init__(self):
            self.max_seqlen_q_calls = []

        def update_meta_flashattn(self, batch_size, max_seqlen_q, block_size, max_seqlen_k, cache_seqlens):
            self.max_seqlen_q_calls.append(max_seqlen_q)
            return torch.zeros(4, dtype=torch.int32)

    model = DummyCudaGraphModel()
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

    assert model.max_seqlen_q_calls == [1, 1]


def test_cudagraph_fill_preserves_runtime_attention_metadata():
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta, CudaGraphMixin

    model = CudaGraphMixin()
    graph_meta = CudaGraphMeta(
        max_batchs=2,
        max_tokens=4,
        num_blocks=2,
        is_decoding=True,
        device=torch.device('cpu'),
        input_buffers={},
        output_buffers={},
        decode_query_len=2,
    )
    input_ids = torch.arange(4).view(1, 4)
    position_ids = input_ids.clone()
    attn_metadata = SimpleNamespace(
        q_seqlens=torch.tensor([2, 2]),
        block_offsets=torch.tensor([[3, 4], [5, 6]]),
        q_start_loc=torch.tensor([0, 2]),
        kv_seqlens=torch.tensor([7, 11]),
    )
    original_fields = {
        name: getattr(attn_metadata, name).clone()
        for name in ('q_seqlens', 'block_offsets', 'q_start_loc', 'kv_seqlens')
    }
    graph_meta.input_buffers = model.make_buffers_cudagraph(
        graph_meta,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=[],
        attn_metadata=attn_metadata,
    )

    for _ in range(2):
        graph_inputs = model.fill_buffers_cudagraph(
            graph_meta,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=[],
            attn_metadata=attn_metadata,
            inputs_embeds=None,
        )
        for name, expected in original_fields.items():
            assert torch.equal(getattr(attn_metadata, name), expected)
        assert torch.equal(graph_inputs['attn_metadata'].q_seqlens[:2], original_fields['q_seqlens'])
        assert torch.equal(graph_inputs['attn_metadata'].kv_seqlens[:2], original_fields['kv_seqlens'])
        assert torch.equal(graph_inputs['attn_metadata'].block_offsets[:2], original_fields['block_offsets'])


def test_cudagraph_capture_rolls_back_state_before_semantic_forward(monkeypatch):
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.backends.cuda import graph_runner as graph_runner_mod

    cache = [torch.arange(24).view(6, 4), torch.arange(24, 48).view(6, 4)]
    original = [tensor.clone() for tensor in cache]
    block_offsets = torch.tensor([[1, 3], [3, 4]])
    block_ids = torch.tensor([1, 3, 4])

    class FakeSingleGraphRunner:

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def capture(self, **kwargs):
            del kwargs
            for tensor in cache:
                tensor[block_ids] += 10
            return 'capture-output'

        def forward(self, **kwargs):
            del kwargs
            for tensor, expected in zip(cache, original):
                torch.testing.assert_close(tensor, expected)
                tensor[block_ids] += 1
            return 'semantic-output'

    monkeypatch.setattr(graph_runner_mod, 'CUDASingleGraphRunner', FakeSingleGraphRunner)
    monkeypatch.setattr(
        graph_runner_mod,
        'get_deepep_state',
        lambda: SimpleNamespace(enabled=lambda: False),
    )

    model = SimpleNamespace(
        get_cudagraph_capture_cache=lambda past_key_values, spec_step_idx: cache,
        get_cudagraph_extra_key=lambda **kwargs: (),
    )
    runner = graph_runner_mod.CUDAGraphRunner.__new__(graph_runner_mod.CUDAGraphRunner)
    runner.model = model
    runner.ctx_mgr = SimpleNamespace(
        current_context=lambda: SimpleNamespace(global_is_decoding=lambda: True))
    runner.enable_graph = lambda **kwargs: True
    runner.get_graph_key = lambda **kwargs: (2, True, False, 2)
    runner._get_max_tokens = lambda *args: 4
    runner._get_decode_model_forward = lambda: model
    runner._runner_map = {}
    runner.num_blocks = 8
    runner.graph_pool_handle = None
    runner.model_config = SimpleNamespace()
    runner.device = torch.device('cpu')

    output = runner(
        input_ids=torch.zeros((1, 4), dtype=torch.long),
        position_ids=torch.zeros((1, 4), dtype=torch.long),
        past_key_values=[],
        attn_metadata=SimpleNamespace(
            q_seqlens=torch.tensor([2, 2]),
            block_offsets=block_offsets.to(torch.int32),
        ),
        inputs_embeds=None,
        spec_step_idx=0,
    )

    assert output == 'semantic-output'
    assert (2, True, False, 2) in runner._runner_map
    for actual, expected in zip(cache, original):
        expected[block_ids] += 1
        torch.testing.assert_close(actual, expected)


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
