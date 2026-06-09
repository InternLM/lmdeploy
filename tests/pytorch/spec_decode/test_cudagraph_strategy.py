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

        def update_meta_flashattn(self, batch_size, max_seqlen_q, block_size, max_seqlen_k, cache_seqlens,
                                  num_splits=0):
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


def test_cuda_graph_key_separates_query_len_without_target_hidden_size(monkeypatch):
    from types import SimpleNamespace

    import torch

    from lmdeploy.pytorch.backends.cuda import graph_runner as cuda_graph_runner

    context = SimpleNamespace(
        global_is_decoding=lambda: True,
        target_hidden_states=torch.zeros((1, 8, 16)),
    )
    runner = cuda_graph_runner.CUDAGraphRunner.__new__(cuda_graph_runner.CUDAGraphRunner)
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: context)
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
