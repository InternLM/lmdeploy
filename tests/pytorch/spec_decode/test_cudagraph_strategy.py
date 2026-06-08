from lmdeploy.pytorch.strategies.ar_spec.cudagraph import ARSpecCudagraphStrategy


def test_arspec_cudagraph_allocates_max_decode_width_for_single_token_capture():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='qwen3_5_mtp')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=8) == 40


def test_arspec_cudagraph_uses_same_allocation_for_full_spec_capture():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='qwen3_5_mtp')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=40) == 40


def test_arspec_cudagraph_keeps_single_token_graph_for_eagle3():
    strategy = ARSpecCudagraphStrategy(num_spec_tokens=4, method='eagle3')

    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=8) == 8
    assert strategy.get_max_tokens(batch_size=8, origin_batch_size=8, num_tokens=40) == 40


def test_cudagraph_fa3_metadata_uses_padded_query_len_for_single_token_capture():
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
        max_tokens=32,
        num_blocks=1,
        is_decoding=True,
        device=torch.device('cpu'),
        input_buffers={},
        output_buffers={},
        use_fa3_decoding=True,
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

    assert model.max_seqlen_q_calls == [4, 4]
