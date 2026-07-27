import torch

from lmdeploy.pytorch.model_inputs import DPMeta, ModelInputs, StepContext


def _make_model_inputs(is_decoding: bool, dp_is_decoding: bool | None = None):
    dp_meta = None
    if dp_is_decoding is not None:
        dp_meta = DPMeta(dp_is_decoding=dp_is_decoding)
    return ModelInputs(
        input_ids=torch.zeros((1, 1), dtype=torch.long),
        seq_length=torch.ones(1, dtype=torch.long),
        history_lengths=torch.zeros(1, dtype=torch.long),
        block_offsets=torch.zeros((1, 1), dtype=torch.long),
        is_decoding=is_decoding,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=1,
        max_kv_seqlen=1,
        sum_kv_seqlen=1,
        dp_meta=dp_meta,
    )


def test_model_inputs_global_is_decoding_uses_local_without_dp_meta():
    assert _make_model_inputs(is_decoding=True).global_is_decoding()
    assert not _make_model_inputs(is_decoding=False).global_is_decoding()


def test_model_inputs_global_is_decoding_uses_dp_global_state():
    assert not _make_model_inputs(is_decoding=True, dp_is_decoding=False).global_is_decoding()
    assert _make_model_inputs(is_decoding=False, dp_is_decoding=True).global_is_decoding()


def test_step_context_global_is_decoding_uses_dp_global_state():
    step_ctx = StepContext(
        input_ids=torch.zeros((1, 1), dtype=torch.long),
        model_config=None,
        cache_config=None,
        block_offsets=torch.zeros((1, 1), dtype=torch.long),
        position_ids=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=None,
        q_seqlens=torch.ones(1, dtype=torch.long),
        kv_seqlens=torch.ones(1, dtype=torch.long),
        q_start_loc=torch.zeros(1, dtype=torch.long),
        kv_caches=[],
        is_decoding=True,
        sum_kv_seqlen=1,
        dp_meta=DPMeta(dp_is_decoding=False),
    )
    assert not step_ctx.global_is_decoding()


def test_step_context_single_token_decode_metadata(monkeypatch):
    import lmdeploy.pytorch.model_inputs as model_inputs

    monkeypatch.setattr(model_inputs, 'get_backend',
                        lambda: type('Backend', (), {'update_step_context': staticmethod(lambda ctx: ctx)})())
    inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12]]),
        seq_length=torch.ones(3, dtype=torch.long),
        history_lengths=torch.tensor([4, 9, 11]),
        block_offsets=torch.zeros((3, 1), dtype=torch.long),
        is_decoding=True,
        num_ignored_history=torch.tensor([0, 2, 3]),
        max_q_seqlen=1,
        max_kv_seqlen=12,
        sum_kv_seqlen=22,
    )

    context = StepContext.new(inputs, model_config=None, cache_config=None)

    torch.testing.assert_close(context.attention_mask, torch.ones((3, 1), dtype=torch.long))
    torch.testing.assert_close(context.position_ids, torch.tensor([[4, 9, 11]]))
    torch.testing.assert_close(context.q_start_loc, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(context.kv_seqlens, torch.tensor([5, 8, 9]))
