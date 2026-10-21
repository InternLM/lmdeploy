# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('num_spec_tokens', [0, 1, 3])
@pytest.mark.parametrize('model_type', [None, 'glm5_next'])
def test_restored_kda_short_prefill_matches_full_context(monkeypatch, num_spec_tokens, model_type):
    from lmdeploy.pytorch.backends import gated_delta_rule as meta_module
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl

    torch.manual_seed(17)
    impl = CudaKdaImpl(model_type=model_type)
    dim, width = 128, 4
    histories, tails = (64, 128, 192, 256), (1, 2, 3, 8)
    source_ids, restored_ids, reference_ids = (5, 6, 7, 8), (3, 1, 4, 2), (9, 10, 11, 12)
    conv = torch.full((13, 3 * dim, width + num_spec_tokens), float('nan'), dtype=torch.bfloat16, device='cuda')
    recurrent_shape = (13, num_spec_tokens + 1, 1, dim, dim) if num_spec_tokens else (13, 1, dim, dim)
    recurrent = torch.full(recurrent_shape, float('nan'), dtype=torch.float32, device='cuda')
    kwargs = dict(conv_state=conv, recurrent_state=recurrent,
                  conv_weight=torch.randn(3 * dim, width, dtype=torch.bfloat16, device='cuda') / 8,
                  conv_bias=None, a_log=torch.zeros(1, device='cuda'), dt_bias=torch.zeros(dim, device='cuda'),
                  num_heads=1, head_dim=dim, lower_bound=-5.0)
    inputs = [dict(mixed_qkv=torch.randn(1, history + tail + 1, 3 * dim,
                                       dtype=torch.bfloat16, device='cuda') / 4,
                   raw_gate=torch.randn(1, history + tail + 1, dim, dtype=torch.bfloat16, device='cuda'),
                   raw_beta=torch.randn(1, history + tail + 1, 1, dtype=torch.bfloat16, device='cuda'))
              for history, tail in zip(histories, tails)]
    monkeypatch.setattr(meta_module, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(build_ctx=SimpleNamespace(num_spec_tokens=num_spec_tokens)))

    def forward(values, ids, history_lengths, is_decoding=False):
        lengths = [value['mixed_qkv'].size(1) for value in values]
        q_lengths = torch.tensor(lengths, dtype=torch.int32, device='cuda')
        cu_seqlens = torch.cat((q_lengths.new_zeros(1), q_lengths.cumsum(0)))
        metadata = meta_module.GatedDeltaMeta(sum(lengths), width, torch.tensor(ids, device='cuda'), SimpleNamespace(
            is_decoding=is_decoding, cu_seqlens_q=cu_seqlens, q_seqlens=q_lengths,
            kv_seqlens=q_lengths + torch.tensor(history_lengths, device='cuda')))
        packed = {name: torch.cat([value[name] for value in values], dim=1) for name in values[0]}
        return impl.forward(**packed, metadata=metadata, **kwargs)

    prefix = [{name: value[:, :history] for name, value in values.items()}
              for values, history in zip(inputs, histories)]
    forward(prefix, source_ids, (0,) * len(inputs))
    # Store restores the complete conv/recurrent checkpoint into new physical
    # slots. The short tails must keep their own history after this relocation.
    conv[list(restored_ids)] = conv[list(source_ids)]
    recurrent[list(restored_ids)] = recurrent[list(source_ids)]
    checkpoints = (conv[list(source_ids)].clone(), recurrent[list(source_ids)].clone())

    full_context = [{name: value[:, :-1] for name, value in values.items()} for values in inputs]
    ends = [history + tail for history, tail in zip(histories, tails)]
    reference = forward(full_context, reference_ids, (0,) * len(inputs))
    expected = torch.cat([value[:, -tail:] for value, tail in zip(reference.split(ends, dim=1), tails)], dim=1)
    suffix = [{name: value[:, history:-1] for name, value in values.items()}
              for values, history in zip(inputs, histories)]
    actual = forward(suffix, restored_ids, histories)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-4)

    for index, (slot, ref_slot, end) in enumerate(zip(restored_ids, reference_ids, ends)):
        offsets = (end + torch.arange(-width, 0, device='cuda')) % conv.size(-1)
        actual_conv = conv[slot, :, offsets] if num_spec_tokens else conv[slot]
        expected_conv = inputs[index]['mixed_qkv'][0, end - width:end].T
        torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
        actual_state = recurrent[slot, end % (num_spec_tokens + 1)] if num_spec_tokens else recurrent[slot]
        expected_state = recurrent[ref_slot, end % (num_spec_tokens + 1)] if num_spec_tokens else recurrent[ref_slot]
        torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-4)

    # A subsequent decode must consume the state written by the short prefill.
    next_token = [{name: value[:, -1:] for name, value in values.items()} for values in inputs]
    actual_next = forward(next_token, restored_ids, ends, is_decoding=True)
    expected_next = forward(next_token, reference_ids, ends, is_decoding=True)
    torch.testing.assert_close(actual_next, expected_next, rtol=1e-2, atol=5e-4)
    for state, checkpoint in zip((conv, recurrent), checkpoints):
        torch.testing.assert_close(state[list(source_ids)], checkpoint, rtol=0, atol=0, equal_nan=True)
        assert torch.isnan(state[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('model_type', [None, 'glm5_next'])
def test_restored_kda_checkpoint_survives_partial_mtp_acceptance(model_type):
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl

    torch.manual_seed(7)
    impl = CudaKdaImpl(model_type=model_type)
    dim, width, num_spec_tokens = 128, 4, 3
    conv = torch.full((3, 3 * dim, width + num_spec_tokens), float('nan'), dtype=torch.bfloat16, device='cuda')
    recurrent = torch.full((3, 4, 1, dim, dim), float('nan'), dtype=torch.float32, device='cuda')
    # Restore the checkpoint at token 64 into another request slot. Slot 1
    # verifies batches; slot 2 advances only through accepted tokens.
    conv[1].normal_(std=0.1)
    recurrent[1, 0].normal_(std=0.1)
    conv[2].copy_(conv[1])
    recurrent[2].copy_(recurrent[1])
    kwargs = dict(conv_state=conv, recurrent_state=recurrent,
                  conv_weight=torch.randn(3 * dim, width, dtype=torch.bfloat16, device='cuda') / 8,
                  conv_bias=None, a_log=torch.zeros(1, device='cuda'), dt_bias=torch.zeros(dim, device='cuda'),
                  num_heads=1, head_dim=dim, lower_bound=-5.0)

    def forward(inputs, state_id, history):
        metadata = SimpleNamespace(is_decoding=True, num_spec_tokens=num_spec_tokens,
                                   state_ids=torch.tensor([state_id], device='cuda'),
                                   valid_state=torch.tensor([True], device='cuda'),
                                   cache_seqlens=torch.tensor([history], dtype=torch.int32, device='cuda'))
        return impl.forward(**inputs, metadata=metadata, **kwargs)

    history = 64
    for accepted in (1, 2, 4, 1, 3):
        inputs = dict(mixed_qkv=torch.randn(1, 4, 3 * dim, dtype=torch.bfloat16, device='cuda') / 4,
                      raw_gate=torch.randn(1, 4, dim, dtype=torch.bfloat16, device='cuda'),
                      raw_beta=torch.randn(1, 4, 1, dtype=torch.bfloat16, device='cuda'))
        actual = forward(inputs, 1, history)
        expected = [forward({name: value[:, step:step + 1] for name, value in inputs.items()}, 2, history + step)
                    for step in range(accepted)]
        torch.testing.assert_close(actual[:, :accepted], torch.cat(expected, dim=1), rtol=1e-3, atol=1e-3)
        history += accepted
        torch.testing.assert_close(recurrent[1, history % 4], recurrent[2, history % 4], rtol=1e-4, atol=1e-4)
        window = (history + torch.arange(-width, 0, device='cuda')) % conv.size(-1)
        torch.testing.assert_close(conv[1, :, window], conv[2, :, window], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('seqlen_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('offsets', [(16, 64), (0, 64)])
def test_flash_kda_internal_checkpoint_matches_prefix_and_resume(seqlen_dtype, offsets, monkeypatch):
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl

    if torch.cuda.get_device_capability() != (9, 0) or torch.version.hip is not None:
        pytest.skip('direct GLM-5.3 FlashKDA deployment requires SM90')
    impl = CudaKdaImpl(model_type='glm5_next')

    def reject_wrapper(*args, **kwargs):
        pytest.fail('GLM-5.3 prefill must not call the public FlashKDA wrapper')

    monkeypatch.setattr('flash_kda.fwd', reject_wrapper)
    torch.manual_seed(42)
    lengths, heads, dim = (65, 129), 2, 128
    values = {name: torch.randn(1, sum(lengths), heads, dim, device='cuda', dtype=torch.bfloat16) / 4
              for name in ('q', 'k', 'v', 'g')}
    values['beta'] = torch.randn(1, sum(lengths), heads, device='cuda', dtype=torch.bfloat16)
    initial = torch.randn(2, heads, dim, dim, device='cuda') / 8
    params = dict(A_log=torch.zeros(heads, device='cuda'), dt_bias=torch.zeros(heads * dim, device='cuda'),
                  lower_bound=-5.0)
    cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=seqlen_dtype, device='cuda')
    checkpoint = torch.full_like(initial, float('nan'))
    with torch.inference_mode():
        plain, plain_final = impl._prefill_recurrent(**values, **params, initial_state=initial, cu_seqlens=cu)
        merged, merged_final = impl._prefill_recurrent(
            **values, **params, initial_state=initial, cu_seqlens=cu, checkpoint_state=checkpoint,
            checkpoint_offsets=torch.tensor(offsets, dtype=seqlen_dtype, device='cuda'))
        torch.testing.assert_close(merged, plain, rtol=0, atol=0)
        torch.testing.assert_close(merged_final, plain_final, rtol=0, atol=0)
        start = 0
        for index, (length, offset) in enumerate(zip(lengths, offsets)):
            if offset == 0:
                assert torch.isnan(checkpoint[index]).all()
                start += length
                continue
            prefix = {name: value[:, start:start + offset].contiguous() for name, value in values.items()}
            _, prefix_state = impl._prefill_recurrent(
                **prefix, **params, initial_state=initial[index:index + 1],
                cu_seqlens=torch.tensor([0, offset], dtype=seqlen_dtype, device='cuda'))
            torch.testing.assert_close(checkpoint[index:index + 1], prefix_state, rtol=1e-2, atol=1e-4)
            suffix = {name: value[:, start + offset:start + length].contiguous() for name, value in values.items()}
            resumed, resumed_final = impl._prefill_recurrent(
                **suffix, **params, initial_state=checkpoint[index:index + 1],
                cu_seqlens=torch.tensor([0, length - offset], dtype=seqlen_dtype, device='cuda'))
            torch.testing.assert_close(resumed, plain[:, start + offset:start + length], rtol=1e-2, atol=5e-4)
            torch.testing.assert_close(resumed_final, plain_final[index:index + 1], rtol=1e-2, atol=1e-4)
            start += length


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('num_spec_tokens', [0, 1, 3, 5])
def test_flash_kda_checkpoint_ring_view_preserves_live_prefill_state(num_spec_tokens):
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl

    if torch.cuda.get_device_capability() != (9, 0) or torch.version.hip is not None:
        pytest.skip('direct GLM-5.3 FlashKDA deployment requires SM90')
    impl = CudaKdaImpl(model_type='glm5_next')
    torch.manual_seed(23)
    dim, width, history, length, offset = 128, 4, 17, 65, 16
    ring = num_spec_tokens + 1
    conv = torch.randn(4, 3 * dim, width + num_spec_tokens, device='cuda', dtype=torch.bfloat16) / 8
    shape = (4, ring, 1, dim, dim) if num_spec_tokens else (4, 1, dim, dim)
    recurrent = torch.randn(shape, device='cuda') / 8
    conv[2:].copy_(conv[1:2].expand_as(conv[2:]))
    recurrent[2:].copy_(recurrent[1:2].expand_as(recurrent[2:]))
    values = dict(mixed_qkv=torch.randn(1, length, 3 * dim, device='cuda', dtype=torch.bfloat16) / 4,
                  raw_gate=torch.randn(1, length, dim, device='cuda', dtype=torch.bfloat16),
                  raw_beta=torch.randn(1, length, 1, device='cuda', dtype=torch.bfloat16))
    params = dict(conv_weight=torch.randn(3 * dim, width, device='cuda', dtype=torch.bfloat16) / 8,
                  conv_bias=None, a_log=torch.zeros(1, device='cuda'), dt_bias=torch.zeros(dim, device='cuda'),
                  conv_state=conv, recurrent_state=recurrent, num_heads=1, head_dim=dim, lower_bound=-5.0)
    reserved = torch.full((1, ring, 1, dim, dim), -99.0, device='cuda')
    checkpoint = reserved[:, (history + offset) % ring]

    def forward(state_id, tokens, capture=False):
        metadata = SimpleNamespace(is_decoding=False, num_spec_tokens=num_spec_tokens,
                                   state_ids=torch.tensor([state_id], device='cuda'),
                                   valid_state=torch.tensor([True], device='cuda'),
                                   is_init=torch.tensor([False], device='cuda'),
                                   cache_seqlens=torch.tensor([history], device='cuda'),
                                   cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device='cuda'))
        return impl.forward(**{name: value[:, :tokens] for name, value in values.items()}, **params, metadata=metadata,
                            checkpoint_state=checkpoint if capture else None,
                            checkpoint_offsets=torch.tensor([offset], dtype=torch.int32, device='cuda')
                            if capture else None)

    with torch.inference_mode():
        plain = forward(1, length)
        merged = forward(2, length, capture=True)
        forward(3, offset)
    torch.testing.assert_close(merged, plain, rtol=0, atol=0)
    torch.testing.assert_close(recurrent[2], recurrent[1], rtol=0, atol=0)
    torch.testing.assert_close(conv[2], conv[1], rtol=0, atol=0)
    expected = recurrent[3:4, (history + offset) % ring] if num_spec_tokens else recurrent[3:4]
    torch.testing.assert_close(checkpoint, expected, rtol=1e-2, atol=1e-4)
    for slot in range(ring):
        if slot != (history + offset) % ring:
            assert torch.all(reserved[:, slot] == -99)
