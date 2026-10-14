# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('num_spec_tokens', [0, 1, 3])
def test_restored_kda_short_prefill_matches_full_context(monkeypatch, num_spec_tokens):
    from lmdeploy.pytorch.backends import gated_delta_rule as meta_module
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl

    torch.manual_seed(17)
    impl = CudaKdaImpl()
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
def test_restored_kda_checkpoint_survives_partial_mtp_acceptance():
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl

    torch.manual_seed(7)
    impl = CudaKdaImpl()
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
