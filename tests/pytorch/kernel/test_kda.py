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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('num_spec_tokens', [0, 3, 5])
@pytest.mark.parametrize('relocate', [False, True])
def test_replayssm_prefill_decode_partial_acceptance_and_continuation(monkeypatch, num_spec_tokens, relocate):
    from lmdeploy.pytorch.backends import gated_delta_rule as meta_module
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaReplayCursor, GatedDeltaReplayState
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import advance_gdn_replayssm_standard_cursor
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import commit_gdn_replayssm_spec_cursor

    torch.manual_seed(71)
    dim, heads, width, length, slots = 128, 2, 4, 16, 3
    steps = num_spec_tokens + 1
    baseline, replay = CudaKdaImpl(), CudaKdaImpl()
    conv = torch.zeros(slots, 3 * heads * dim, width + num_spec_tokens, device='cuda', dtype=torch.bfloat16)
    checkpoint = torch.zeros(slots, heads, dim, dim, device='cuda')
    state = GatedDeltaReplayState(
        conv_state=conv.clone(), checkpoint=checkpoint.clone(),
        d_ring=torch.zeros(slots, heads, length, dim, device='cuda'),
        k_ring=torch.zeros(slots, heads, length, dim, device='cuda'),
        g_ring=torch.zeros(slots, heads, length, dim, device='cuda'),
        cursor=GatedDeltaReplayCursor(torch.zeros(slots, 3, device='cuda', dtype=torch.int32), length, length, steps),
        raw_v_ring=torch.zeros(slots, heads, length, dim, device='cuda', dtype=torch.bfloat16),
        raw_k_ring=torch.zeros(slots, heads, length, dim, device='cuda', dtype=torch.bfloat16),
        beta_ring=torch.zeros(slots, heads, length, device='cuda'))
    recurrent = checkpoint[:, None].repeat(1, steps, 1, 1, 1) if num_spec_tokens else checkpoint
    kwargs = dict(conv_weight=torch.randn(3 * heads * dim, width, device='cuda', dtype=torch.bfloat16) / 8,
                  conv_bias=None, a_log=torch.randn(heads, device='cuda'),
                  dt_bias=torch.randn(heads * dim, device='cuda'), num_heads=heads, head_dim=dim, lower_bound=-5.)
    monkeypatch.setattr(meta_module, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(build_ctx=SimpleNamespace(num_spec_tokens=num_spec_tokens)))
    # Restore the complete live tuple, not only its materialized checkpoint.
    # This exercises backend addressing after relocation, not allocator/Store I/O.
    banks = [state.conv_state, state.checkpoint, state.d_ring, state.k_ring,
             state.g_ring, state.raw_v_ring, state.raw_k_ring, state.beta_ring,
             state.cursor.packed_cursor]
    history, slot = 0, 1
    source_snapshots = None
    for iteration, count in enumerate([65] + [steps] * 18 + [1, 7, steps]):
        if relocate and iteration == 3:
            assert state.cursor.packed_cursor[slot, 0].item() > 0
            source_snapshots = [bank[slot].clone() for bank in banks]
            for bank in banks:
                bank[2].copy_(bank[slot])
            conv[2].copy_(conv[slot])
            recurrent[2].copy_(recurrent[slot])
            slot = 2
        decoding = iteration not in (0, 20)
        qlen = torch.tensor([count], device='cuda', dtype=torch.int32)
        ids = torch.tensor([slot], device='cuda')
        metadata = meta_module.GatedDeltaMeta(count, width, ids, SimpleNamespace(
            is_decoding=decoding, cu_seqlens_q=torch.tensor([0, count], device='cuda'),
            q_seqlens=qlen, kv_seqlens=qlen + history))
        # Verify buffers retain configured Tmax even when only one input is valid.
        # Do not accidentally test a different, unsupported physical-width ABI.
        physical_steps = steps if decoding else count
        inputs = dict(mixed_qkv=torch.randn(1, physical_steps, 3 * heads * dim,
                                           device='cuda', dtype=torch.bfloat16) / 4,
                      raw_gate=torch.randn(1, physical_steps, heads * dim, device='cuda', dtype=torch.bfloat16),
                      raw_beta=torch.randn(1, physical_steps, heads, device='cuda', dtype=torch.bfloat16))
        if relocate:
            # Slow decay keeps missing retained history observable after the move.
            inputs['raw_gate'].sub_(4)
        expected = baseline.forward(**inputs, **kwargs, metadata=metadata, conv_state=conv, recurrent_state=recurrent)
        actual = replay.forward(**inputs, **kwargs, metadata=metadata, conv_state=None, recurrent_state=state)
        torch.testing.assert_close(actual[:, :count], expected[:, :count], atol=2e-4, rtol=1e-2)
        if not decoding:
            expected_state = recurrent[slot, (history + count) % steps] if num_spec_tokens else recurrent[slot]
            torch.testing.assert_close(state.checkpoint[slot], expected_state, atol=2e-4, rtol=1e-2)
            # The model resets only the scheduled request's cursor after prefill.
            state.cursor.packed_cursor[slot].zero_()
            accepted = count
        elif num_spec_tokens:
            accepted = min(count, iteration % steps + 1)
            rejected = torch.tensor([count - accepted], device='cuda', dtype=torch.int32)
            commit_gdn_replayssm_spec_cursor(state.cursor.packed_cursor, ids, qlen, rejected, length, steps)
        else:
            accepted = count
            advance_gdn_replayssm_standard_cursor(state.cursor.packed_cursor, ids, length)
        history += accepted
        if num_spec_tokens:
            window = (history + torch.arange(-width, 0, device='cuda')) % conv.size(-1)
            torch.testing.assert_close(state.conv_state[slot, :, window], conv[slot, :, window], rtol=0, atol=0)
        else:
            torch.testing.assert_close(state.conv_state[slot], conv[slot], rtol=0, atol=0)
        for bank in banks:
            assert torch.count_nonzero(bank[0]) == 0
        if source_snapshots is not None:
            for bank, saved in zip(banks, source_snapshots):
                torch.testing.assert_close(bank[1], saved, rtol=0, atol=0)
    if not relocate:
        for bank in banks:
            assert torch.count_nonzero(bank[2]) == 0
