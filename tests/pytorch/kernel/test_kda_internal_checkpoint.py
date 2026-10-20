from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize(('heads', 'length', 'offset', 'history'), [(4, 65, 64, 0), (64, 8256, 8192, 0),
                                                                  (4, 11808, 8192, 8192)])
def test_native_backend_internal_checkpoint_and_restore(monkeypatch, heads, length, offset, history):
    pytest.importorskip('flash_kda')
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
    from lmdeploy.pytorch.backends.kda import KdaPrefillCheckpoint

    monkeypatch.setattr(envs, 'kda_prefill_backend', 'flash_kda')
    impl = CudaKdaImpl()
    torch.manual_seed(42)
    width, dim, slots = 4, 128, 5
    conv_pool = torch.full((slots, 2, heads * dim * 3, width), float('nan'), dtype=torch.bfloat16, device='cuda')
    state_pool = torch.full((slots, 2, heads, dim, dim), float('nan'), dtype=torch.float32, device='cuda')
    conv, state = conv_pool[:, 1], state_pool[:, 1]
    conv[1].zero_()
    state[1].zero_()
    if history:
        conv[1].normal_(std=0.1)
        state[1].normal_(std=0.1)
    conv[2].copy_(conv[1])
    state[2].copy_(state[1])
    values = dict(mixed_qkv=torch.randn(1, length + 1, 3 * heads * dim, dtype=torch.bfloat16, device='cuda') / 4,
                  raw_gate=torch.randn(1, length + 1, heads * dim, dtype=torch.bfloat16, device='cuda'),
                  raw_beta=torch.randn(1, length + 1, heads, dtype=torch.bfloat16, device='cuda'))
    kwargs = dict(conv_state=conv, recurrent_state=state,
                  conv_weight=torch.randn(3 * heads * dim, width, dtype=torch.bfloat16, device='cuda') / 8,
                  conv_bias=None, a_log=torch.zeros(heads, device='cuda'),
                  dt_bias=torch.zeros(heads * dim, device='cuda'),
                  num_heads=heads, head_dim=dim, lower_bound=-5.0)

    def forward(start, end, slot, checkpoint=None, decode=False):
        meta = SimpleNamespace(is_decoding=decode, num_spec_tokens=0, is_init=None,
                               valid_state=torch.tensor([True], device='cuda'),
                               state_ids=torch.tensor([slot], device='cuda'),
                               cu_seqlens=torch.tensor([0, end - start], dtype=torch.int64, device='cuda'))
        return impl.forward(**{name: value[:, start:end] for name, value in values.items()},
                            metadata=meta, checkpoint=checkpoint, **kwargs)

    checkpoint = KdaPrefillCheckpoint(offset, torch.tensor([offset], dtype=torch.int64, device='cuda'),
                                     conv[3:4], state[3:4])
    full = forward(0, length, 1, checkpoint)
    prefix = forward(0, offset, 2)
    torch.testing.assert_close(conv[3], conv[2], atol=0, rtol=0)
    torch.testing.assert_close(state[3], state[2], atol=0, rtol=0)
    conv[4].copy_(conv[3])
    state[4].copy_(state[3])
    saved_state = state[3].clone()
    suffix = forward(offset, length, 2)
    restored_suffix = forward(offset, length, 4)
    torch.testing.assert_close(full, torch.cat((prefix, suffix), dim=1), atol=0, rtol=0)
    torch.testing.assert_close(suffix, restored_suffix, atol=0, rtol=0)
    torch.testing.assert_close(state[1], state[2], atol=0, rtol=0)
    torch.testing.assert_close(conv[1], conv[2], atol=0, rtol=0)
    torch.testing.assert_close(forward(length, length + 1, 1, decode=True),
                               forward(length, length + 1, 2, decode=True), atol=0, rtol=0)
    torch.testing.assert_close(state[3], saved_state, atol=0, rtol=0)
    assert torch.isnan(conv_pool[:, 0]).all() and torch.isnan(state_pool[:, 0]).all()
    assert torch.isnan(conv[0]).all() and torch.isnan(state[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize(('draft', 'heads', 'length', 'offset', 'history'),
                         [(1, 4, 65, 64, 0), (3, 4, 8256, 8192, 0), (5, 4, 8256, 8192, 0),
                          (3, 4, 11808, 8192, 8192), (5, 64, 8256, 8192, 8192)])
def test_mtp_internal_checkpoint_ring_restore_and_rejection(monkeypatch, draft, heads, length, offset, history):
    pytest.importorskip('flash_kda')
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
    from lmdeploy.pytorch.backends.kda import KdaPrefillCheckpoint

    monkeypatch.setattr(envs, 'kda_prefill_backend', 'flash_kda')
    impl = CudaKdaImpl()
    torch.manual_seed(42)
    width, dim, slots, ring = 4, 128, 7, draft + 1
    channels = 3 * heads * dim
    conv_pool = torch.full((slots, 2, channels, width + draft), float('nan'), dtype=torch.bfloat16, device='cuda')
    state_pool = torch.full((slots, 2, ring, heads, dim, dim), float('nan'), dtype=torch.float32, device='cuda')
    conv, state = conv_pool[:, 1], state_pool[:, 1]
    conv[1].zero_()
    state[1, history % ring].zero_()
    if history:
        conv[1].normal_(std=0.1)
        state[1, history % ring].normal_(std=0.1)
    conv[2].copy_(conv[1])
    state[2].copy_(state[1])
    values = dict(mixed_qkv=torch.randn(1, length, channels, dtype=torch.bfloat16, device='cuda') / 4,
                  raw_gate=torch.randn(1, length, heads * dim, dtype=torch.bfloat16, device='cuda'),
                  raw_beta=torch.randn(1, length, heads, dtype=torch.bfloat16, device='cuda'))
    kwargs = dict(conv_state=conv, recurrent_state=state,
                  conv_weight=torch.randn(channels, width, dtype=torch.bfloat16, device='cuda') / 8,
                  conv_bias=None, a_log=torch.zeros(heads, device='cuda'),
                  dt_bias=torch.zeros(heads * dim, device='cuda'),
                  num_heads=heads, head_dim=dim, lower_bound=-5.0)

    def forward(inputs, slot, accepted_history, checkpoint=None, decode=False):
        count = inputs['mixed_qkv'].size(1)
        metadata = SimpleNamespace(is_decoding=decode, num_spec_tokens=draft, is_init=None,
                                   state_ids=torch.tensor([slot], device='cuda'),
                                   valid_state=torch.tensor([True], device='cuda'),
                                   cache_seqlens=torch.tensor([accepted_history], dtype=torch.int32, device='cuda'),
                                   cu_seqlens=torch.tensor([0, count], dtype=torch.int64, device='cuda'))
        return impl.forward(**inputs, metadata=metadata, checkpoint=checkpoint, **kwargs)

    step = history + offset
    checkpoint = KdaPrefillCheckpoint(offset, torch.tensor([offset], dtype=torch.int64, device='cuda'),
                                     conv[3:4], state[3:4], step=step)
    full = forward(values, 1, history, checkpoint)
    prefix = forward({name: value[:, :offset] for name, value in values.items()}, 2, history)
    window = (step + torch.arange(-width, 0, device='cuda')) % conv.size(-1)
    torch.testing.assert_close(conv[3, :, window], conv[2, :, window], atol=0, rtol=0)
    torch.testing.assert_close(state[3, step % ring], state[2, step % ring], atol=0, rtol=0)
    untouched = [index for index in range(ring) if index != step % ring]
    assert torch.isnan(state[3, untouched]).all()
    conv[4].copy_(conv[3])
    state[4].copy_(state[3])
    saved_conv, saved_state = conv[3].clone(), state[3].clone()
    tail = {name: value[:, offset:] for name, value in values.items()}
    suffix = forward(tail, 2, step)
    restored = forward(tail, 4, step)
    torch.testing.assert_close(full, torch.cat((prefix, suffix), dim=1), atol=0, rtol=0)
    torch.testing.assert_close(suffix, restored, atol=0, rtol=0)
    end = history + length
    torch.testing.assert_close(state[1, end % ring], state[2, end % ring], atol=0, rtol=0)
    torch.testing.assert_close(state[4, end % ring], state[2, end % ring], atol=0, rtol=0)
    window = (end + torch.arange(-width, 0, device='cuda')) % conv.size(-1)
    torch.testing.assert_close(conv[1, :, window], conv[2, :, window], atol=0, rtol=0)

    for slot in (5, 6):
        conv[slot].copy_(conv[3])
        state[slot].copy_(state[3])
    accepted_history = step
    for accepted in (1, draft, ring, 1):
        verify = dict(mixed_qkv=torch.randn(1, ring, channels, dtype=torch.bfloat16, device='cuda') / 4,
                      raw_gate=torch.randn(1, ring, heads * dim, dtype=torch.bfloat16, device='cuda'),
                      raw_beta=torch.randn(1, ring, heads, dtype=torch.bfloat16, device='cuda'))
        actual = forward(verify, 5, accepted_history, decode=True)
        expected = [forward({name: value[:, index:index + 1] for name, value in verify.items()},
                            6, accepted_history + index, decode=True) for index in range(accepted)]
        torch.testing.assert_close(actual[:, :accepted], torch.cat(expected, dim=1), rtol=1e-3, atol=1e-3)
        accepted_history += accepted
        torch.testing.assert_close(state[5, accepted_history % ring], state[6, accepted_history % ring],
                                   rtol=1e-4, atol=1e-4)
        window = (accepted_history + torch.arange(-width, 0, device='cuda')) % conv.size(-1)
        torch.testing.assert_close(conv[5, :, window], conv[6, :, window], atol=0, rtol=0)
    torch.testing.assert_close(conv[3], saved_conv, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(state[3], saved_state, atol=0, rtol=0, equal_nan=True)
    assert torch.isnan(conv_pool[:, 0]).all() and torch.isnan(state_pool[:, 0]).all()
    assert torch.isnan(conv[0]).all() and torch.isnan(state[0]).all()
