# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch


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
