# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""TurboMind float-lane linear modules (NVFP4 and CT FP8).

Kernel-level numeric proof on SM86: checkpoint-layout bundles packed through
the PT route reconstruct the fp32 reference. Loader tests cover the
checkpoint-suffix parameter names and the per-shard global-scale routing.
"""
import pytest
import torch

from lmdeploy.pytorch.nn.linear.tm_quant import MergedTmQuantLinear, TmQuantLinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant_nvfp4(codes_packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Fp32 reconstruction of a raw NVFP4 bundle -> [N, K]."""
    n, k2 = codes_packed.shape
    lo = (codes_packed & 0xF).long()
    hi = (codes_packed >> 4).long()
    table = torch.tensor(_E2M1_VALUES, device=codes_packed.device)
    v_lo = table[lo & 0x7] * torch.where((lo & 8) > 0, -1.0, 1.0)
    v_hi = table[hi & 0x7] * torch.where((hi & 8) > 0, -1.0, 1.0)
    codes = torch.stack((v_lo, v_hi), dim=-1).view(n, k2 * 2)
    expanded = scales.float().repeat_interleave(16, dim=-1)
    return codes * expanded / global_scale


def _quantize_nvfp4(w_dense: torch.Tensor):
    """Quantize a [N, K] dense weight to the raw checkpoint bundle.

    Follows the compressed-tensors nvfp4-pack-quantized convention (same
    construction as the TM-side native test): per-tensor global scale is
    the divisor ``2688 / amax_w``, e4m3 block scales ``448 * amax_g /
    amax_w``, two codes per byte with the even index in the low nibble.
    """
    n, k = w_dense.shape
    groups = w_dense.view(n, k // 16, 16)
    amax_w = w_dense.abs().amax().clamp_min(1e-8)
    scale = (groups.abs().amax(dim=-1, keepdim=True) * (448.0 / amax_w)).clamp_max(448.0)
    scale = scale.to(torch.float8_e4m3fn).float()
    scaled = groups / scale

    grid = torch.tensor(_E2M1_VALUES, device=w_dense.device)
    mag = torch.argmin((scaled.unsqueeze(-1) - grid).abs(), dim=-1)
    sign = (scaled < 0).to(torch.uint8)
    codes = ((mag.to(torch.uint8) & 0x7) | (sign << 3)).view(n, k)
    bits = codes[..., 0::2] | (codes[..., 1::2] << 4)
    global_scale = (2688.0 / amax_w).float()
    return bits, scale.view(n, k // 16).to(torch.float8_e4m3fn), global_scale


class TestNvfp4Linear:

    def test_plain_matches_reference(self):
        torch.manual_seed(0)
        K, N = 512, 256
        module = TmQuantLinear(K, N, lane='nvfp4', bias=False, device=torch.device('cuda')).cuda()
        bits, scale, gs = _quantize_nvfp4(torch.randn(N, K) * 0.05)
        module.weight_packed.data.copy_(bits)
        module.weight_scale.data.copy_(scale)
        module.weight_global_scale.data[0] = gs
        module.update_weights()

        x = torch.randn(8, K, dtype=torch.bfloat16, device='cuda')
        out = module(x)
        w_ref = _dequant_nvfp4(bits.cuda(), scale.cuda(), gs.cuda())
        ref = x.float() @ w_ref.T
        err = (out.float() - ref).abs().max().item()
        assert out.dtype == torch.bfloat16
        assert err < 1e-2 * ref.abs().max().item() + 0.05, err

    def test_param_names_match_checkpoint_suffixes(self):
        module = TmQuantLinear(256, 64, lane='nvfp4', bias=False, device=torch.device('cuda'))
        names = dict(module.named_parameters())
        assert 'weight_packed' in names and 'weight_scale' in names
        assert 'weight_global_scale' in names
        assert names['weight_packed'].dtype == torch.uint8
        assert names['weight_packed'].shape == (64, 128)
        assert names['weight_scale'].dtype == torch.float8_e4m3fn
        assert names['weight_scale'].shape == (64, 16)
        assert names['weight_global_scale'].shape == (1, )

    def test_merged_keeps_per_shard_global_scales(self):
        torch.manual_seed(0)
        K, N = 256, 128
        module = MergedTmQuantLinear(K, [N, N], lane='nvfp4', bias=False, device=torch.device('cuda')).cuda()
        assert module.num_shards == 2
        bundles = []
        for shard in range(2):
            bits, scale, gs = _quantize_nvfp4(torch.randn(N, K) * 0.05)
            module.weight_packed.data.split(module.all_out_features, 0)[shard].copy_(bits.cuda())
            module.weight_scale.data.split(module.all_out_features, 0)[shard].copy_(scale.cuda())
            module.weight_global_scale.data[shard] = gs
            bundles.append((bits.cuda(), scale.cuda(), gs.cuda(), gs.item()))
        # loader routing: shard ids map into the fused parameter rows
        module.global_scale_loader(module.weight_global_scale, torch.tensor(4.5), shard_id=0)
        module.global_scale_loader(module.weight_global_scale, torch.tensor(8.0), shard_id=1)
        assert module.weight_global_scale.tolist() == [4.5, 8.0]
        bundles[0] = (bundles[0][0], bundles[0][1], bundles[0][2], 4.5)
        bundles[1] = (bundles[1][0], bundles[1][1], bundles[1][2], 8.0)
        module.update_weights()

        x = torch.randn(4, K, dtype=torch.bfloat16, device='cuda')
        out = module(x)
        refs = [x.float() @ _dequant_nvfp4(b, s, g.cuda()).T for b, s, g, _ in bundles]
        ref = torch.cat(refs, dim=-1)
        err = (out.float() - ref).abs().max().item()
        assert err < 1e-2 * ref.abs().max().item() + 0.05, err


class TestCtFp8Linear:

    def test_plain_per_channel_matches_reference(self):
        torch.manual_seed(0)
        K, N = 512, 256
        module = TmQuantLinear(K, N, lane='ct_fp8', bias=False, device=torch.device('cuda')).cuda()
        w = (torch.randn(N, K) * 0.05).to(torch.float8_e4m3fn).cuda()
        s = (torch.rand(N, 1).bfloat16() * 0.01 + 0.005).cuda()
        module.weight.data.copy_(w)
        module.weight_scale.data.copy_(s)
        module.update_weights()

        x = torch.randn(8, K, dtype=torch.bfloat16, device='cuda')
        out = module(x)
        ref = x.float() @ w.float().T * s.float().T
        err = (out.float() - ref).abs().max().item()
        assert out.dtype == torch.bfloat16
        assert err < 1e-2 * ref.abs().max().item() + 0.05, err

    def test_blocked_lane_matches_reference(self):
        torch.manual_seed(0)
        K, N = 512, 256
        module = TmQuantLinear(K, N, lane='ct_fp8', fp8_blocked=True, bias=False,
                               device=torch.device('cuda')).cuda()
        assert module.weight_scale.shape == (2, 4)  # [N/128, K/128]
        w = (torch.randn(N, K) * 0.05).to(torch.float8_e4m3fn).cuda()
        # Blocked checkpoint scales: [ceil(N/128), ceil(K/128)], one scale
        # per output block per input group.
        s = (torch.rand(N // 128, K // 128).bfloat16() * 0.01 + 0.005).cuda()
        module.weight.data.copy_(w)
        module.weight_scale.data.copy_(s)
        module.update_weights()

        x = torch.randn(8, K, dtype=torch.bfloat16, device='cuda')
        out = module(x)
        expanded = s.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:N, :K]
        ref = x.float() @ (w.float() * expanded).T
        err = (out.float() - ref).abs().max().item()
        assert err < 1e-2 * ref.abs().max().item() + 0.05, err

    def test_param_names_match_checkpoint_suffixes(self):
        module = TmQuantLinear(256, 64, lane='ct_fp8', bias=False, device=torch.device('cuda'))
        names = dict(module.named_parameters())
        assert 'weight' in names and 'weight_scale' in names
        assert names['weight'].dtype == torch.float8_e4m3fn
        assert names['weight_scale'].shape == (64, 1)
        assert names['weight_scale'].dtype == torch.bfloat16

    def test_blocked_tp_is_rejected(self):
        with pytest.raises(NotImplementedError, match='Blocked CT FP8 TP sharding'):
            TmQuantLinear(256, 64, lane='ct_fp8', fp8_blocked=True, bias=False, device=torch.device('cuda'),
                          is_tp=True)


class TestEnvGates:

    def test_off_is_loud(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'nvfp4=off')
        with pytest.raises(RuntimeError, match='no native fallback'):
            TmQuantLinear(256, 64, lane='nvfp4', bias=False, device=torch.device('cuda'))

    def test_ct_fp8_off_is_loud(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'ct_fp8=off')
        with pytest.raises(RuntimeError, match='no native fallback'):
            TmQuantLinear(256, 64, lane='ct_fp8', bias=False, device=torch.device('cuda'))
