# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""TurboMind float-lane linear modules (NVFP4 and CT FP8).

Kernel-level numeric proof on SM86: checkpoint-layout bundles packed through
the PT route reconstruct the fp32 reference. Loader tests cover the
checkpoint-suffix parameter names and the per-shard global-scale routing.
"""
import pytest
import torch

from lmdeploy.pytorch.nn.linear.tm_quant import MergedTmQuantLinear, QKVTmQuantLinear, TmQuantLinear

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
    construction as the TM-side native test; keep the copies in lockstep):
    per-tensor global scale is the divisor ``2688 / amax_w``, e4m3 block
    scales ``448 * amax_g / amax_w``, two codes per byte with the even
    index in the low nibble.
    """
    n, k = w_dense.shape
    groups = w_dense.view(n, k // 16, 16)
    amax_w = w_dense.abs().amax().clamp_min(1e-8)
    scale = (groups.abs().amax(dim=-1, keepdim=True) * (448.0 / amax_w)).clamp_max(448.0)
    scale = scale.to(torch.float8_e4m3fn).float()
    # codes = round(w * gs / scale); the raw ``groups / scale`` (the
    # historical bug) collapses every magnitude code onto zero.
    scaled = groups * ((2688.0 / amax_w) / scale)

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

    def test_quantizer_produces_nonzero_codes(self):
        """Dropping the gs factor (the historical bug) collapsed every
        magnitude code onto zero, making the kernel proof above vacuous."""
        torch.manual_seed(0)
        w = torch.randn(64, 512) * 0.05
        bits, scale, gs = _quantize_nvfp4(w)
        codes = torch.stack(((bits & 0xF).long(), (bits >> 4).long()), dim=-1).view(64, 512)
        nz_mag = ((codes & 0x7) != 0).float().mean().item()
        assert nz_mag > 0.3, nz_mag
        w_ref = _dequant_nvfp4(bits, scale, gs)
        assert w_ref.abs().max() > 0

    def test_plain_wrong_global_scale_is_loud(self):
        """Reloading the same bundle with a wrong scale must move the output
        far from the reference; with all-zero codes this passes vacuously."""
        torch.manual_seed(0)
        K, N = 512, 256
        module = TmQuantLinear(K, N, lane='nvfp4', bias=False, device=torch.device('cuda')).cuda()
        bits, scale, gs = _quantize_nvfp4(torch.randn(N, K) * 0.05)
        module.weight_packed.data.copy_(bits)
        module.weight_scale.data.copy_(scale)
        module.weight_global_scale.data[0] = gs
        module.update_weights()

        x = torch.randn(8, K, dtype=torch.bfloat16, device='cuda')
        ref = x.float() @ _dequant_nvfp4(bits.cuda(), scale.cuda(), gs.cuda()).T
        out = module(x)
        err_ok = (out.float() - ref).abs().max().item()
        assert err_ok < 1e-2 * ref.abs().max().item() + 0.05, err_ok

        # Same codes, epilogue factor off by 8x.
        module.weight_global_scale.data[0] = gs * 8
        module.update_weights()
        err_wrong = (module(x).float() - ref).abs().max().item()
        assert err_wrong > 0.5 * ref.abs().max().item(), (err_ok, err_wrong)

    def test_gs_less_bundle_raises_at_prepare(self):
        """A checkpoint bundle with no global scale for a shard must fail
        loudly at update_weights; previously the never-loaded slot kept its
        initializer garbage and the epilogue silently skipped the multiply
        (output off by gs)."""
        K, N = 512, 256
        module = TmQuantLinear(K, N, lane='nvfp4', bias=False, device=torch.device('cuda')).cuda()
        bits, scale, gs = _quantize_nvfp4(torch.randn(N, K) * 0.05)
        module.weight_packed.data.copy_(bits)
        module.weight_scale.data.copy_(scale)
        # No gs load: the slot stays at its NaN initializer.
        assert not bool(torch.isfinite(module.weight_global_scale).all())
        with pytest.raises(ValueError, match='uninitialized'):
            module.update_weights()

        # Loading the scale clears the gate.
        module.weight_global_scale.data[0] = gs
        module.update_weights()

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
            bits, scale, _ = _quantize_nvfp4(torch.randn(N, K) * 0.05)
            module.weight_packed.data.split(module.all_out_features, 0)[shard].copy_(bits.cuda())
            module.weight_scale.data.split(module.all_out_features, 0)[shard].copy_(scale.cuda())
            bundles.append((bits.cuda(), scale.cuda()))
        # loader routing: shard ids map into the fused parameter rows and
        # each shard's GEMM applies its own slot's scale; with collapsed
        # codes this proof used to pass vacuously
        module.global_scale_loader(module.weight_global_scale, torch.tensor(4.5), shard_id=0)
        module.global_scale_loader(module.weight_global_scale, torch.tensor(8.0), shard_id=1)
        assert module.weight_global_scale.tolist() == [4.5, 8.0]
        module.update_weights()

        x = torch.randn(4, K, dtype=torch.bfloat16, device='cuda')
        refs = [
            x.float() @ _dequant_nvfp4(b, s, torch.tensor(g).cuda()).T
            for (b, s), g in zip(bundles, module.weight_global_scale.tolist())
        ]
        ref = torch.cat(refs, dim=-1)
        out = module(x)
        err = (out.float() - ref).abs().max().item()
        assert err < 1e-2 * ref.abs().max().item() + 0.05, err


class TestQkvGlobalScale:
    """A fused QKV checkpoint module carries ONE per-tensor global scale; every
    shard that packs it must receive it."""

    def _make_module(self):
        K = 256
        module = QKVTmQuantLinear(K, 4, 2, 32, 32, lane='nvfp4', bias=False, device=torch.device('cuda')).cuda()
        assert list(module.get_qkv_out_feautures()) == [128, 64, 64]
        return module, K

    def test_fused_single_global_scale_reaches_all_shards(self):
        torch.manual_seed(0)
        module, K = self._make_module()
        sections = module.get_qkv_out_feautures()
        bits, scale, gs = _quantize_nvfp4(torch.randn(sum(sections), K) * 0.05)
        module.weight_packed.data.copy_(bits)
        module.weight_scale.data.copy_(scale)
        # fused checkpoint style: one key, no shard id
        module.global_scale_loader(module.weight_global_scale, gs)
        assert module.weight_global_scale.tolist() == pytest.approx([gs.item()] * 3)
        module.update_weights()

        x = torch.randn(4, K, dtype=torch.bfloat16, device='cuda')
        out = module(x)
        ref = x.float() @ _dequant_nvfp4(bits.cuda(), scale.cuda(), gs.cuda()).T
        assert torch.isfinite(out).all()
        err = (out.float() - ref).abs().max().item()
        assert err < 1e-2 * ref.abs().max().item() + 0.05, err

    def test_shard_id_routing_stays_per_slot(self):
        """Unfused checkpoints (separate q/k/v keys) keep their distinct per-
        tensor scales in their own slots."""
        module, _ = self._make_module()
        module.global_scale_loader(module.weight_global_scale, torch.tensor(4.5), shard_id='q')
        module.global_scale_loader(module.weight_global_scale, torch.tensor(8.0), shard_id='k')
        module.global_scale_loader(module.weight_global_scale, torch.tensor(16.0), shard_id='v')
        assert module.weight_global_scale.tolist() == [4.5, 8.0, 16.0]


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


class TestCpuResidency:
    """B6: the checkpoint-layout quant params are the CPU source of truth
    (single occupancy); module moves must not re-upload them, and packing
    goes through the channel's device-agnostic prepare_weight."""

    def test_params_stay_cpu_through_module_moves(self):
        module = TmQuantLinear(256, 64, lane='nvfp4', bias=True, device=torch.device('cuda')).cuda()
        assert module.cpu_resident
        assert module.weight_packed.device.type == 'cpu'
        assert module.weight_scale.device.type == 'cpu'
        assert module.weight_global_scale.device.type == 'cpu'
        # bias travels with the module: forward adds it python-side
        assert module.bias.device.type == 'cuda'

    def test_fp8_lane_params_stay_cpu(self):
        module = TmQuantLinear(256, 64, lane='ct_fp8', bias=False, device=torch.device('cuda')).cuda()
        assert module.weight.device.type == 'cpu'
        assert module.weight_scale.device.type == 'cpu'
