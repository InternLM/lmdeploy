# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Tests for the TurboMind GEMM W4A16 linear route.

All cases run on tiny random weights; no real checkpoint is needed. The registry needs a CUDA device, so registry-backed
cases skip without one.
"""

import pytest
import torch
import torch.nn.functional as F

from lmdeploy import turbomind

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

import lmdeploy.pytorch.envs as envs
from lmdeploy.pytorch.backends.awq_modules import LinearW4A16BuildSpec
from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl
from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl, _build_linear_tm_gemm
from lmdeploy.pytorch.backends.tm_gemm_modules import LinearTmGemmBuildSpec

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

SUPPORTED = dict(in_features=512, out_features=256, w_bit=4, group_size=128)
UNSUPPORTED = dict(in_features=512, out_features=254, w_bit=4, group_size=128)

ZERO_PACKED = 0x88888888 - (1 << 32)  # AWQ qzeros encoding of zero point 0


def _random_awq(k, n, group_size=128, seed=0):
    """Build random AWQ tensors in the canonical checkpoint layout."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    qweight = torch.randint(-(1 << 31), (1 << 31) - 1, (k, n // 8), dtype=torch.int32, generator=generator)
    scales = torch.randn(k // group_size, n, dtype=torch.float16, generator=generator)
    qzeros = torch.full((k // group_size, n // 8), ZERO_PACKED, dtype=torch.int32)
    bias = torch.randn(n, dtype=torch.float16, generator=generator)
    return qweight.cuda(), scales.cuda(), qzeros.cuda(), bias.cuda()


def _softmax_kl(a, b):
    """KL(P||Q) between the softmax distributions of two logit tensors."""
    p = torch.log_softmax(a.float(), dim=-1)
    q = torch.log_softmax(b.float(), dim=-1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def _supported_on_device(k, n, group_size=128):
    """Whether the registry probe accepts the shape on this device."""
    from lmdeploy.pytorch.backends.cuda.tm_probe import turbomind_support_reason

    return turbomind_support_reason(k, n, 4, group_size, torch.float16) is None


def _fusion_available():
    """Whether the selected family exposes a SiLU gate/up epilogue."""
    from lmdeploy.turbomind.linear import get_linear
    from lmdeploy.turbomind.weight_format import AWQFormat

    k, n, group_size = 512, 256, 128
    qw, sc, qz, _ = _random_awq(k, n, group_size)
    plan = get_linear().get_weight_plan(
        weight_format=AWQFormat(block_in=group_size), dtype=sc.dtype, fusion_type='silu'
    )
    weight = None
    try:
        weight = get_linear().fuse_weight((qw, qw), plan=plan, scales=(sc, sc), zeros=(qz, qz))
    except NotImplementedError:
        return False
    finally:
        if weight is not None:
            weight.close()
    return True


# ---------------------------------------------------------------------------
# S1: dual-path numeric comparison against the AWQ Triton kernel.
# ---------------------------------------------------------------------------


@cuda_required
@pytest.mark.parametrize('k,n,m', [(512, 256, 1), (512, 256, 4), (1024, 512, 4)])
def test_s1_dual_path_matches_triton(k, n, m):
    """TurboMind GEMM and the AWQ Triton kernel agree on logits."""
    if not _supported_on_device(k, n):
        pytest.skip('registry rejects the shape on this device')

    qw, sc, qz, bias = _random_awq(k, n, seed=k + n)
    x = torch.randn(m, k, dtype=torch.float16, device='cuda')

    impl_triton = AwqLinearW4A16Impl(k, n, 4, 128)
    out_triton = impl_triton.forward(x, qw, sc, qz, bias=bias).squeeze(0)

    impl_tm = TurbomindGemmW4A16Impl(k, n, 128)
    impl_tm.update_weights(qw, sc, qz)
    with torch.no_grad():
        out_tm = impl_tm.forward(x, qw, sc, qz, bias=bias).squeeze(0)

    kl = _softmax_kl(out_triton, out_tm)
    agreement = (out_triton.argmax(-1) == out_tm.argmax(-1)).float().mean().item()
    max_abs = (out_triton.float() - out_tm.float()).abs().max().item()
    # Random-weight draws make the softmax flatter than real checkpoints, so
    # the real-weight threshold (1.9e-4) is relaxed here; the real-weight
    # golden gate reports the strict numbers.
    assert agreement >= 0.99, f"argmax agreement {agreement:.4f}, KL {kl:.3e}"
    assert kl <= 1e-2, f"KL {kl:.3e}"
    assert max_abs <= 1.0, f"max abs diff {max_abs:.3e}"


# ---------------------------------------------------------------------------
# gate/up fusion.
# ---------------------------------------------------------------------------


@cuda_required
def test_fuse_on_off_consistency():
    """The fused SiLU epilogue equals composing two unfused projections."""
    if not _supported_on_device(512, 256):
        pytest.skip('registry rejects the shape on this device')
    if not _fusion_available():
        # The SiLU epilogue ships with the SM90 kernel families only; the
        # SM80/SIMT families used on e.g. RTX 3090 reject fusion. Coverage is
        # exercised on the SM90 acceptance hardware.
        pytest.skip('selected family cannot represent SiLU fusion')

    k, n, group_size = 512, 256, 128
    gate_qw, gate_sc, gate_qz, _ = _random_awq(k, n, group_size, seed=1)
    up_qw, up_sc, up_qz, _ = _random_awq(k, n, group_size, seed=2)
    x = torch.randn(4, k, dtype=torch.float16, device='cuda')

    fused = TurbomindGemmW4A16Impl(k, n, group_size, fusion_type='silu')
    fused.update_weights((gate_qw, up_qw), (gate_sc, up_sc), (gate_qz, up_qz))
    out_fused = fused.forward(x, None, None, None).squeeze(0)

    gate_impl = TurbomindGemmW4A16Impl(k, n, group_size)
    gate_impl.update_weights(gate_qw, gate_sc, gate_qz)
    up_impl = TurbomindGemmW4A16Impl(k, n, group_size)
    up_impl.update_weights(up_qw, up_sc, up_qz)
    expected = F.silu(gate_impl.forward(x, gate_qw, gate_sc, gate_qz).squeeze(0)) * up_impl.forward(
        x, up_qw, up_sc, up_qz
    ).squeeze(0)

    assert torch.allclose(out_fused.float(), expected.float(), atol=2e-3, rtol=2e-3)


@cuda_required
def test_fused_bias_rejected_on_shape_level():
    """A fused spec keeps the no-bias contract so bias cannot be misapplied."""
    from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import _build_linear_tm_gemm

    spec = LinearTmGemmBuildSpec(
        in_features=512,
        out_features=512,
        w_bit=4,
        group_size=128,
        bias=True,
        output_dtype=torch.float16,
        fusion_type='silu',
    )
    with pytest.raises(ValueError, match='bias'):
        _build_linear_tm_gemm(spec)


# ---------------------------------------------------------------------------
# spill and hot update.
# ---------------------------------------------------------------------------


@cuda_required
def test_spill_frees_gpu_and_hot_update_restores():
    """Spilling drops allocated GPU bytes; restore keeps values bit-exact."""
    if not _supported_on_device(512, 256):
        pytest.skip('registry rejects the shape on this device')

    k, n, group_size = 2048, 2048, 128
    qw, sc, qz, _ = _random_awq(k, n, group_size, seed=3)
    x = torch.randn(4, k, dtype=torch.float16, device='cuda')

    impl = TurbomindGemmW4A16Impl(k, n, group_size)
    impl.update_weights(qw, sc, qz)
    out_first = impl.forward(x, qw, sc, qz).squeeze(0).cpu()
    torch.cuda.synchronize()

    # Mirror the module flow: the parameters own the storages, the raw
    # tensors go away, then spilling frees the GPU bytes.
    from lmdeploy.pytorch.backends.cuda.tm_weight_release import TorchWeightSpill

    expected_bytes = qw.numel() * 4 + sc.numel() * 2 + qz.numel() * 4
    qw_cpu, sc_cpu, qz_cpu = qw.cpu(), sc.cpu(), qz.cpu()
    params = {
        'qweight': torch.nn.Parameter(qw, requires_grad=False),
        'scales': torch.nn.Parameter(sc, requires_grad=False),
        'qzeros': torch.nn.Parameter(qz, requires_grad=False),
    }
    qw = sc = qz = None
    spill = TorchWeightSpill()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    spill.spill(params)
    torch.cuda.synchronize()

    freed = baseline - torch.cuda.memory_allocated()
    assert freed >= expected_bytes, f"freed only {freed} of {expected_bytes} bytes"
    assert spill.resident_bytes == expected_bytes
    # Forward keeps working from the packed handle; placeholders are ignored.
    out_spilled = impl.forward(x, params['qweight'], params['scales'], params['qzeros']).squeeze(0)
    assert torch.equal(out_first, out_spilled.cpu())

    # A hot-update loader touch restores the module bit-exactly.
    spill.restore()
    torch.cuda.synchronize()
    assert torch.equal(params['qweight'].data.cpu(), qw_cpu)
    assert torch.equal(params['scales'].data.cpu(), sc_cpu)
    assert torch.equal(params['qzeros'].data.cpu(), qz_cpu)

    # Re-pack after restore keeps the numerics stable.
    impl.update_weights(params['qweight'].data, params['scales'].data, params['qzeros'].data)
    out_repaked = impl.forward(x, params['qweight'], params['scales'], params['qzeros']).squeeze(0)
    assert torch.equal(out_first, out_repaked.cpu())


@cuda_required
def test_hot_update_syncs_and_repacks():
    """A second update_weights keeps the sync and reproduces the same
    logits."""
    if not _supported_on_device(512, 256):
        pytest.skip('registry rejects the shape on this device')

    k, n, group_size = 512, 256, 128
    qw, sc, qz, _ = _random_awq(k, n, group_size, seed=4)
    x = torch.randn(2, k, dtype=torch.float16, device='cuda')

    impl = TurbomindGemmW4A16Impl(k, n, group_size)
    impl.update_weights(qw, sc, qz)
    assert impl._published
    out_first = impl.forward(x, qw, sc, qz).squeeze(0).cpu()

    impl.update_weights(qw, sc, qz)  # hot update re-pack
    out_second = impl.forward(x, qw, sc, qz).squeeze(0).cpu()
    assert torch.equal(out_first, out_second)


@cuda_required
def test_forward_converts_non_fp16_activations():
    """Non-fp16 activations keep the AWQ Triton contract: fp16 compute in,
    caller dtype out."""
    if not _supported_on_device(512, 256):
        pytest.skip('registry rejects the shape on this device')

    k, n, group_size = 512, 256, 128
    qw, sc, qz, _ = _random_awq(k, n, group_size, seed=6)
    x_fp16 = torch.randn(4, k, dtype=torch.float16, device='cuda')
    x_bf16 = x_fp16.bfloat16()

    impl = TurbomindGemmW4A16Impl(k, n, group_size)
    impl.update_weights(qw, sc, qz)
    with torch.no_grad():
        # The documented contract: non-fp16 input is converted to fp16 for
        # the native call and the output is cast back to the caller dtype.
        manual = impl.forward(x_bf16.half(), qw, sc, qz).squeeze(0).bfloat16()
        out_bf16 = impl.forward(x_bf16, qw, sc, qz).squeeze(0)

    assert out_bf16.dtype == torch.bfloat16
    assert torch.equal(out_bf16, manual)


# ---------------------------------------------------------------------------
# batched rejection, bias, and builder routing.
# ---------------------------------------------------------------------------


@cuda_required
def test_batched_forward_explicitly_rejected():
    """supports_batched is False and forward_batched raises."""
    assert TurbomindGemmW4A16Impl.supports_batched is False
    impl = TurbomindGemmW4A16Impl(512, 256, 128)
    x = torch.randn(2, 3, 512, dtype=torch.float16, device='cuda')
    weight = torch.randn(3, 256, 512, dtype=torch.float16, device='cuda')
    with pytest.raises(RuntimeError):
        impl.forward_batched(x, weight)


@cuda_required
def test_bias_added_per_layer():
    """Bias is added in python after the kernel call."""
    if not _supported_on_device(512, 256):
        pytest.skip('registry rejects the shape on this device')

    k, n, group_size = 512, 256, 128
    qw, sc, qz, bias = _random_awq(k, n, group_size, seed=5)
    x = torch.randn(4, k, dtype=torch.float16, device='cuda')

    impl = TurbomindGemmW4A16Impl(k, n, group_size)
    impl.update_weights(qw, sc, qz)
    out_plain = impl.forward(x, qw, sc, qz).squeeze(0)
    out_bias = impl.forward(x, qw, sc, qz, bias=bias).squeeze(0)
    assert torch.allclose(out_bias.float(), out_plain.float() + bias.float(), atol=0.5)


def _tm_spec(**kwargs):
    params = dict(bias=False, output_dtype=torch.float16)
    params.update(kwargs)
    return LinearTmGemmBuildSpec(**params)


def _set_backend(monkeypatch, value):
    monkeypatch.setattr(envs, 'w4a16_gemm_backend', value)
    # An exported LMDEPLOY_TM_KERNELS would win over the legacy attribute;
    # clear it so the tests exercise exactly the patched value.
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', '')


@cuda_required
@pytest.mark.parametrize('backend', ['auto', 'turbomind'])
def test_builder_selects_turbomind(monkeypatch, backend):
    """A registry-supported layer builds the TurboMind implementation."""
    _set_backend(monkeypatch, backend)
    impl = _build_linear_tm_gemm(_tm_spec(**SUPPORTED))
    assert isinstance(impl, TurbomindGemmW4A16Impl)


@cuda_required
def test_builder_auto_falls_back_to_triton(monkeypatch):
    """Auto falls back to the AWQ Triton kernel when the probe rejects."""
    _set_backend(monkeypatch, 'auto')
    impl = _build_linear_tm_gemm(_tm_spec(**UNSUPPORTED))
    assert isinstance(impl, AwqLinearW4A16Impl)


@cuda_required
def test_builder_forced_raises_on_unsupported(monkeypatch):
    """A forced backend raises on an unsupported layer."""
    _set_backend(monkeypatch, 'turbomind')
    with pytest.raises(RuntimeError):
        _build_linear_tm_gemm(_tm_spec(**UNSUPPORTED))


@cuda_required
def test_builder_other_value_keeps_triton(monkeypatch):
    """Any other env value keeps the Triton kernel without probing."""
    _set_backend(monkeypatch, 'triton')
    impl = _build_linear_tm_gemm(_tm_spec(**SUPPORTED))
    assert isinstance(impl, AwqLinearW4A16Impl)


@cuda_required
def test_w4a16_dispatch_unchanged(monkeypatch):
    """The legacy W4A16 spec routes through the shared builder to the unified
    TurboMind implementation."""
    from lmdeploy.pytorch.backends import get_backend

    _set_backend(monkeypatch, 'auto')
    spec = LinearW4A16BuildSpec(
        in_features=512, out_features=256, w_bit=4, group_size=128, bias=False, output_dtype=torch.float16
    )
    impl = get_backend().build_op(spec)
    assert isinstance(impl, TurbomindGemmW4A16Impl)
    spec_tm = LinearTmGemmBuildSpec(
        in_features=512, out_features=256, w_bit=4, group_size=128, bias=False, output_dtype=torch.float16
    )
    impl_tm = get_backend().build_op(spec_tm)
    assert isinstance(impl_tm, TurbomindGemmW4A16Impl)


# ---------------------------------------------------------------------------
# nn module: pack, spill, hot update through the real loaders.
# ---------------------------------------------------------------------------


@cuda_required
def test_nn_module_cpu_direct_and_hot_update():
    """TmGemmLinear packs CPU-direct: the raw weights never occupy device
    memory, hot updates write through the loaders into the host buffers."""
    from lmdeploy.pytorch.nn.linear.tm_gemm import TmGemmLinear

    k, n, group_size = 512, 256, 128
    qw, sc, qz, bias = _random_awq(k, n, group_size, seed=6)
    module = TmGemmLinear(k, n, 4, group_size, bias=True, device='cuda')
    # The TurboMind route routes the packed storages to the host at build
    # time; bias stays on the module device for forward.
    assert module.qweight.device.type == 'cpu'
    assert module.scales.device.type == 'cpu'
    assert module.qzeros.device.type == 'cpu'
    assert module.bias.device.type == 'cuda'
    module.qweight.data.copy_(qw)
    module.scales.data.copy_(sc)
    module.qzeros.data.copy_(qz)
    module.bias.data.copy_(bias)
    x = torch.randn(4, k, dtype=torch.float16, device='cuda')

    baseline = torch.cuda.memory_allocated()
    module.update_weights()
    torch.cuda.synchronize()

    expected_bytes = qw.numel() * 4 + sc.numel() * 2 + qz.numel() * 4
    grown = torch.cuda.memory_allocated() - baseline
    assert grown < expected_bytes // 4, (
        f'torch allocations grew by {grown} bytes while packing; the raw '
        'weights must stay host-resident (the packed handle is a native '
        'allocation outside the torch allocator)')
    assert module.qweight.data.numel() != 0  # CPU storages stay authoritative

    out_first = module._forward_default(x, all_reduce=False, tp_sizes=None).cpu()

    # A hot update writes through the loader straight into the CPU buffers;
    # no spill restore exists or is needed.
    module.qweight.weight_loader(module.qweight, qw.cpu())
    torch.cuda.synchronize()
    assert module.qweight.data.shape == qw.shape
    assert torch.equal(module.qweight.data.cpu(), qw.cpu())
    assert module.scales.data.shape == sc.shape

    module.update_weights()
    out_repaked = module._forward_default(x, all_reduce=False, tp_sizes=None).cpu()
    assert torch.equal(out_first, out_repaked)


# ---------------------------------------------------------------------------
# sleep / wakeup: release the native handle, restore, and re-pack.
# ---------------------------------------------------------------------------


@cuda_required
def test_impl_sleep_release_frees_packed_handle():
    """sleep_release closes the native weight; nothing else frees it."""
    if not _supported_on_device(512, 256):
        pytest.skip('registry rejects the shape on this device')

    k, n, group_size = 512, 256, 128
    qw, sc, qz, _ = _random_awq(k, n, group_size, seed=8)

    impl = TurbomindGemmW4A16Impl(k, n, group_size)
    impl.update_weights(qw, sc, qz)
    torch.cuda.synchronize()
    assert impl._weight is not None

    impl.sleep_release()
    torch.cuda.synchronize()
    # The packed weight is a native allocation outside the torch allocator;
    # the observable release is the closed handle itself.
    assert impl._weight is None
    assert impl._linear is None

    # A released impl repacks from the restored sources on wakeup.
    impl.sleep_wakeup(qw, sc, qz)
    x = torch.randn(4, k, dtype=torch.float16, device='cuda')
    out = impl.forward(x, qw, sc, qz).squeeze(0).cpu()

    reference = TurbomindGemmW4A16Impl(k, n, group_size)
    reference.update_weights(qw, sc, qz)
    expected = reference.forward(x, qw, sc, qz).squeeze(0).cpu()
    assert torch.equal(out, expected)


@cuda_required
def test_nn_module_sleep_wakeup_roundtrip():
    """A layer survives sleep: the host storages never move, wakeup re-packs
    from them and forward is bit-exact."""
    from lmdeploy.pytorch.nn.linear.tm_gemm import TmGemmLinear

    k, n, group_size = 512, 256, 128
    qw, sc, qz, bias = _random_awq(k, n, group_size, seed=9)
    module = TmGemmLinear(k, n, 4, group_size, bias=True, device='cuda')
    module.qweight.data.copy_(qw)
    module.scales.data.copy_(sc)
    module.qzeros.data.copy_(qz)
    module.bias.data.copy_(bias)
    x = torch.randn(4, k, dtype=torch.float16, device='cuda')

    module.update_weights()
    torch.cuda.synchronize()
    out_first = module._forward_default(x, all_reduce=False, tp_sizes=None).cpu()

    # Sleep: the native handle drops; the host storages stay put.
    module.sleep_release()
    torch.cuda.synchronize()
    assert module.impl._weight is None
    assert module.qweight.device.type == 'cpu' and module.qweight.data.numel() != 0

    # Wakeup (the engine moves the model to CUDA first): the CPU-direct
    # storages skip the move, and the re-pack reproduces forward bit-exact.
    module.to('cuda')
    assert module.qweight.device.type == 'cpu'
    module.sleep_wakeup()
    torch.cuda.synchronize()
    assert module.impl._weight is not None
    out_wakeup = module._forward_default(x, all_reduce=False, tp_sizes=None).cpu()
    assert torch.equal(out_first, out_wakeup)

    # Sleep/wakeup stays repeatable.
    module.sleep_release()
    module.sleep_wakeup()
    out_again = module._forward_default(x, all_reduce=False, tp_sizes=None).cpu()
    assert torch.equal(out_first, out_again)


@cuda_required
def test_nn_module_sleep_release_keeps_resident_weights():
    """An MLA-style module (release_torch_weights=False) keeps its torch copies
    resident.

    Its registered parameters travel with ``Module.to`` during sleep, so
    sleep_release drops only the native handle; wakeup re-packs from the
    restored resident copies without touching a spill.
    """
    from lmdeploy.pytorch.nn.linear.tm_gemm import TmGemmLinear

    k, n, group_size = 512, 256, 128
    qw, sc, qz, bias = _random_awq(k, n, group_size, seed=10)
    # release_torch_weights=False mirrors MLA-absorbed modules: update_weights
    # keeps the torch copies resident.
    module = TmGemmLinear(k, n, 4, group_size, bias=True, device='cuda', release_torch_weights=False)
    module.qweight.data.copy_(qw)
    module.scales.data.copy_(sc)
    module.qzeros.data.copy_(qz)
    module.bias.data.copy_(bias)

    module.update_weights()
    assert module.qweight.device.type == 'cuda'
    assert module.qweight.data.numel() != 0

    # The native handle drops, but the torch copies stay registered: they
    # travel with ``Module.to`` during sleep and feed the re-pack on wakeup.
    module.sleep_release()
    torch.cuda.synchronize()
    assert module.impl._weight is None
    assert module.qweight.device.type == 'cuda'
    assert module.qweight.data.numel() != 0

    module.sleep_wakeup()
    torch.cuda.synchronize()
    assert module.impl._weight is not None
    assert torch.equal(module.qweight.data.cpu(), qw.cpu())
    assert torch.equal(module.scales.data.cpu(), sc.cpu())
    assert torch.equal(module.qzeros.data.cpu(), qz.cpu())
