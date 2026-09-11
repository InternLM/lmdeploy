# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
    pytest.skip('requires Hopper SM90', allow_module_level=True)

pytest.importorskip('cutlass.cute', reason='requires optional nvidia-cutlass-dsl')
pytest.importorskip('cuda.bindings.driver', reason='requires optional cuda-python')


def weights(e, n, k):
    q = torch.randint(-8, 8, (e, n, k), device='cuda', dtype=torch.int32)
    shifts = torch.arange(8, device='cuda', dtype=torch.int64) * 4
    p = (((q.long() + 8).unflatten(-1, (-1, 8))) << shifts).sum(-1).int()
    s = (torch.rand(e, n, k // 32, device='cuda') * 0.04 + .001).bfloat16()
    ref = (q.float() * s.float().repeat_interleave(32, -1)).bfloat16()
    return p, s, ref


def nrmse(x, ref):
    return ((x.float() - ref.float()).square().sum() / ref.float().square().sum().clamp_min(1e-20)).sqrt().item()


@pytest.mark.parametrize('m,n,k,e,topk', [(1, 64, 128, 4, 2), (17, 96, 256, 8, 2), (9, 128, 7168, 4, 2),
                                          (3, 64, 64, 1, 1), (2, 64, 32, 2, 1)])
@pytest.mark.parametrize('stages', [1, 2])
@pytest.mark.parametrize('fast', [False, True])
def test_gemm(m, n, k, e, topk, stages, fast, record_property):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute import _launch_gemm, gather_routed
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_blocks
    torch.manual_seed(13)
    packed, s, w = weights(e, n, k)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    ids = torch.rand(m, e, device='cuda').topk(topk, -1).indices
    meta = _get_sorted_idx_blocks(ids, e, e, 0, 8)
    padded = gather_routed(x, meta, topk)
    out = x.new_empty((m * topk, n))
    _launch_gemm(padded, packed, s, out, meta, stages=stages, fast_dequant=fast)
    ref = torch.stack([x[r // topk].float() @ w[expert].float().T for r, expert in enumerate(ids.flatten())])
    err = nrmse(out, ref)
    record_property('bf16_reference_nrmse', err)
    assert err < .003


@pytest.mark.parametrize('m,topk,renormalize', [(1, 2, False), (17, 2, True), (4, 8, False), (65, 8, False),
                                                (0, 2, False)])
def test_moe_graph(m, topk, renormalize):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute import fused_moe_w4a16_cute
    torch.manual_seed(42)
    e, h, f = 8, 128, 64
    gp, gs, gw = weights(e, 2 * f, h)
    dp, ds, dw = weights(e, h, f)
    x = torch.randn(m, h, device='cuda', dtype=torch.bfloat16)
    ids = torch.rand(m, e, device='cuda').topk(topk, -1).indices.int()
    tw = torch.rand(m, topk, device='cuda')

    def run():
        return fused_moe_w4a16_cute(x, gp, gs, dp, ds, tw, ids, topk, renormalize)

    def reference():
        gu = torch.stack([x[r // topk].float() @ gw[expert].float().T
                          for r, expert in enumerate(ids.flatten())]).bfloat16()
        act = (F.silu(gu[:, :f].float()) * gu[:, f:].float()).bfloat16()
        y = torch.stack([act[r].float() @ dw[expert].float().T for r, expert in enumerate(ids.flatten())]).bfloat16()
        rw = tw / tw.sum(-1, keepdim=True) if renormalize else tw
        return (y.view(m, topk, h).float() * rw[..., None]).sum(1).bfloat16()

    out = run()
    if m == 0:
        assert out.shape == x.shape
        return
    assert nrmse(out, reference()) < .008
    assert nrmse(out, fused_moe_w4a16(x, gp, gs, dp, ds, tw, ids, topk, renormalize)) < .012
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = run()
    ids.copy_((ids + 3) % e)
    x.normal_()
    tw.uniform_()
    graph.replay()
    assert nrmse(gout, reference()) < .008


def test_zero_and_provider(monkeypatch):
    from dataclasses import replace

    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.backends.cuda.moe.compressed_tensors import CuteFusedMoEW4A16Impl, _build_fused_moe_w4a16
    from lmdeploy.pytorch.backends.moe import FusedMoEW4A16BuildSpec
    monkeypatch.setattr(envs, 'w4a16_moe_backend', 'cute')
    spec = FusedMoEW4A16BuildSpec(top_k=2,
                                  num_experts=8,
                                  renormalize=False,
                                  num_bits=4,
                                  group_size=32,
                                  hidden_dim=128,
                                  ep_size=1,
                                  ep_group=None,
                                  output_dtype=torch.bfloat16,
                                  num_max_dispatch_tokens_per_rank=128,
                                  layer_idx=0)
    impl = _build_fused_moe_w4a16(spec)
    assert isinstance(impl, CuteFusedMoEW4A16Impl)
    for bad, message in [(replace(spec, ep_size=8), 'EP process group'), (replace(spec,
                                                                                  group_size=128), 'group-size 32'),
                         (replace(spec, output_dtype=torch.float16), 'BF16')]:
        with pytest.raises(ValueError, match=message):
            _build_fused_moe_w4a16(bad)
    gp, gs, _ = weights(8, 128, 128)
    dp, ds, _ = weights(8, 128, 64)
    gs.zero_()
    x = torch.randn(2, 128, device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([[0, 7], [7, 0]], device='cuda')
    tw = torch.ones(2, 2, device='cuda')
    out = impl.forward(x, tw, ids, gp, gs, dp, ds)
    assert torch.count_nonzero(out) == 0
    with pytest.raises(ValueError, match='contiguous activation channels'):
        impl.forward(torch.empty(2, 256, device='cuda', dtype=torch.bfloat16)[:, ::2], tw, ids, gp, gs, dp, ds)


@pytest.mark.parametrize('m', [1, 17])
def test_ep_normal_invalid_routes_graph(m):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute import fused_moe_w4a16_cute
    torch.manual_seed(37)
    e, h, f, topk = 2, 128, 64, 4  # Global topk may exceed E_local.
    gp, gs, _ = weights(e, 2 * f, h)
    dp, ds, _ = weights(e, h, f)
    x = torch.randn(m, h, device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([0, -1, 1, e], device='cuda').expand(m, -1).clone()
    tw = torch.rand(m, topk, device='cuda')
    tw[:, 1] = float('nan')

    def run(fn=fused_moe_w4a16_cute):
        return fn(x, gp, gs, dp, ds, tw, ids, topk, allow_invalid_routes=True)

    assert nrmse(run(), run(fused_moe_w4a16)) < .012
    with pytest.raises(ValueError, match='renormalize'):
        fused_moe_w4a16_cute(x, gp, gs, dp, ds, tw, ids, topk, True, allow_invalid_routes=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for all_invalid in (True, False):
        ids.fill_(-1)
        if not all_invalid:
            ids[:, 0] = 1
        x.normal_()
        graph.replay()
        ref = run(fused_moe_w4a16)
        assert torch.isfinite(out).all()
        assert nrmse(out, ref) < .012


@pytest.mark.parametrize('capacity', [5, 32])
def test_ep_masked_graph(capacity):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_masked
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute import fused_moe_w4a16_cute_masked
    torch.manual_seed(38)
    e, h, f = 3, 256, 128
    gp, gs, _ = weights(e, 2 * f, h)
    dp, ds, _ = weights(e, h, f)
    x = torch.randn(e, capacity, h, device='cuda', dtype=torch.bfloat16)
    counts = torch.tensor([capacity, 2, 0], device='cuda', dtype=torch.int32)

    def run(fn=fused_moe_w4a16_cute_masked):
        return fn(x, gp, gs, dp, ds, counts)

    assert nrmse(run(), run(fused_moe_w4a16_masked)) < .012
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for values in ([0, 0, 0], [1, capacity, 2], [capacity + 3, -1, capacity]):
        counts.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
        x.normal_()
        for expert, count in enumerate(values):
            x[expert, max(0, min(count, capacity)):] = float('nan')
        graph.replay()
        assert torch.isfinite(out).all()
        assert nrmse(out, run(fused_moe_w4a16_masked)) < .012


@pytest.mark.parametrize('bm,bk,split', [(8, 64, 4), (8, 128, 8), (8, 256, 8), (16, 128, 4), (32, 128, 4),
                                         (64, 128, 4)])
@pytest.mark.parametrize('fast', [False, True])
def test_split_k(bm, bk, split, fast):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute import _launch_gemm, gather_routed
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_blocks
    torch.manual_seed(24)
    e, m, n, k, topk = 4, 37, 128, 7168, 2
    packed, s, _ = weights(e, n, k)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    ids = torch.rand(m, e, device='cuda').topk(topk, -1).indices
    meta = _get_sorted_idx_blocks(ids, e, e, 0, bm)
    padded = gather_routed(x, meta, topk, bm)
    ref = x.new_empty((padded.shape[0], n))
    out = torch.empty_like(ref)
    _launch_gemm(padded, packed, s, ref, meta, False, bm, bk, 2, 1)
    _launch_gemm(padded, packed, s, out, meta, False, bm, bk, 2, split, fast)
    count = int(meta[3][-1]) * bm
    assert nrmse(out[:count], ref[:count]) < .001


@pytest.mark.parametrize('duplicate', [False, True])
def test_single_token_split_activation_graph(duplicate):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute import fused_moe_w4a16_cute
    torch.manual_seed(124)
    gp, gs, _ = weights(4, 128, 1024)
    dp, ds, _ = weights(4, 1024, 64)
    x = torch.randn(1, 1024, device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([[1, 1] if duplicate else [3, 1]], device='cuda', dtype=torch.int32)
    tw = torch.tensor([[.3, .7]], device='cuda')

    def run(fn):
        return fn(x, gp, gs, dp, ds, tw, ids, 2)

    assert nrmse(run(fused_moe_w4a16_cute), run(fused_moe_w4a16)) < .012
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(fused_moe_w4a16_cute)
    ids.copy_((ids + 1) % 4)
    x.normal_()
    graph.replay()
    assert nrmse(output, run(fused_moe_w4a16)) < .012


@pytest.mark.parametrize('k', [96, 128, 256])
def test_fp32_scales(k):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute import _launch_gemm, gather_routed
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_blocks
    torch.manual_seed(9)
    packed, _, _ = weights(2, 96, k)
    s = torch.rand(2, 96, k // 32, device='cuda') * .03
    shifts = torch.arange(8, device='cuda', dtype=torch.int32) * 4
    q = (((packed[..., None] >> shifts) & 15) - 8).flatten(-2)
    w = (q.float() * s.repeat_interleave(32, -1)).bfloat16()
    x = torch.randn(3, k, device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([[0], [1], [0]], device='cuda')
    meta = _get_sorted_idx_blocks(ids, 2, 2, 0, 8)
    padded = gather_routed(x, meta, 1)
    out = x.new_empty((3, 96))
    _launch_gemm(padded, packed, s, out, meta, fast_dequant=True)
    ref = torch.stack([x[r].float() @ w[expert].float().T for r, expert in enumerate(ids.flatten())])
    assert nrmse(out, ref) < .003
