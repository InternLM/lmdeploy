# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""TP regression for the merged TM-quant projections.

Each fused section must be partitioned exactly once: the merged classes
distribute per-section sizes themselves, so the base allocation must not
divide the (already local) sum again. Runs on CPU with a mocked rank
context and a stubbed native backend.
"""
from types import SimpleNamespace

import torch

import lmdeploy.pytorch.nn.linear.base as linear_base
import lmdeploy.pytorch.nn.linear.tm_quant as tm_quant_module
import lmdeploy.pytorch.nn.linear.utils as linear_utils
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.nn.linear.tm_quant import MergedTmQuantLinear, QKVTmQuantLinear


class _StubImpl:

    def __init__(self, spec):
        self.spec = spec

    def update_weights(self, *args, **kwargs):
        pass


class _FakeBackend:

    def build_op(self, spec, enable_deterministic=False):
        return _StubImpl(spec)


class _FakeDistManager:

    def current_config(self):
        return SimpleNamespace(get_tp_by_layer=lambda layer_type: (TP_ENV['tp'], TPMode.DEFAULT))

    def current_context(self):
        return SimpleNamespace(dist_config=SimpleNamespace(mlp_tp=TP_ENV['tp'], attn_tp=TP_ENV['tp']))


class _FakeGroup:
    gpu_group = None
    gpu_gather_group = None


TP_ENV = {'tp': 1, 'rank': 0}


def _patch_rank_ctx(monkeypatch, tp: int, rank: int):
    TP_ENV['tp'] = tp
    TP_ENV['rank'] = rank

    def fake_tp_world_rank(layer_type=None):
        return tp, rank

    def fake_dist_group(layer_type='attn'):
        return _FakeGroup()

    for mod in (linear_base, linear_utils):
        monkeypatch.setattr(mod, 'get_tp_world_rank', fake_tp_world_rank)
    monkeypatch.setattr(linear_base, 'get_dist_manager', _FakeDistManager)
    monkeypatch.setattr(linear_base, 'get_dist_group', fake_dist_group)

    def fake_build_impl(self, device):
        return _StubImpl(SimpleNamespace(in_features=self.in_features, out_features=self.out_features))

    monkeypatch.setattr(tm_quant_module.TmQuantLinear, '_build_impl', fake_build_impl)


def _load_shard(module, bits, scale, shard_id):
    """Load one checkpoint projection through the module's loaders."""
    module.weight_loader(module.weight_packed, bits, shard_id)
    module.weight_loader(module.weight_scale, scale, shard_id)


class TestMergedTpPartitionedOnce:

    def test_gate_up_tp2_rank0(self, monkeypatch):
        """Two 256-output checkpoint projections at TP=2 rank 0: local sections
        [128, 128]; the fused allocation must keep 256 rows and loading both
        shards through param.data.split must succeed."""
        _patch_rank_ctx(monkeypatch, tp=2, rank=0)
        module = MergedTmQuantLinear(256, [256, 256], lane='nvfp4', bias=False, device=torch.device('cpu'))
        assert module.all_out_features == [128, 128]
        assert module.out_features == 256
        assert module.weight_packed.shape == (256, 128)

        bits0, scale0, _ = _quantize(torch.randn(256, 256) * 0.05)
        bits1, scale1, _ = _quantize(torch.randn(256, 256) * 0.05)
        _load_shard(module, bits0, scale0, 0)
        _load_shard(module, bits1, scale1, 1)
        assert module.weight_packed.shape == (256, 128)

    def test_qkv_tp2_rank0(self, monkeypatch):
        """QKV fused at TP=2 rank 0: head widths are already local; loading
        separate checkpoint q/k/v projections must succeed."""
        _patch_rank_ctx(monkeypatch, tp=2, rank=0)
        module = QKVTmQuantLinear(256, 4, 2, 32, 32, lane='nvfp4', bias=False, device=torch.device('cpu'))
        sections = module.get_qkv_out_feautures()
        assert sections == (64, 32, 32)
        assert list(module.all_out_features) == [64, 32, 32]
        assert module.out_features == 128
        assert module.weight_packed.shape == (128, 128)

        for shard_id, n in zip(('q', 'k', 'v'), (128, 64, 64)):
            bits, scale, _ = _quantize(torch.randn(n, 256) * 0.05)
            _load_shard(module, bits, scale, shard_id)
        assert module.weight_packed.shape == (128, 128)


def _quantize(w_dense: torch.Tensor):
    """Minimal NVFP4 checkpoint bundle: codes packed two-per-byte, e4m3
    block scales, per-tensor global scale."""
    n, k = w_dense.shape
    groups = w_dense.view(n, k // 16, 16)
    amax_w = w_dense.abs().amax().clamp_min(1e-8)
    scale = (groups.abs().amax(dim=-1, keepdim=True) * (448.0 / amax_w)).clamp_max(448.0)
    scale = scale.to(torch.float8_e4m3fn).float()
    scaled = groups * ((2688.0 / amax_w) / scale)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mag = torch.argmin((scaled.unsqueeze(-1) - grid).abs(), dim=-1)
    sign = (scaled < 0).to(torch.uint8)
    codes = ((mag.to(torch.uint8) & 0x7) | (sign << 3)).view(n, k)
    bits = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return bits, scale.view(n, k // 16).to(torch.float8_e4m3fn), (2688.0 / amax_w).float()
