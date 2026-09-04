# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

import lmdeploy.pytorch.nn.linear.blocked_fp8 as blocked_fp8
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.nn.linear.blocked_fp8 import QKVBlockedF8Linear


class _ReferenceBlockedF8Impl:

    def set_scale_fmt(self, scale_fmt):
        del scale_fmt

    def forward(self, x, weight, weight_scale_inv, *args, **kwargs):
        del args, kwargs
        row_scales = weight_scale_inv.repeat_interleave(128, dim=0)
        col_scales = row_scales.repeat_interleave(128, dim=1)
        dequantized = weight.float() * col_scales[:weight.size(0), :weight.size(1)]
        return (x.float() @ dequantized.T).to(x.dtype)


def _build_qkv(monkeypatch, *, tp, rank, num_kv_heads, num_replicas,
               checkpoint_sections, continuous):

    def _init_tp_args(self, *args, **kwargs):
        del args, kwargs
        self.is_tp = True
        self.all_reduce = False
        self.tp = tp
        self.tp_rank = rank
        self.tp_mode = TPMode.DEFAULT
        self.tp_group = None
        self.gather_group = None
        self._tp_args_initialized = True

    backend = SimpleNamespace(
        get_layer_impl_builder=lambda op: SimpleNamespace(
            build=lambda *args, **kwargs: _ReferenceBlockedF8Impl()))
    monkeypatch.setattr(QKVBlockedF8Linear, 'init_tp_args', _init_tp_args)
    monkeypatch.setattr(blocked_fp8, 'get_backend', lambda: backend)
    return QKVBlockedF8Linear(
        in_features=128,
        num_q_heads=64,
        num_kv_heads=num_kv_heads,
        head_size=192,
        head_size_v=128,
        dtype=torch.bfloat16,
        device=torch.device('cpu'),
        num_replicate_kv_heads=num_replicas,
        checkpoint_output_shard_sizes=checkpoint_sections,
        continuous_qkv_scale_layout=continuous,
    )


def _source_weight(rows, offset):
    values = (torch.arange(rows, dtype=torch.float32) + offset).remainder(13) - 6
    return values[:, None].expand(rows, 128).to(torch.float8_e4m3fn)


def _load_qkv(linear, weights, scales):
    for shard_id in ('q', 'k', 'v'):
        linear.weight.weight_loader(linear.weight, weights[shard_id], shard_id)
        linear.weight_scale_inv.weight_loader(
            linear.weight_scale_inv, scales[shard_id], shard_id)


def _project(x, weight, row_scales):
    dequantized = weight.float() * row_scales[:, None]
    return (x.float() @ dequantized.T).to(x.dtype)


@pytest.mark.parametrize(
    ('tp', 'rank', 'num_kv_heads', 'num_replicas'),
    [(4, 3, 4, 1), (8, 7, 8, 2)],
)
def test_full_qkv_loader_matches_checkpoint_quantization(monkeypatch, tp, rank,
                                                         num_kv_heads,
                                                         num_replicas):
    """Full QKV preserves its TP4 K/V boundary under TP4 and TP8."""
    linear = _build_qkv(
        monkeypatch,
        tp=tp,
        rank=rank,
        num_kv_heads=num_kv_heads,
        num_replicas=num_replicas,
        checkpoint_sections=(3072, 192, 128),
        continuous=True,
    )
    weights = {
        'q': _source_weight(12288, 0),
        'k': _source_weight(768, 3),
        'v': _source_weight(512, 7),
    }
    scales = {
        'q': (1 + torch.arange(96, dtype=torch.float32) / 32)[:, None],
        'k': (5 + torch.arange(8, dtype=torch.float32) / 16)[:, None],
        'v': (9 + torch.arange(4, dtype=torch.float32) / 8)[:, None],
    }
    _load_qkv(linear, weights, scales)

    x = torch.linspace(-1, 1, 128, dtype=torch.bfloat16).view(1, 128)
    query, key, value = linear.split_qkv(linear(x))

    q_rows = 12288 // tp
    q_start = rank * q_rows
    kv_rank = rank // num_replicas
    k_start = kv_rank * 192
    v_start = kv_rank * 128
    q_scale = scales['q'].flatten().repeat_interleave(128)[q_start:q_start + q_rows]
    k_scale = scales['k'][kv_rank * 2:kv_rank * 2 + 2].flatten().repeat_interleave(128)[:192]
    # The checkpoint block crossing K/V uses K's final scale for V[0:64].
    v_scale = torch.cat((
        scales['k'][kv_rank * 2 + 1].expand(64),
        scales['v'][kv_rank].expand(64),
    ))

    expected_q = _project(x, weights['q'][q_start:q_start + q_rows], q_scale)
    expected_k = _project(x, weights['k'][k_start:k_start + 192], k_scale)
    expected_v = _project(x, weights['v'][v_start:v_start + 128], v_scale)
    torch.testing.assert_close(query.flatten(1), expected_q)
    torch.testing.assert_close(key.flatten(1), expected_k)
    torch.testing.assert_close(value.flatten(1), expected_v)


def test_swa_qkv_loader_matches_unaligned_tp8_shard(monkeypatch):
    """SWA TP8 crops an unaligned K shard without changing its scale grid."""
    linear = _build_qkv(
        monkeypatch,
        tp=8,
        rank=7,
        num_kv_heads=8,
        num_replicas=1,
        checkpoint_sections=(3072, 384, 256),
        continuous=False,
    )
    weights = {
        'q': _source_weight(12288, 0),
        'k': _source_weight(1536, 3),
        'v': _source_weight(1024, 7),
    }
    scales = {
        'q': (1 + torch.arange(96, dtype=torch.float32) / 32)[:, None],
        'k': (5 + torch.arange(12, dtype=torch.float32) / 16)[:, None],
        'v': (9 + torch.arange(8, dtype=torch.float32) / 8)[:, None],
    }
    _load_qkv(linear, weights, scales)

    x = torch.linspace(-1, 1, 128, dtype=torch.bfloat16).view(1, 128)
    query, key, value = linear.split_qkv(linear(x))
    q_start, k_start, v_start = 7 * 1536, 7 * 192, 7 * 128
    q_scale = scales['q'].flatten().repeat_interleave(128)[q_start:q_start + 1536]
    k_scale = scales['k'].flatten().repeat_interleave(128)[k_start:k_start + 192]
    v_scale = scales['v'].flatten().repeat_interleave(128)[v_start:v_start + 128]

    expected_q = _project(x, weights['q'][q_start:q_start + 1536], q_scale)
    expected_k = _project(x, weights['k'][k_start:k_start + 192], k_scale)
    expected_v = _project(x, weights['v'][v_start:v_start + 128], v_scale)
    torch.testing.assert_close(query.flatten(1), expected_q)
    torch.testing.assert_close(key.flatten(1), expected_k)
    torch.testing.assert_close(value.flatten(1), expected_v)
