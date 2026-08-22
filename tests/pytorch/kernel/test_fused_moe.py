import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize(('num_routes', 'block_m'), [(640, 16), (512 * 24, 32), (512 * 40, 64)])
def test_origin_blocked_fp8_small_m_configs_use_average_routes(num_routes, block_m):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=64,
                                                               num_routes=num_routes,
                                                               num_experts=512,
                                                               local_experts=512)
    assert gate_config == dict(block_m=max(64, block_m), block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=block_m, block_n=128, num_warps=4, num_stages=3)


def test_origin_blocked_fp8_large_m_uses_bm64_down_config():
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=65,
                                                               num_routes=650,
                                                               num_experts=512,
                                                               local_experts=512)
    assert gate_config == dict(block_m=128, block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=64, block_n=128, num_warps=4, num_stages=3)


def test_origin_blocked_fp8_large_m_high_avg_routes_uses_default_down_config():
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=2048,
                                                               num_routes=512 * 40,
                                                               num_experts=512,
                                                               local_experts=512)
    expected = dict(block_m=128, block_n=128, num_warps=4, num_stages=3)
    assert gate_config == expected
    assert down_config == expected


def test_origin_blocked_fp8_uses_average_routes_for_256_experts():
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=64,
                                                               num_routes=256 * 2,
                                                               num_experts=256,
                                                               local_experts=256)
    assert gate_config == dict(block_m=64, block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=16, block_n=128, num_warps=4, num_stages=3)


def test_origin_blocked_fp8_large_m_uses_average_routes_for_256_experts():
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=512,
                                                               num_routes=256 * 16,
                                                               num_experts=256,
                                                               local_experts=256)
    assert gate_config == dict(block_m=128, block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=64, block_n=128, num_warps=4, num_stages=3)


@pytest.mark.parametrize(('num_routes', 'block_m'), [(640, 64), (512 * 40, 64), (512 * 64, 128),
                                                     (512 * 160, 128)])
def test_compact_blocked_fp8_configs_use_average_routes(num_routes, block_m):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _compact_blocked_fp8_moe_config

    assert _compact_blocked_fp8_moe_config(num_routes, num_experts=512) == dict(block_m=block_m,
                                                                                block_n=128,
                                                                                num_warps=4,
                                                                                num_stages=3)


@pytest.mark.parametrize(('num_routes', 'gate_out_features', 'block_m', 'block_n'), [
    (256 * 3, 512, 16, 128),
    (256 * 4, 512, 16, 64),
    (256 * 8, 512, 16, 64),
    (256 * 16, 512, 32, 128),
    (256 * 32, 512, 64, 128),
    (256 * 4, 4096, 64, 128),
])
def test_compact_blocked_fp8_both_configs(num_routes, gate_out_features, block_m, block_n):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _compact_blocked_fp8_moe_both_config

    assert _compact_blocked_fp8_moe_both_config(num_routes, 256, gate_out_features) == dict(
        block_m=block_m, block_n=block_n, num_warps=4, num_stages=3)


@pytest.mark.parametrize(('num_tokens', 'num_routes', 'num_experts', 'local_experts', 'gate_features',
                          'input_features', 'expected_tile'), [
    (32, 256, 256, 256, 512, 6144, None),
    (64, 256 * 2, 256, 256, 512, 6144, (16, 64)),
    (80, 256 * 3, 256, 256, 512, 6144, (16, 128)),
    (1536, 256 * 48, 256, 256, 512, 6144, (64, 128)),
    (2048, 256 * 64, 256, 256, 512, 6144, None),
    (64, 256 * 2, 256, 256, 512, 2048, None),
    (65, 65 * 8, 256, 256, 512, 2048, (64, 128)),
    (1280, 256 * 40, 256, 256, 512, 2048, (64, 128)),
    (2048, 256 * 64, 256, 256, 512, 2048, (64, 128)),
    (2080, 256 * 65, 256, 256, 512, 2048, None),
    (2048, 256 * 64, 256, 256, 1024, 2048, (64, 128)),
    (512, 256 * 16, 256, 256, 256, 2048, None),
    (144, 384 * 3, 384, 384, 1024, 2048, (64, 128)),
    (64, 512, 512, 512, 512, 4096, (16, 64)),
    (3840, 512 * 60, 512, 512, 512, 4096, (64, 128)),
    (4480, 512 * 70, 512, 512, 512, 4096, None),
    (7680, 512 * 120, 512, 512, 512, 4096, None),
    (8960, 512 * 140, 512, 512, 512, 4096, (64, 128)),
    (10240, 512 * 160, 512, 512, 512, 4096, (64, 128)),
    (10304, 512 * 161, 512, 512, 512, 4096, None),
    (48, 128 * 3, 128, 128, 1024, 2048, None),
    (96, 256 * 3, 512, 256, 1024, 2048, None),
])
def test_compact_blocked_fp8_both_strategy_uses_launch_features(num_tokens, num_routes, num_experts, local_experts,
                                                               gate_features, input_features, expected_tile):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import (
        _select_compact_blocked_fp8_moe_both_config,
    )

    expected = None
    if expected_tile is not None:
        expected = dict(block_m=expected_tile[0], block_n=expected_tile[1], num_warps=4, num_stages=3)
    assert _select_compact_blocked_fp8_moe_both_config(
        num_tokens, num_routes, num_experts, local_experts, gate_features, input_features) == expected


@pytest.mark.parametrize(('num_tokens', 'num_routes', 'origin_ctas', 'compact_ctas'), [
    (65, 650, 512 * 2 * 32, 512 * 1 * 32),
    (1024, 512 * 20, 512 * 16 * 32, 512 * 1 * 32),
    (4096, 512 * 80, 512 * 32 * 32, 512 * 1 * 32),
    (8192, 512 * 160, 512 * 64 * 32, 512 * 2 * 32),
])
def test_blocked_fp8_moe_cta_estimates(num_tokens, num_routes, origin_ctas, compact_ctas):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import _blocked_fp8_moe_cta_estimates

    assert _blocked_fp8_moe_cta_estimates(num_tokens,
                                          num_routes,
                                          num_experts=512,
                                          local_experts=512,
                                          out_features=4096) == (origin_ctas, compact_ctas)


@pytest.mark.parametrize(('num_tokens', 'num_routes', 'num_experts', 'local_experts', 'out_features', 'expected'), [
    (64, 640, 512, 512, 4096, False),
    (511, 512 * 10, 512, 512, 4096, False),
    (512, 512 * 10, 512, 512, 4096, True),
    (1024, 512 * 20, 512, 512, 4096, True),
    (1024, 512 * 20, 512, 128, 4096, False),
    (4096, 512 * 80, 512, 256, 128, True),
    (2048, 256 * 64, 256, 256, 2048, False),
    (4096, 256 * 128, 256, 256, 2048, True),
    (512, 256 * 16, 256, 256, 7168, False),
    (1024, 256 * 32, 256, 256, 7168, True),
])
def test_compact_blocked_fp8_down_policy_is_prefill_and_cta_gated(num_tokens, num_routes, num_experts, local_experts,
                                                                  out_features, expected):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import (
        _should_use_compact_blocked_fp8_moe_down_by_shape,
    )

    assert _should_use_compact_blocked_fp8_moe_down_by_shape(num_tokens,
                                                             num_routes,
                                                             num_experts=num_experts,
                                                             local_experts=local_experts,
                                                             out_features=out_features) is expected

def test_compact_moe_dispatch_prefers_many_local_experts(monkeypatch):
    """Large local expert counts should select compact routed-block
    scheduling."""
    import importlib

    fused_moe_module = importlib.import_module('lmdeploy.pytorch.kernels.cuda.moe.fused_moe')
    monkeypatch.setattr(fused_moe_module, '_supports_compact_moe', lambda *args: True)

    hidden_states = torch.empty(1, 4)
    w1 = torch.empty(1024, 8, 4)
    w2 = torch.empty(1024, 4, 4)
    topk_ids = torch.zeros(1, 1, dtype=torch.long)

    assert fused_moe_module._should_use_compact_moe(hidden_states, w1, w2, topk_ids, num_experts=1024)


def test_compact_moe_dispatch_keeps_dense_route_fallback(monkeypatch):
    """Keep the existing compact path for dense routing on smaller expert
    counts."""
    import importlib

    fused_moe_module = importlib.import_module('lmdeploy.pytorch.kernels.cuda.moe.fused_moe')
    monkeypatch.setattr(fused_moe_module, '_supports_compact_moe', lambda *args: True)

    hidden_states = torch.empty(1, 4)
    w1 = torch.empty(64, 8, 4)
    w2 = torch.empty(64, 4, 4)

    sparse_topk_ids = torch.zeros(128, 1, dtype=torch.long)
    dense_topk_ids = torch.zeros(2048, 1, dtype=torch.long)

    assert not fused_moe_module._should_use_compact_moe(hidden_states, w1, w2, sparse_topk_ids, num_experts=64)
    assert fused_moe_module._should_use_compact_moe(hidden_states, w1, w2, dense_topk_ids, num_experts=64)


def test_compact_blocked_fp8_down_dispatch_prefers_wasteful_large_experts(monkeypatch):
    """Large local expert counts should select compact down scheduling when
    enough CTAs are saved."""
    import importlib

    blocked_fp8_module = importlib.import_module('lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8')
    monkeypatch.setattr(blocked_fp8_module, '_supports_compact_blocked_fp8_moe', lambda *args: True)

    input_quant = torch.empty(512, 4, dtype=torch.float8_e4m3fn)
    input_scale = torch.empty(512, 1)
    w1 = torch.empty(1024, 8, 4, dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty(1024, 1, 1)
    w2 = torch.empty(1024, 4096, 4, dtype=torch.float8_e4m3fn)
    w2_scale = torch.empty(1024, 32, 1)
    topk_ids = torch.zeros(512, 4, dtype=torch.long)

    assert blocked_fp8_module._should_use_compact_blocked_fp8_moe_down(
        input_quant, input_scale, w1, w1_scale, w2, w2_scale, topk_ids, num_experts=1024)


def test_compact_blocked_fp8_down_dispatch_rejects_small_local_experts(monkeypatch):
    """Blocked-FP8 compact down scheduling is disabled for small local expert
    counts."""
    import importlib

    blocked_fp8_module = importlib.import_module('lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8')
    monkeypatch.setattr(blocked_fp8_module, '_supports_compact_blocked_fp8_moe', lambda *args: True)

    input_quant = torch.empty(2048, 4, dtype=torch.float8_e4m3fn)
    input_scale = torch.empty(2048, 1)
    w1 = torch.empty(64, 8, 4, dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty(64, 1, 1)
    w2 = torch.empty(64, 4096, 4, dtype=torch.float8_e4m3fn)
    w2_scale = torch.empty(64, 32, 1)
    topk_ids = torch.zeros(2048, 1, dtype=torch.long)

    assert not blocked_fp8_module._should_use_compact_blocked_fp8_moe_down(
        input_quant, input_scale, w1, w1_scale, w2, w2_scale, topk_ids, num_experts=64)


@pytest.mark.parametrize(('num_routes', 'num_experts', 'expected'), [
    (2048, 2048, True),
    (2049, 2048, False),
    (4096, 256, False),
    (2048, 2049, False),
])
def test_single_cta_sorted_idx_policy(num_routes, num_experts, expected):
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _should_use_single_cta_sorted_idx

    assert _should_use_single_cta_sorted_idx(num_routes, num_experts) is expected


@pytest.mark.parametrize(('num_routes', 'num_experts', 'expected'), [
    (2048, 2048, True),
    (2049, 2048, False),
    (2048, 2049, False),
])
def test_single_cta_sorted_idx_blocks_policy(num_routes, num_experts, expected):
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _should_use_single_cta_sorted_idx_blocks

    assert _should_use_single_cta_sorted_idx_blocks(num_routes, num_experts) is expected


@pytest.mark.parametrize(('block_r', 'expected'), [
    (2048, 8),
    (4096, 16),
    (8192, 32),
])
def test_single_cta_route_prepare_num_warps(block_r, expected):
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _single_cta_route_prepare_num_warps

    assert _single_cta_route_prepare_num_warps(block_r) == expected


def test_sorted_idx_dispatch_uses_route_prepare_policy(monkeypatch):
    import importlib

    moe_module = importlib.import_module('lmdeploy.pytorch.kernels.cuda.moe.fused_moe')
    topk_ids = torch.zeros((128, 8), dtype=torch.int64)
    single_cta = object()
    parallel = object()
    monkeypatch.setattr(moe_module, '_get_sorted_idx_single_cta', lambda *_: single_cta)
    monkeypatch.setattr(moe_module, '_get_sorted_idx_triton', lambda *_: parallel)
    monkeypatch.setattr(moe_module, '_supports_single_cta_route_prepare', lambda *_: True)

    assert moe_module._get_sorted_idx(topk_ids, 512) is single_cta
    large_topk_ids = torch.zeros((1, 2049), dtype=torch.int64)
    assert moe_module._get_sorted_idx(large_topk_ids, 512) is parallel
    monkeypatch.setattr(moe_module, '_supports_single_cta_route_prepare', lambda *_: False)
    assert moe_module._get_sorted_idx(topk_ids, 512) is parallel


def test_sorted_idx_blocks_dispatch_requires_full_expert_range(monkeypatch):
    import importlib

    moe_module = importlib.import_module('lmdeploy.pytorch.kernels.cuda.moe.fused_moe')
    topk_ids = torch.zeros((128, 8), dtype=torch.int64)
    single_cta = object()
    parallel = object()
    monkeypatch.setattr(moe_module, '_get_sorted_idx_blocks_single_cta', lambda *_: single_cta)
    monkeypatch.setattr(moe_module, '_get_sorted_idx_blocks_parallel', lambda *_: parallel)
    monkeypatch.setattr(moe_module, '_supports_single_cta_route_prepare', lambda *_: True)

    assert moe_module._get_sorted_idx_blocks(topk_ids, 2048, 2048, 0, 8) is single_cta
    assert moe_module._get_sorted_idx_blocks(topk_ids, 2048, 256, 0, 8) is parallel
    assert moe_module._get_sorted_idx_blocks(topk_ids, 2048, 2048, 1, 8) is parallel


def _make_route_ids(num_tokens, topk, num_experts, routing):
    route = torch.arange(num_tokens * topk, device='cuda', dtype=torch.int64)
    if routing == 'balanced':
        return (route % num_experts).view(num_tokens, topk)
    if routing == 'hot':
        return torch.arange(topk, device='cuda', dtype=torch.int64)[None].expand(num_tokens, -1).clone()
    if routing == 'noncontiguous':
        topk_ids = route.view(topk, num_tokens).T % num_experts
        assert not topk_ids.is_contiguous()
        return topk_ids
    logits = torch.rand((num_tokens, num_experts), device='cuda')
    return logits.topk(topk, dim=-1).indices


def _assert_sorted_idx_metadata(topk_ids, num_experts, metadata):
    sorted_idx, exp_start, exp_end = metadata[:3]
    assert sorted_idx.dtype == torch.int32
    assert exp_start.dtype == torch.int32
    assert exp_end.dtype == torch.int32

    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).to(torch.int32)
    expected_exp_end = counts.cumsum(0, dtype=torch.int32)
    torch.testing.assert_close(exp_start, expected_exp_end - counts)
    torch.testing.assert_close(exp_end, expected_exp_end)
    routes = torch.arange(topk_ids.numel(), device='cuda', dtype=torch.int32)
    torch.testing.assert_close(torch.sort(sorted_idx).values, routes)
    expected_experts = torch.repeat_interleave(torch.arange(num_experts, device='cuda'), counts.to(torch.int64))
    torch.testing.assert_close(topk_ids.flatten()[sorted_idx.to(torch.int64)], expected_experts)


def _assert_sorted_idx_block_metadata(topk_ids,
                                      num_experts,
                                      block_m,
                                      metadata,
                                      local_num_experts=None,
                                      expert_offset=0):
    if local_num_experts is None:
        local_num_experts = num_experts
    _assert_sorted_idx_metadata(topk_ids, num_experts, metadata)
    sorted_idx, exp_start, _, block_end, block_expert_ids, block_offsets = metadata
    assert block_end.dtype == torch.int32
    assert block_expert_ids.dtype == torch.int32
    assert block_offsets.dtype == torch.int32

    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).to(torch.int32)
    local_counts = counts[expert_offset:expert_offset + local_num_experts]
    block_counts = (local_counts + block_m - 1) // block_m
    torch.testing.assert_close(block_end, block_counts.cumsum(0, dtype=torch.int32))
    num_blocks = int(block_end[-1])
    expected_block_experts = torch.repeat_interleave(
        torch.arange(local_num_experts, device='cuda', dtype=torch.int32), block_counts.to(torch.int64))
    block_start = block_end - block_counts
    block_rank = torch.arange(num_blocks, device='cuda', dtype=torch.int32)
    block_rank -= torch.repeat_interleave(block_start, block_counts.to(torch.int64))
    local_exp_start = exp_start[expert_offset:expert_offset + local_num_experts]
    expected_block_offsets = torch.repeat_interleave(local_exp_start,
                                                     block_counts.to(torch.int64)) + block_rank * block_m
    torch.testing.assert_close(block_expert_ids[:num_blocks], expected_block_experts)
    torch.testing.assert_close(block_offsets[:num_blocks], expected_block_offsets)


@pytest.mark.parametrize(('num_tokens', 'num_experts', 'routing'), [
    (96, 256, 'balanced'),
    (96, 256, 'random'),
    (96, 256, 'hot'),
    (96, 256, 'noncontiguous'),
    (96, 2048, 'balanced'),
    (96, 2048, 'random'),
    (511, 257, 'random'),
])
def test_single_cta_sorted_idx(num_tokens, num_experts, routing):
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_single_cta

    torch.manual_seed(13)
    topk_ids = _make_route_ids(num_tokens, 8, num_experts, routing)
    metadata = _get_sorted_idx_single_cta(topk_ids, num_experts)
    _assert_sorted_idx_metadata(topk_ids, num_experts, metadata)


@pytest.mark.parametrize(('num_tokens', 'num_experts', 'routing'), [
    (96, 256, 'balanced'),
    (96, 256, 'random'),
    (96, 256, 'hot'),
    (96, 256, 'noncontiguous'),
    (96, 2048, 'random'),
    (1023, 257, 'random'),
])
def test_single_cta_sorted_idx_blocks(num_tokens, num_experts, routing):
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_blocks_single_cta

    torch.manual_seed(13)
    block_m = 16
    topk_ids = _make_route_ids(num_tokens, 8, num_experts, routing)
    metadata = _get_sorted_idx_blocks_single_cta(topk_ids, num_experts, block_m)
    _assert_sorted_idx_block_metadata(topk_ids, num_experts, block_m, metadata)
    block_expert_ids, block_offsets = metadata[-2:]
    expected_capacity = min(topk_ids.numel(), (topk_ids.numel() + block_m - 1) // block_m + num_experts)
    assert block_expert_ids.numel() == expected_capacity
    assert block_offsets.numel() == expected_capacity


@pytest.mark.parametrize(('num_tokens', 'num_experts'), [(1023, 257), (1024, 2048)])
def test_parallel_sorted_idx(num_tokens, num_experts):
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_triton

    torch.manual_seed(13)
    topk_ids = _make_route_ids(num_tokens, 8, num_experts, 'random')
    metadata = _get_sorted_idx_triton(topk_ids, num_experts)
    _assert_sorted_idx_metadata(topk_ids, num_experts, metadata)


@pytest.mark.parametrize(('num_experts', 'local_num_experts', 'expert_offset'), [
    (257, 257, 0),
    (2048, 257, 511),
])
def test_parallel_sorted_idx_blocks(num_experts, local_num_experts, expert_offset):
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_blocks_parallel

    torch.manual_seed(13)
    block_m = 64
    topk_ids = _make_route_ids(1023, 8, num_experts, 'random')
    metadata = _get_sorted_idx_blocks_parallel(topk_ids, num_experts, local_num_experts, expert_offset, block_m)
    _assert_sorted_idx_block_metadata(topk_ids, num_experts, block_m, metadata, local_num_experts, expert_offset)
    expected_capacity = (topk_ids.numel() + block_m - 1) // block_m + local_num_experts
    assert metadata[-2].numel() == expected_capacity
    assert metadata[-1].numel() == expected_capacity


def test_single_cta_sorted_idx_rejects_too_many_experts():
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import (
        _get_sorted_idx_blocks_single_cta,
        _get_sorted_idx_single_cta,
    )

    topk_ids = torch.zeros((1, 1), device='meta', dtype=torch.int64)
    with pytest.raises(ValueError, match='supports at most 2048 experts'):
        _get_sorted_idx_single_cta(topk_ids, 2049)
    with pytest.raises(ValueError, match='supports at most 2048 experts'):
        _get_sorted_idx_blocks_single_cta(topk_ids, 2049, 8)


def _get_sorted_idx(topk_idx: torch.Tensor, num_experts: int):
    flatten_topk_idx = topk_idx.flatten()
    sorted_ids = flatten_topk_idx.argsort()
    exp_range = torch.arange(0, num_experts, device=topk_idx.device)
    exp_tok_cnt = (flatten_topk_idx[None, :] == exp_range[:, None]).sum(1)
    return sorted_ids, exp_tok_cnt


class TestFusedMoEKernelLauncher:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def N(self):
        yield 128

    @pytest.fixture
    def K(self):
        yield 64

    @pytest.fixture
    def M(self):
        yield 256

    @pytest.fixture
    def num_experts(self):
        yield 64

    @pytest.fixture
    def top_k(self):
        yield 6

    @pytest.fixture
    def A(self, M, K, device, dtype):
        ret = torch.rand(M, K, device=device, dtype=dtype)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def B(self, num_experts, N, K, device, dtype):
        ret = torch.rand(num_experts, N, K, device=device, dtype=dtype)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def bias(self, num_experts, N, device, dtype):
        yield torch.rand(num_experts, N, device=device, dtype=dtype) - 0.5

    @pytest.fixture
    def router_weights(self, M, num_experts, device, dtype):
        yield torch.rand(M, num_experts, device=device, dtype=dtype)

    @pytest.fixture
    def topk_weights(self, router_weights, top_k):
        yield router_weights.topk(top_k, dim=-1)

    @pytest.fixture
    def topk_idx(self, topk_weights):
        yield topk_weights[1]

    @pytest.fixture
    def sort_and_cnt(self, topk_idx, num_experts):
        yield _get_sorted_idx(topk_idx, num_experts)

    @pytest.fixture
    def sorted_idx(self, sort_and_cnt):
        yield sort_and_cnt[0]

    @pytest.fixture
    def exp_tok_cnt(self, sort_and_cnt):
        yield sort_and_cnt[1]

    @pytest.fixture
    def exp_end(self, exp_tok_cnt):
        yield exp_tok_cnt.cumsum(0)

    @pytest.fixture
    def exp_start(self, exp_end, exp_tok_cnt):
        yield exp_end - exp_tok_cnt

    @pytest.fixture
    def gt(self, A, B, bias, top_k, topk_idx):
        M = A.size(0)
        N = B.size(1)
        E = B.size(0)
        C = B.new_empty(M, top_k, N)
        for eid in range(E):
            EB = B[eid].t()
            Ebias = bias[eid]
            token_idx, k_idx = torch.where(topk_idx == eid)
            if len(token_idx) == 0:
                continue
            EC = A[token_idx] @ EB + Ebias
            C[token_idx, k_idx] = EC
        yield C.flatten(0, 1)

    @torch.inference_mode()
    def test_launcher(self, A, B, bias, sorted_idx, exp_start, exp_end, top_k, M, gt):
        from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import fused_moe_kernel_launcher
        N = B.size(1)
        C = B.new_empty(M * top_k, N)

        fused_moe_kernel_launcher(
            A,
            B,
            C,
            sorted_idx,
            exp_start,
            exp_end,
            bias=bias,
            top_k=top_k,
            num_tokens=M,
        )
        torch.testing.assert_close(C, gt, atol=1e-3, rtol=1e-3)


def _mlp_forward(hidden_states, gate_proj, up_proj, down_proj):
    gate = F.linear(hidden_states, gate_proj)
    up = F.linear(hidden_states, up_proj)
    return F.linear(F.silu(gate) * up, down_proj)


class TestFusedMoe:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def in_size(self):
        yield 128

    @pytest.fixture
    def seq_len(seq_len):
        yield 128

    @pytest.fixture
    def hidden_size(self):
        yield 256

    @pytest.fixture
    def out_size(self):
        yield 128

    @pytest.fixture
    def num_experts(self):
        yield 64

    @pytest.fixture
    def top_k(self):
        yield 6

    @pytest.fixture
    def renormalize(self):
        yield True

    @pytest.fixture
    def hidden_states(self, seq_len, in_size, dtype, device):
        ret = torch.rand(seq_len, in_size, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w1(self, num_experts, hidden_size, in_size, dtype, device):
        ret = torch.rand(num_experts, hidden_size, in_size, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w2(self, num_experts, out_size, hidden_size, dtype, device):
        ret = torch.rand(num_experts, out_size, hidden_size // 2, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def router_logits(self, seq_len, num_experts, dtype, device):
        yield torch.rand(seq_len, num_experts, dtype=dtype, device=device)

    @pytest.fixture
    def topk_logits(self, router_logits, top_k):
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        yield torch.topk(routing_weights, top_k, dim=-1)

    @pytest.fixture
    def topk_weights(self, topk_logits):
        yield topk_logits[0]

    @pytest.fixture
    def topk_idx(self, topk_logits):
        yield topk_logits[1]

    @pytest.fixture
    def gt(self, hidden_states, w1, w2, topk_weights, topk_idx, renormalize):
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        seq_len = hidden_states.size(0)
        out_size = w2.size(1)
        output = hidden_states.new_zeros(seq_len, out_size)
        num_experts = w1.size(0)
        for eid in range(num_experts):
            token_idx, k_idx = torch.where(topk_idx == eid)
            gate_proj, up_proj = w1[eid].chunk(2, dim=0)
            down_proj = w2[eid]
            tmp_out = _mlp_forward(hidden_states[token_idx], gate_proj, up_proj, down_proj)
            tmp_out = tmp_out * topk_weights[token_idx, k_idx, None]
            output.index_add_(0, token_idx, tmp_out.to(output.dtype))
        yield output

    @torch.inference_mode()
    def test_fused_moe(self, hidden_states, w1, w2, topk_weights, topk_idx, top_k, renormalize, gt):
        from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import fused_moe
        output = fused_moe(hidden_states, w1, w2, topk_weights, topk_idx, topk=top_k, renormalize=renormalize)
        torch.testing.assert_close(output, gt, atol=1e-3, rtol=1e-3)


class TestFusedMoeW8A8(TestFusedMoe):

    @pytest.fixture
    def quant_states(self, hidden_states):
        from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_token_quant_int8
        states_i8, states_scale = per_token_quant_int8(hidden_states, 1e-7)
        yield states_i8, states_scale

    def quant_weight(self, w):
        from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_channel_quant
        num_experts, num_outs, _ = w.shape
        w = w.flatten(0, -2)
        w_i8, w_scale = per_channel_quant(w, torch.int8)
        w_i8 = w_i8.view(num_experts, num_outs, -1)
        w_scale = w_scale.view(num_experts, num_outs, -1)
        return w_i8, w_scale

    @pytest.fixture
    def quant_w1(self, w1):
        w_i8, w_scale = self.quant_weight(w1)
        yield w_i8, w_scale

    @pytest.fixture
    def quant_w2(self, w2):
        w_i8, w_scale = self.quant_weight(w2)
        yield w_i8, w_scale

    @torch.inference_mode()
    def test_fused_moe(self, quant_states, quant_w1, quant_w2, topk_weights, topk_idx, top_k, renormalize, gt):
        from lmdeploy.pytorch.kernels.cuda.moe.w8a8 import fused_moe_w8a8
        state_i8, state_scale = quant_states
        w1_i8, w1_scale = quant_w1
        w2_i8, w2_scale = quant_w2

        output = fused_moe_w8a8(state_i8,
                                state_scale,
                                w1_i8,
                                w1_scale,
                                w2_i8,
                                w2_scale,
                                topk_weights=topk_weights,
                                topk_ids=topk_idx,
                                topk=top_k,
                                out_dtype=torch.float16,
                                renormalize=renormalize)
        torch.testing.assert_close(output, gt, atol=5e-3, rtol=1e-3)


class TestFusedMoeBlockedFP8Compact:

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA is required for blocked FP8 MoE kernels.')
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            pytest.skip('Compact blocked FP8 MoE requires sm90+.')
        yield torch.device('cuda')

    @pytest.fixture
    def hidden_states(self, device):
        torch.manual_seed(0)
        yield torch.randn(32, 128, device=device, dtype=torch.bfloat16) / 8

    @pytest.fixture
    def quant_states(self, hidden_states):
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
        yield quant_fp8(hidden_states, 128, dtype=torch.float8_e4m3fn)

    def quant_weight(self, weight):
        from lmdeploy.lite.quantization.weight.quant_utils import quant_blocked_fp8
        return quant_blocked_fp8(weight, torch.float8_e4m3fn, 128)

    @pytest.fixture
    def quant_w1(self, device):
        torch.manual_seed(1)
        w1 = torch.randn(1024, 256, 128, device=device, dtype=torch.bfloat16) / 8
        yield self.quant_weight(w1)

    @pytest.fixture
    def quant_w2(self, device):
        torch.manual_seed(2)
        w2 = torch.randn(1024, 128, 128, device=device, dtype=torch.bfloat16) / 8
        yield self.quant_weight(w2)

    @pytest.fixture
    def topk_idx(self, device):
        torch.manual_seed(3)
        yield torch.randint(0, 1024, (32, 4), device=device)

    @pytest.fixture
    def topk_weights(self, device):
        torch.manual_seed(4)
        weights = torch.rand(32, 4, device=device, dtype=torch.bfloat16)
        yield weights / weights.sum(dim=-1, keepdim=True)

    @torch.inference_mode()
    def test_compact_matches_regular(self, monkeypatch, quant_states, quant_w1, quant_w2, topk_weights, topk_idx):
        from lmdeploy.pytorch.kernels.cuda.moe import blocked_fp8 as moe_mod
        state_fp8, state_scale = quant_states
        w1_fp8, w1_scale = quant_w1
        w2_fp8, w2_scale = quant_w2

        monkeypatch.setattr(moe_mod, '_should_use_compact_blocked_fp8_moe_down', lambda *args: False)
        regular = moe_mod.fused_moe_blocked_fp8(state_fp8,
                                                state_scale,
                                                w1_fp8,
                                                w1_scale,
                                                w2_fp8,
                                                w2_scale,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_idx,
                                                topk=topk_idx.size(1),
                                                out_dtype=torch.bfloat16)

        monkeypatch.setattr(moe_mod, '_should_use_compact_blocked_fp8_moe_down', lambda *args: True)
        compact = moe_mod.fused_moe_blocked_fp8(state_fp8,
                                                state_scale,
                                                w1_fp8,
                                                w1_scale,
                                                w2_fp8,
                                                w2_scale,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_idx,
                                                topk=topk_idx.size(1),
                                                out_dtype=torch.bfloat16)

        torch.testing.assert_close(compact, regular, atol=3e-2, rtol=3e-2)
