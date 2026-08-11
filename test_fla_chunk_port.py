"""验证移植到本仓库的 chunk gated-delta-rule triton 前向 kernel。

三组检查：
  1. 数值等价于 FLA `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
     (同一 initial_state 下，o 与 final_state 的 allclose，bf16 容忍度)。
  2. chunk 边界状态正确性：把每个 `chunk_states[:, c]`（= 处理完前 c*64 个
     token 后的 recurrent state）与本仓库已有的 `fused_recurrent_gated_delta_rule`
     (tilelang 递推 kernel) 跑同长度前缀得到的 final_state 交叉验证；
     并验证 `chunk_states[:, 0] == initial_state`、`final_state == 跑全长递推`。
  3. varlen (cu_seqlens) 下 o/final_state 与 FLA 等价、chunk_states 形状正确。

注意 chunk 约定（来自 FLA inter-chunk kernel 的存数顺序：每个 chunk 循环顶部
先存当前 b_h 再做该 chunk 的更新）：
  - chunk_states[:, c] = 处理完 token 0..c*64 后的 state = 第 c 个 chunk
    *开始处* 的 state（c=0 即 initial_state）。
  - final_state = 处理完全部 token 后的 state（比 chunk_states[:, -1] 多一个 chunk）。
  - 故 block 步 s=c*64 (c>=1) 对应 chunk_states[:, c]；全长对应 final_state。
"""
import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule import chunk_gated_delta_rule
from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

BT = 64
DEVICE = 'cuda'


def _make_inputs(B, T, H, HV, K, V, dtype, with_init=True):
    """构造与 lmdeploy 调 kernel 时一致的输入：q/k 已 l2norm、beta 已 sigmoid、
    g 为 log-space per-token decay。"""
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype)
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    v = torch.randn(B, T, HV, V, device=DEVICE, dtype=dtype)
    # g: per-token log decay (negative), matches lmdeploy preprocess output.
    g = -torch.rand(B, T, HV, device=DEVICE, dtype=dtype).abs() * 0.1
    beta = torch.rand(B, T, HV, device=DEVICE, dtype=dtype).sigmoid()
    init = None
    if with_init:
        init = torch.zeros(B, HV, V, K, device=DEVICE, dtype=torch.float32)
    return q, k, v, g, beta, init


def _allclose(a, b, atol, rtol, name=''):
    max_abs = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().clamp_min(1e-6)
    max_rel = ((a.float() - b.float()).abs() / denom).max().item()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    assert ok, (f'{name}: not close (max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, '
                f'atol={atol}, rtol={rtol})')
    print(f'  {name:24s} max_abs={max_abs:.3e} max_rel={max_rel:.3e} OK')


# ---------------------------------------------------------------------------
# 1. 数值等价于 FLA
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('T', [256, 200])
def test_equivalence_with_fla(T):
    torch.manual_seed(0)
    B, H, HV, K, V = 1, 4, 4, 64, 64
    dtype = torch.bfloat16
    scale = K ** -0.5
    q, k, v, g, beta, init = _make_inputs(B, T, H, HV, K, V, dtype)

    o_local, fs_local, chunk_states = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, initial_state=init,
        output_final_state=True, state_v_first=True)
    # FLA reference: lmdeploy calls FLA with the deprecated `transpose_state_layout`
    # kwarg (not `state_v_first`); in this FLA build the two are *not* equivalent —
    # `transpose_state_layout=True` yields the pool/recurrent [V,K] layout that our
    # port reproduces, while `state_v_first=True` is inverted. So we compare against
    # `transpose_state_layout=True`, the convention lmdeploy actually relies on.
    o_fla, fs_fla = fla_chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, initial_state=init,
        output_final_state=True, use_qk_l2norm_in_kernel=False,
        transpose_state_layout=True)

    _allclose(o_local, o_fla, atol=2e-2, rtol=2e-2, name=f'o (T={T})')
    _allclose(fs_local, fs_fla, atol=2e-2, rtol=2e-2, name=f'final_state (T={T})')
    # 形状
    NT = (T + BT - 1) // BT
    assert chunk_states.shape == (B, NT, HV, V, K), chunk_states.shape
    assert torch.isfinite(chunk_states.float()).all()


# ---------------------------------------------------------------------------
# 2. chunk 边界状态 vs 递推 kernel 交叉验证
# ---------------------------------------------------------------------------
def test_chunk_states_vs_recurrent():
    torch.manual_seed(1)
    T = 256                       # NT=4, 整除 64
    B, H, HV, K, V = 1, 4, 4, 64, 64
    dtype = torch.float32         # 递推与 chunk 用 fp32 紧致比较
    scale = K ** -0.5
    q, k, v, g, beta, init = _make_inputs(B, T, H, HV, K, V, dtype)

    o_local, fs_local, chunk_states = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, initial_state=init,
        output_final_state=True, state_v_first=True)
    NT = chunk_states.shape[1]
    state_indices = torch.tensor([0], device=DEVICE, dtype=torch.int64)

    # chunk_states[:, 0] == initial_state (第 0 个 chunk 开始处的 state)
    _allclose(chunk_states[0, 0], init[0], atol=1e-5, rtol=1e-5, name='chunk_states[0,0]==init')

    # 每个 chunk_states[:, c] (c>=1) == 递推跑前缀 0..c*64 的 final_state
    for c in range(1, NT + 1):
        prefix = c * BT
        init_copy = init.clone()
        _, fs_rec = fused_recurrent_gated_delta_rule(
            q[:, :prefix], k[:, :prefix], v[:, :prefix],
            g=g[:, :prefix], beta=beta[:, :prefix], scale=scale,
            initial_state=init_copy, output_final_state=True,
            state_indices=state_indices, transpose_state_layout=True)
        if c < NT:
            _allclose(chunk_states[0, c], fs_rec[0], atol=2e-2, rtol=2e-2,
                      name=f'chunk_states[0,{c}] vs recurrent prefix {prefix}')
        else:
            # c == NT: 全长，应等于 final_state (而非 chunk_states[:,-1])
            _allclose(fs_local[0], fs_rec[0], atol=2e-2, rtol=2e-2,
                      name='final_state vs recurrent full')


# ---------------------------------------------------------------------------
# 3. varlen (cu_seqlens) 等价 + 形状
# ---------------------------------------------------------------------------
def test_varlen_equivalence():
    torch.manual_seed(2)
    H, HV, K, V = 4, 4, 64, 64
    dtype = torch.bfloat16
    scale = K ** -0.5
    # 两条序列：长度 128 (=2 chunks) 和 200 (=4 chunks)，packed 成 B=1
    cu_seqlens = torch.tensor([0, 128, 328], device=DEVICE, dtype=torch.int32)
    T = int(cu_seqlens[-1])
    B = 1
    q, k, v, g, beta, _ = _make_inputs(B, T, H, HV, K, V, dtype, with_init=False)
    init = torch.zeros(len(cu_seqlens) - 1, HV, V, K, device=DEVICE, dtype=torch.float32)

    o_local, fs_local, chunk_states = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, initial_state=init,
        output_final_state=True, cu_seqlens=cu_seqlens, state_v_first=True)
    o_fla, fs_fla = fla_chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, initial_state=init,
        output_final_state=True, use_qk_l2norm_in_kernel=False,
        transpose_state_layout=True, cu_seqlens=cu_seqlens)

    _allclose(o_local, o_fla, atol=2e-2, rtol=2e-2, name='varlen o')
    _allclose(fs_local, fs_fla, atol=2e-2, rtol=2e-2, name='varlen final_state')
    # 两条序列的 chunk 数之和: cdiv(128,64)+cdiv(200,64) = 2+4 = 6
    assert chunk_states.shape == (1, 6, HV, V, K), chunk_states.shape
    assert torch.isfinite(chunk_states.float()).all()


if __name__ == '__main__':
    test_equivalence_with_fla(256)
    test_equivalence_with_fla(200)
    test_chunk_states_vs_recurrent()
    test_varlen_equivalence()
    print('\n=== all chunk-port tests passed ===')
