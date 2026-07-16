import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _apply_rope_first(x, cos, sin, rope_interleaved):
    out = x.clone()
    x_rope = x[..., :64]
    if rope_interleaved:
        pair_cos = cos[..., :32].repeat_interleave(2, dim=-1)
        pair_sin = sin[..., :32].repeat_interleave(2, dim=-1)
        while pair_cos.dim() < x_rope.dim():
            pair_cos = pair_cos.unsqueeze(-2)
            pair_sin = pair_sin.unsqueeze(-2)
        rotated = torch.empty_like(x_rope)
        rotated[..., ::2] = -x_rope[..., 1::2]
        rotated[..., 1::2] = x_rope[..., ::2]
        out[..., :64] = x_rope * pair_cos + rotated * pair_sin
    else:
        cos = cos.unsqueeze(-2) if x_rope.dim() == 3 else cos
        sin = sin.unsqueeze(-2) if x_rope.dim() == 3 else sin
        x_l, x_h = x_rope.split(32, dim=-1)
        out[..., :32] = x_l * cos[..., :32] - x_h * sin[..., :32]
        out[..., 32:64] = x_h * cos[..., 32:64] + x_l * sin[..., 32:64]
    return out


@pytest.mark.parametrize('rope_interleaved', [True, False])
@pytest.mark.parametrize('heads', [32, 64])
def test_prepare_dsa_indexer_q_matches_unfused_quantization(rope_interleaved, heads):
    from lmdeploy.pytorch.kernels.cuda.apply_rotary_pos_emb import apply_rotary_pos_emb
    from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import per_token_group_quant_fp8
    from lmdeploy.pytorch.kernels.cuda.dsa_indexer import prepare_dsa_indexer_q

    torch.manual_seed(0)
    tokens = 11
    q = torch.randn(tokens, heads, 128, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(tokens, heads, device='cuda', dtype=torch.bfloat16)
    cos = torch.randn(tokens, 64, device='cuda', dtype=torch.bfloat16)
    sin = torch.randn(tokens, 64, device='cuda', dtype=torch.bfloat16)
    score_scale = 0.03125

    q_out, q_scale = prepare_dsa_indexer_q(q,
                                           weights,
                                           cos,
                                           sin,
                                           score_scale,
                                           torch.float8_e4m3fn,
                                           rope_interleaved=rope_interleaved)
    q_pe, q_nope = q.split([64, 64], dim=-1)
    k_pe = q.new_empty(tokens, 1, 64)
    q_pe, _ = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, interleaved=rope_interleaved)
    q_ref = torch.cat([q_pe, q_nope], dim=-1)
    q_ref, scale_ref = per_token_group_quant_fp8(q_ref.reshape(-1, 128),
                                                 128,
                                                 dtype=torch.float8_e4m3fn,
                                                 scale_fmt='ue8m0')
    q_ref = q_ref.reshape_as(q)
    scale_ref = scale_ref.reshape_as(weights)

    assert torch.equal(q_out, q_ref)
    torch.testing.assert_close(q_scale, scale_ref * weights.float() * score_scale, rtol=0, atol=0)


@pytest.mark.parametrize('rope_interleaved', [True, False])
@pytest.mark.parametrize('q_seqlens,kv_seqlens', [([1, 1], [3, 2]), ([3, 2], [5, 3])])
def test_prepare_dsa_indexer_k_cache_matches_prepared_k_fill(q_seqlens, kv_seqlens, rope_interleaved):
    from lmdeploy.pytorch.kernels.cuda.dsa_indexer import prepare_dsa_indexer_k, prepare_dsa_indexer_k_cache
    from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8

    torch.manual_seed(1)
    block_size = 4
    total_tokens = sum(q_seqlens)
    k = torch.randn(total_tokens, 128, device='cuda', dtype=torch.bfloat16)
    norm_weight = torch.randn(128, device='cuda', dtype=torch.float32)
    norm_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    cos = torch.randn(total_tokens, 64, device='cuda', dtype=torch.bfloat16)
    sin = torch.randn(total_tokens, 64, device='cuda', dtype=torch.bfloat16)
    cu_seqlen_q = torch.tensor([0] + q_seqlens, device='cuda', dtype=torch.int32).cumsum(0)
    kv_seqlens_t = torch.tensor(kv_seqlens, device='cuda', dtype=torch.int32)
    block_offsets = torch.tensor([[0, 1], [2, 3]], device='cuda', dtype=torch.int32)
    cache_ref = torch.zeros(4, block_size, 128, device='cuda', dtype=torch.float8_e4m3fn)
    scale_ref = torch.zeros(4, block_size, 1, device='cuda', dtype=torch.float32)
    cache_fused = torch.zeros_like(cache_ref)
    scale_fused = torch.zeros_like(scale_ref)

    k_prepared = prepare_dsa_indexer_k(k,
                                       norm_weight,
                                       norm_bias,
                                       cos,
                                       sin,
                                       eps=1e-6,
                                       rope_interleaved=rope_interleaved)
    fill_kv_cache_blocked_fp8(k_prepared[:, None],
                              None,
                              cache_ref[..., None, :],
                              None,
                              scale_ref[..., None, :],
                              None,
                              cu_seqlen_q=cu_seqlen_q,
                              kv_seqlens=kv_seqlens_t,
                              max_q_seqlen=max(q_seqlens),
                              block_offsets=block_offsets,
                              group_size=128,
                              scale_fmt='ue8m0')
    prepare_dsa_indexer_k_cache(k,
                                norm_weight,
                                norm_bias,
                                cos,
                                sin,
                                cache_fused,
                                scale_fused[..., 0],
                                cu_seqlen_q,
                                kv_seqlens_t,
                                block_offsets,
                                max_q_seqlen=max(q_seqlens),
                                eps=1e-6,
                                rope_interleaved=rope_interleaved)

    assert torch.equal(cache_fused, cache_ref)
    assert torch.equal(scale_fused, scale_ref)

    k_torch = F.layer_norm(k.float(), (128, ), norm_weight, norm_bias, 1e-6).to(torch.bfloat16)
    k_torch = _apply_rope_first(k_torch, cos, sin, rope_interleaved).to(torch.bfloat16)
    torch.testing.assert_close(k_prepared.float(), k_torch.float(), rtol=0, atol=0.0625)
