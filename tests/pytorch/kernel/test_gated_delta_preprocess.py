import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaRuleImpl


class DefaultGatedDeltaRuleImpl(GatedDeltaRuleImpl):

    def chunk_gated_delta_rule(self, *args, **kwargs):
        raise NotImplementedError

    def fused_recurrent_gated_delta_rule(self, *args, **kwargs):
        raise NotImplementedError


def do_test():
    try:
        import triton  # noqa: F401
        from fla.modules.l2norm import l2norm_fwd  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _skip_bf16(dtype):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('bf16 not supported.')


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected)


def _make_qk_views(batch, seqlen, num_k_heads, num_v_heads, head_dim, dtype):
    qkv = torch.randn(
        batch,
        seqlen,
        (2 * num_k_heads + num_v_heads) * head_dim,
        dtype=dtype,
        device='cuda',
    )
    query, key, _ = torch.split(
        qkv,
        [
            num_k_heads * head_dim,
            num_k_heads * head_dim,
            num_v_heads * head_dim,
        ],
        dim=-1,
    )
    return query.unflatten(-1, (num_k_heads, head_dim)), key.unflatten(-1, (num_k_heads, head_dim))


def _fla_l2norm_reference(query, key, b, a, dt_bias, a_log_exp, kv_ratio, init_token_mask=None,
                          apply_qk_l2norm=True):
    if apply_qk_l2norm:
        from fla.modules.l2norm import l2norm_fwd

        query = query.repeat_interleave(kv_ratio, dim=-2)
        key = key.repeat_interleave(kv_ratio, dim=-2)
        query_shape = query.shape
        key_shape = key.shape
        query, _ = l2norm_fwd(query.reshape(-1, query.shape[-1]))
        key, _ = l2norm_fwd(key.reshape(-1, key.shape[-1]))
        query = query.view(query_shape)
        key = key.view(key_shape)
    else:
        query = query.repeat_interleave(kv_ratio, dim=-2)
        key = key.repeat_interleave(kv_ratio, dim=-2)
    if b.dim() == 4:
        b = b.flatten(-2, -1)
        a = a.flatten(-2, -1)
    beta = b.sigmoid()
    g = a_log_exp * F.softplus(a.float() + dt_bias)
    if init_token_mask is not None:
        g = g.masked_fill(init_token_mask[None, :, None], -1.0e6)
    return query, key, beta, g


@pytest.mark.skipif(not do_test(), reason='triton or cuda is not available')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('kv_ratio', [1, 4])
@pytest.mark.parametrize('use_init_token_mask', [False, True])
@pytest.mark.parametrize('apply_qk_l2norm', [False, True])
def test_gated_delta_preprocess_3d_ba(dtype, kv_ratio, use_init_token_mask, apply_qk_l2norm):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_preprocess import gated_delta_preprocess

    _skip_bf16(dtype)
    batch, seqlen, num_k_heads, head_dim = 2, 17, 4, 128
    num_v_heads = num_k_heads * kv_ratio
    query, key = _make_qk_views(batch, seqlen, num_k_heads, num_v_heads, head_dim, dtype)
    mixed_ba = torch.randn(batch, seqlen, 2 * num_v_heads, dtype=dtype, device='cuda')
    b, a = torch.split(mixed_ba, [num_v_heads, num_v_heads], dim=-1)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device='cuda')
    a_log_exp = -torch.rand(num_v_heads, dtype=torch.float32, device='cuda')
    init_token_mask = torch.zeros(seqlen, dtype=torch.bool, device='cuda')
    init_token_mask[0] = True
    init_token_mask[7] = True
    if not use_init_token_mask:
        init_token_mask = None

    out = gated_delta_preprocess(query,
                                 key,
                                 b,
                                 a,
                                 dt_bias,
                                 a_log_exp,
                                 kv_ratio,
                                 init_token_mask=init_token_mask,
                                 apply_qk_l2norm=apply_qk_l2norm)
    ref = _fla_l2norm_reference(query,
                                key,
                                b,
                                a,
                                dt_bias,
                                a_log_exp,
                                kv_ratio,
                                init_token_mask=init_token_mask,
                                apply_qk_l2norm=apply_qk_l2norm)

    _assert_close(out[0], ref[0])
    _assert_close(out[1], ref[1])
    _assert_close(out[2], ref[2])
    _assert_close(out[3], ref[3])


@pytest.mark.skipif(not do_test(), reason='triton or cuda is not available')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('kv_ratio', [1, 4])
@pytest.mark.parametrize('use_init_token_mask', [False, True])
@pytest.mark.parametrize('apply_qk_l2norm', [False, True])
def test_gated_delta_preprocess_4d_grouped_ba(dtype, kv_ratio, use_init_token_mask, apply_qk_l2norm):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_preprocess import gated_delta_preprocess

    _skip_bf16(dtype)
    batch, seqlen, num_k_heads, head_dim = 2, 17, 4, 128
    num_v_heads = num_k_heads * kv_ratio
    query, key = _make_qk_views(batch, seqlen, num_k_heads, num_v_heads, head_dim, dtype)
    mixed_ba = torch.randn(batch, seqlen, num_k_heads * 2 * kv_ratio, dtype=dtype, device='cuda')
    mixed_ba = mixed_ba.unflatten(-1, (num_k_heads, 2 * kv_ratio))
    b, a = mixed_ba.chunk(2, -1)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device='cuda')
    a_log_exp = -torch.rand(num_v_heads, dtype=torch.float32, device='cuda')
    init_token_mask = torch.zeros(seqlen, dtype=torch.bool, device='cuda')
    init_token_mask[0] = True
    init_token_mask[7] = True
    if not use_init_token_mask:
        init_token_mask = None

    out = gated_delta_preprocess(query,
                                 key,
                                 b,
                                 a,
                                 dt_bias,
                                 a_log_exp,
                                 kv_ratio,
                                 init_token_mask=init_token_mask,
                                 apply_qk_l2norm=apply_qk_l2norm)
    ref = _fla_l2norm_reference(query,
                                key,
                                b,
                                a,
                                dt_bias,
                                a_log_exp,
                                kv_ratio,
                                init_token_mask=init_token_mask,
                                apply_qk_l2norm=apply_qk_l2norm)

    _assert_close(out[0], ref[0])
    _assert_close(out[1], ref[1])
    _assert_close(out[2], ref[2])
    _assert_close(out[3], ref[3])


@pytest.mark.skipif(not do_test(), reason='triton or cuda is not available')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('kv_ratio', [1, 4])
@pytest.mark.parametrize('grouped_ba', [False, True])
@pytest.mark.parametrize('is_decoding', [False, True])
def test_gated_delta_preprocess_matches_default_prepare_inputs(dtype, kv_ratio, grouped_ba, is_decoding):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_preprocess import gated_delta_preprocess

    _skip_bf16(dtype)
    batch, seqlen, num_k_heads, head_dim = 2, 17, 4, 128
    num_v_heads = num_k_heads * kv_ratio
    query, key = _make_qk_views(batch, seqlen, num_k_heads, num_v_heads, head_dim, dtype)
    if grouped_ba:
        mixed_ba = torch.randn(batch, seqlen, num_k_heads * 2 * kv_ratio, dtype=dtype, device='cuda')
        mixed_ba = mixed_ba.unflatten(-1, (num_k_heads, 2 * kv_ratio))
        b, a = mixed_ba.chunk(2, -1)
    else:
        mixed_ba = torch.randn(batch, seqlen, 2 * num_v_heads, dtype=dtype, device='cuda')
        b, a = torch.split(mixed_ba, [num_v_heads, num_v_heads], dim=-1)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device='cuda')
    a_log_exp = -torch.rand(num_v_heads, dtype=torch.float32, device='cuda')
    init_token_mask = torch.zeros(seqlen, dtype=torch.bool, device='cuda')
    init_token_mask[0] = True
    init_token_mask[7] = True

    mask_arg = None if is_decoding else init_token_mask
    out = gated_delta_preprocess(query,
                                 key,
                                 b,
                                 a,
                                 dt_bias,
                                 a_log_exp,
                                 kv_ratio,
                                 init_token_mask=mask_arg,
                                 apply_qk_l2norm=False)
    ref = DefaultGatedDeltaRuleImpl().prepare_inputs(
        query,
        key,
        b,
        a,
        dt_bias,
        a_log_exp,
        kv_ratio,
        use_qk_l2norm_in_kernel=True,
        is_decoding=is_decoding,
        init_token_mask=init_token_mask,
    )

    _assert_close(out[0], ref[0])
    _assert_close(out[1], ref[1])
    _assert_close(out[2], ref[3])
    _assert_close(out[3], ref[2])
    assert not ref[4]
