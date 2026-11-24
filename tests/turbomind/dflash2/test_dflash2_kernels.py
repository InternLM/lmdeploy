import math

import pytest
import torch

from lmdeploy.messages import SpeculativeConfig
from lmdeploy.turbomind.spec_decode import DRAFT_WEIGHT_SPECS, draft_quant_formats, normalize_spec_method

from . import dflash2_kernels

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
KERNELS_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available() or not dflash2_kernels.is_available(),
                                      reason='TurboMind DFlash2 kernels unavailable')
DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def ref_grouped_conv(hidden, delta, base, block_size, group_size, taps):
    rows, hidden_size = hidden.shape
    out = torch.zeros_like(hidden)
    group_of = torch.arange(hidden_size, device=hidden.device) // group_size
    for t in range(rows):
        pos = t % block_size
        for tap in range(taps):
            if t - tap < 0:
                break
            if tap > 0 and pos < tap:
                continue
            coeff = base[tap] + delta[t, tap][group_of]
            out[t] += coeff * hidden[t - tap]
    return out


def ref_selector(hidden, unary, predecessor, successor, candidate_ids, anchor, top_k):
    batch, length, rank = hidden.shape
    out = torch.empty(batch, length, top_k, top_k, dtype=torch.float32, device=hidden.device)
    for b in range(batch):
        for l in range(length):
            cand = candidate_ids[b, l]
            succ = successor[cand]
            for p in range(top_k):
                pred = anchor[b] if l == 0 else candidate_ids[b, l - 1, p]
                acc = (predecessor[pred][None, :] * hidden[b, l][None, :] * succ).sum(-1)
                out[b, l, p] = unary[b, l] + acc
    return out


def ref_attention(q, k_ctx, v_ctx, k_blk, v_blk, scale, window):
    tq, nh, hd = q.shape
    tc, nkv, _ = k_ctx.shape
    out = torch.zeros_like(q)
    for i in range(tq):
        first = max(0, tc + i - window + 1)
        for h in range(nh):
            kvh = h * nkv // nh
            k = torch.cat([k_ctx[first:tc, kvh], k_blk[:, kvh]], dim=0)
            v = torch.cat([v_ctx[first:tc, kvh], v_blk[:, kvh]], dim=0)
            p = torch.softmax((k @ q[i, h]) * scale, dim=0)
            out[i, h] = p @ v
    return out


def pack_int4(x):
    """Pack [rows, nkv, 128] into int4 data/params and return the dequantized
    value."""
    assert x.shape[-1] == 128
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / 15).to(x.dtype)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = mn.to(x.dtype)
    q = torch.round((x.float() - zero.float()) / scale.float()).clamp(0, 15).to(torch.uint8)
    q0, q1, q2, q3 = q[..., 0:32], q[..., 32:64], q[..., 64:96], q[..., 96:128]
    data = torch.empty(*x.shape[:-1], 64, dtype=torch.uint8, device=x.device)
    data[..., 0::2] = q0 | (q1 << 4)
    data[..., 1::2] = q2 | (q3 << 4)
    params = torch.cat([scale, zero], dim=-1)
    dequant = q.float() * scale.float() + zero.float()
    return data, params, dequant


# --------------------------------------------------------------------------- #
# kernel tests
# --------------------------------------------------------------------------- #
@KERNELS_REQUIRED
@pytest.mark.parametrize('dtype', DTYPES)
def test_grouped_conv_matches_reference(dtype):
    torch.manual_seed(0)
    rows, hidden_size, group_size, taps, block_size = 8, 32, 16, 2, 4
    groups = hidden_size // group_size
    hidden = torch.randn(rows, hidden_size, dtype=dtype, device='cuda') * 0.5
    delta = torch.randn(rows, taps, groups, dtype=dtype, device='cuda') * 0.5
    base = torch.randn(taps, hidden_size, dtype=dtype, device='cuda') * 0.5

    out = dflash2_kernels.grouped_conv(hidden, delta, base, block_size, groups, group_size, taps)
    ref = ref_grouped_conv(hidden.float(), delta.float(), base.float(), block_size, group_size, taps)
    torch.testing.assert_close(out.float(), ref, atol=2e-3 if dtype == torch.float16 else 2e-2, rtol=0)


@KERNELS_REQUIRED
@pytest.mark.parametrize('dtype', DTYPES)
def test_candidate_selector_matches_reference(dtype):
    torch.manual_seed(0)
    batch, length, top_k, rank, vocab = 1, 2, 4, 8, 16
    hidden = torch.randn(batch, length, rank, dtype=dtype, device='cuda') * 0.5
    unary = torch.randn(batch, length, top_k, dtype=torch.float32, device='cuda')
    predecessor = torch.randn(vocab, rank, dtype=dtype, device='cuda') * 0.5
    successor = torch.randn(vocab, rank, dtype=dtype, device='cuda') * 0.5
    candidate_ids = torch.randint(0, vocab, (batch, length, top_k), dtype=torch.int32, device='cuda')
    anchor = torch.randint(0, vocab, (batch, ), dtype=torch.int32, device='cuda')

    out = dflash2_kernels.candidate_selector_scores(hidden, unary, predecessor, successor, candidate_ids, anchor,
                                                    top_k)
    ref = ref_selector(hidden.float(), unary, predecessor.float(), successor.float(), candidate_ids, anchor, top_k)
    torch.testing.assert_close(out, ref, atol=2e-4 if dtype == torch.float16 else 2e-2, rtol=0)


def _attention_inputs(dtype=torch.float16):
    torch.manual_seed(0)
    tq, tc, nh, nkv, head_dim = 3, 5, 4, 2, 128
    q = torch.randn(tq, nh, head_dim, dtype=dtype, device='cuda') * 0.5
    k_ctx = torch.randn(tc, nkv, head_dim, dtype=dtype, device='cuda') * 0.5
    v_ctx = torch.randn(tc, nkv, head_dim, dtype=dtype, device='cuda') * 0.5
    k_blk = torch.randn(tq, nkv, head_dim, dtype=dtype, device='cuda') * 0.5
    v_blk = torch.randn(tq, nkv, head_dim, dtype=dtype, device='cuda') * 0.5
    return q, k_ctx, v_ctx, k_blk, v_blk


@KERNELS_REQUIRED
@pytest.mark.parametrize('dtype', DTYPES)
def test_block_attention_matches_reference(dtype):
    q, k_ctx, v_ctx, k_blk, v_blk = _attention_inputs(dtype=dtype)
    scale, window = 1.0 / math.sqrt(q.shape[-1]), 4
    out = dflash2_kernels.dflash2_attention(q, k_ctx, v_ctx, k_blk, v_blk, scale, window)
    ref = ref_attention(q.float(), k_ctx.float(), v_ctx.float(), k_blk.float(), v_blk.float(), scale, window)
    torch.testing.assert_close(out.float(), ref, atol=3e-3 if dtype == torch.float16 else 3e-2, rtol=0)


@KERNELS_REQUIRED
@pytest.mark.parametrize('dtype', DTYPES)
def test_int4_context_kv_matches_dequantized_reference(dtype):
    q, k_ctx, v_ctx, k_blk, v_blk = _attention_inputs(dtype=dtype)
    k_data, k_param, k_deq = pack_int4(k_ctx)
    v_data, v_param, v_deq = pack_int4(v_ctx)
    scale, window = 1.0 / math.sqrt(q.shape[-1]), 4

    out = dflash2_kernels.dflash2_attention_int4(q, k_data, k_param, v_data, v_param, k_blk, v_blk, scale, window)
    ref = ref_attention(q.float(), k_deq, v_deq, k_blk.float(), v_blk.float(), scale, window)
    torch.testing.assert_close(out.float(), ref, atol=5e-3 if dtype == torch.float16 else 5e-2, rtol=0)


@KERNELS_REQUIRED
def test_select_base_resets_on_new_request():
    ctx_len = torch.tensor([11], dtype=torch.int32, device='cuda')
    entry0 = torch.tensor([0], dtype=torch.int32, device='cuda')
    assert dflash2_kernels.select_base(ctx_len, entry0).item() == 0
    entry9 = torch.tensor([9], dtype=torch.int32, device='cuda')
    assert dflash2_kernels.select_base(ctx_len, entry9).item() == 11


# --------------------------------------------------------------------------- #
# loader / config tests (no CUDA needed)
# --------------------------------------------------------------------------- #
def test_dflash2_draft_spec():
    spec = DRAFT_WEIGHT_SPECS['dflash2']
    assert spec.input_model == 'qwen3-dflash2'
    assert spec.weight_source == 'sidecar'


def test_spec_method_aliases():
    assert normalize_spec_method('qwen3_5_mtp') == 'mtp'
    assert normalize_spec_method('dflash2') == 'dflash2'


def test_ctx_kv_int4_requires_dflash2():
    SpeculativeConfig(method='dflash2', dflash_ctx_kv_int4=True)
    with pytest.raises(ValueError):
        SpeculativeConfig(method='eagle3', dflash_ctx_kv_int4=True)


def test_draft_quant_formats_nvfp4():
    hf_config = {
        'quantization_config': {
            'quant_method': 'compressed-tensors',
            'config_groups': {
                'group_0': {
                    'format': 'nvfp4-pack-quantized',
                    'weights': {
                        'group_size': 16
                    }
                }
            }
        }
    }
    formats = draft_quant_formats(hf_config)
    assert len(formats) == 1
    assert formats[0].block_in == 16


def test_draft_quant_formats_plain():
    assert draft_quant_formats({}) == []
