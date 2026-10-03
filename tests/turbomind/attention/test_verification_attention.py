import math

import pytest
import torch

from lmdeploy.turbomind import _tm

from .verification_attention import run_verification_attention


def _supports_verification_attention():
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    optin_shared_memory = getattr(properties, 'shared_memory_per_block_optin',
                                  properties.shared_memory_per_block)
    return (hasattr(_tm, 'verification_attention')
            and properties.major == 9
            and optin_shared_memory >= 112 * 1024)


pytestmark = pytest.mark.skipif(
    not _supports_verification_attention(),
    reason='verification attention requires SM90 and 112 KiB opt-in shared memory')


def _rotate(x,
            position,
            *,
            dim,
            base,
            factor,
            mrope_mode=0,
            section=(0, 0, 0)):
    if dim == 0:
        return x
    pairs = torch.arange(dim // 2, device=x.device)
    frequency = factor**-1 * base**(-(2 * pairs.float()) / dim)
    if mrope_mode == 0:
        timestep = position[..., None].float()
    elif mrope_mode == 1:
        limits = torch.tensor(section, device=x.device).cumsum(0)
        axis = torch.where(pairs < limits[0], 0,
                           torch.where(pairs < limits[1], 1, 2))
        timestep = position[..., axis].float()
    else:
        cycle = pairs // 3
        axis = pairs % 3
        use_axis = torch.where((axis == 1) & (cycle < section[1]), 1,
                               torch.where((axis == 2) & (cycle < section[2]),
                                           2, 0))
        timestep = position[..., use_axis].float()
    angle = timestep * frequency
    source = x[..., :dim].float().reshape(*x.shape[:-1], dim // 2, 2)
    first = torch.cos(angle) * source[..., 0] - torch.sin(angle) * source[...,
                                                                          1]
    second = torch.cos(angle) * source[..., 1] + torch.sin(angle) * source[...,
                                                                           0]
    rotated = torch.stack((first, second), dim=-1).flatten(-2).to(x.dtype)
    return torch.cat((rotated, x[..., dim:]), dim=-1)


def _positions(request, length, *, mrope_mode, position_ids, position_delta,
               mrope_length, device):
    scalar = torch.arange(length, device=device)
    if mrope_mode == 0:
        return scalar
    explicit = scalar < int(mrope_length[request])
    fallback = (scalar + int(position_delta[request]))[:, None].expand(-1, 3)
    return torch.where(explicit[:, None], position_ids[request, :length],
                       fallback)


def _reference(case, prefix_k, prefix_v, packed_qkv, q_bias,
               mrope_position_ids, mrope_position_delta, mrope_length):
    outputs = []
    p = case['p']
    hq = case['hq']
    hk = case['hk']
    d = case['d']
    qkv_heads = hq + 2 * hk
    prefix_begin = 0
    query_begin = 0
    for request, history in enumerate(case['histories']):
        prefix_end = prefix_begin + history
        query_end = query_begin + p
        all_positions = _positions(request,
                                   history + p,
                                   mrope_mode=case['mrope_mode'],
                                   position_ids=mrope_position_ids,
                                   position_delta=mrope_position_delta,
                                   mrope_length=mrope_length,
                                   device=packed_qkv.device)
        query_positions = all_positions[history:]

        q = packed_qkv[query_begin:query_end, :hq].clone()
        if q_bias.numel():
            q = (q + q_bias).to(q.dtype)
        q = _rotate(q,
                    query_positions[:, None],
                    dim=case['rope_dim'],
                    base=case['rope_base'],
                    factor=case['rope_factor'],
                    mrope_mode=case['mrope_mode'],
                    section=case['mrope_section'])
        tail_k = packed_qkv[query_begin:query_end, hq:hq + hk]
        k = torch.cat((prefix_k[prefix_begin:prefix_end], tail_k), dim=0)
        k = _rotate(k,
                    all_positions[:, None],
                    dim=case['rope_dim'],
                    base=case['rope_base'],
                    factor=case['rope_factor'],
                    mrope_mode=case['mrope_mode'],
                    section=case['mrope_section'])
        tail_v = packed_qkv[query_begin:query_end, hq + hk:qkv_heads]
        v = torch.cat((prefix_v[prefix_begin:prefix_end], tail_v), dim=0)

        q_group = q.reshape(p, hk, hq // hk, d).float()
        score = torch.einsum('thgd,shd->thgs', q_group,
                             k.float()) / math.sqrt(d)
        keys = torch.arange(history + p, device=q.device)
        queries = history + torch.arange(p, device=q.device)
        valid = keys[None, :] <= queries[:, None]
        if case['window']:
            valid &= keys[None, :] >= queries[:, None] - case['window'] + 1
        score = score.masked_fill(~valid[:, None, None, :], float('-inf'))
        probability = score.softmax(-1)
        out = torch.einsum('thgs,shd->thgd', probability, v.float())
        outputs.append(out.reshape(p, hq, d).to(packed_qkv.dtype))
        prefix_begin = prefix_end
        query_begin = query_end
    return torch.cat(outputs)


def _make_case(*,
               dtype=torch.bfloat16,
               d=128,
               p=8,
               hq=4,
               hk=1,
               histories=(63, 257, 1025),
               window=0,
               bias=False,
               rope_type=0,
               rope_dim=0,
               mrope_mode=0,
               mrope_section=(0, 0, 0),
               requested_splits=128):
    return dict(dtype=dtype,
                d=d,
                p=p,
                hq=hq,
                hk=hk,
                histories=histories,
                window=window,
                bias=bias,
                rope_type=rope_type,
                rope_dim=rope_dim,
                rope_base=1_000_000.,
                rope_factor=1.,
                mrope_mode=mrope_mode,
                mrope_section=mrope_section,
                requested_splits=requested_splits)


def _run(case, finished=None):
    torch.manual_seed(91)
    device = torch.device('cuda')
    dtype = case['dtype']
    b = len(case['histories'])
    d, p, hq, hk = case['d'], case['p'], case['hq'], case['hk']
    sum_history = sum(case['histories'])
    sum_query = b * p
    scale = 0.2
    prefix_k = torch.randn(sum_history, hk, d, device=device,
                           dtype=dtype) * scale
    prefix_v = torch.randn_like(prefix_k) * scale
    packed_qkv = torch.randn(
        sum_query, hq + 2 * hk, d, device=device, dtype=dtype) * scale
    q_bias = (torch.randn(hq, d, device=device, dtype=dtype) * scale
              if case['bias'] else torch.empty(0, device=device, dtype=dtype))

    prefix_offsets = torch.tensor(
        [0] + list(torch.tensor(case['histories']).cumsum(0).tolist()),
        device=device,
        dtype=torch.int32)
    q_offsets = torch.arange(0, (b + 1) * p,
                             p,
                             device=device,
                             dtype=torch.int32)
    key_lengths = [history + p for history in case['histories']]
    k_offsets = torch.tensor(
        [0] + list(torch.tensor(key_lengths).cumsum(0).tolist()),
        device=device,
        dtype=torch.int32)
    if finished is None:
        finished = torch.zeros(b, device=device, dtype=torch.bool)

    block_len = 64
    page_counts = [(length + block_len - 1) // block_len
                   for length in key_lengths]
    block_ptr_offsets = torch.tensor(
        [0] + list(torch.tensor(page_counts).cumsum(0).tolist()),
        device=device,
        dtype=torch.int32)
    page_count = sum(page_counts)
    page_bytes = 2 * hk * block_len * d * torch.tensor(
        [], dtype=dtype).element_size()
    cache = torch.full((page_count, page_bytes),
                       0xA5,
                       device=device,
                       dtype=torch.uint8)
    permutation = torch.randperm(page_count, device='cpu').tolist()
    base = cache.data_ptr()
    block_ptrs = torch.tensor(
        [base + physical * page_bytes for physical in permutation],
        device=device,
        dtype=torch.int64)

    max_key_length = max(key_lengths)
    mrope_position_ids = torch.empty((b, max_key_length, 3),
                                     device=device,
                                     dtype=torch.int32)
    for request, length in enumerate(key_lengths):
        pos = torch.arange(length, device=device, dtype=torch.int32)
        mrope_position_ids[request, :length, 0] = pos
        mrope_position_ids[request, :length, 1] = pos * 2 + 1
        mrope_position_ids[request, :length, 2] = pos * 3 + 2
    mrope_position_delta = torch.tensor([3, 5, 7],
                                        device=device,
                                        dtype=torch.int32)
    mrope_length = torch.tensor(
        [key_lengths[0], case['histories'][1] + p // 2, key_lengths[2]],
        device=device,
        dtype=torch.int32)

    output = torch.full((sum_query, hq, d),
                        float('nan'),
                        device=device,
                        dtype=dtype)
    partial_capacity = 4096
    partial_o = torch.full((partial_capacity, hq, d),
                           float('nan'),
                           device=device)
    partial_ml = torch.full((partial_capacity, hq, 2),
                            float('nan'),
                            device=device)
    split_count = run_verification_attention(
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        prefix_offsets=prefix_offsets,
        packed_qkv=packed_qkv,
        q_bias=q_bias,
        output=output,
        cache_storage=cache,
        block_ptrs=block_ptrs,
        block_ptr_offsets=block_ptr_offsets,
        q_offsets=q_offsets,
        k_offsets=k_offsets,
        finished=finished,
        partial_o=partial_o,
        partial_ml=partial_ml,
        query_head_count=hq,
        kv_head_count=hk,
        head_dim=d,
        block_len=block_len,
        max_query_length=p,
        max_key_length=max_key_length,
        window_size=case['window'],
        requested_max_split_count=case['requested_splits'],
        rope_type=case['rope_type'],
        rope_dim=case['rope_dim'],
        rope_base=case['rope_base'],
        rope_factor=case['rope_factor'],
        mrope_mode=case['mrope_mode'],
        mrope_section=case['mrope_section'],
        mrope_position_ids=mrope_position_ids,
        mrope_position_delta=mrope_position_delta,
        mrope_length=mrope_length)
    torch.cuda.synchronize()
    reference = _reference(case, prefix_k, prefix_v, packed_qkv, q_bias,
                           mrope_position_ids, mrope_position_delta,
                           mrope_length)
    return output, reference, partial_o, partial_ml, split_count


@pytest.mark.parametrize('case', [
    _make_case(d=128, p=8, hq=4, bias=True, requested_splits=8),
    _make_case(d=256, p=8, hq=6),
    _make_case(d=256, p=16, hq=8, histories=(65, 511, 1025)),
    _make_case(dtype=torch.float16, d=256, p=4, hq=16, window=129),
])
def test_verification_attention(case):
    output, reference, _, _, split_count = _run(case)
    if case['d'] == 128:
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
        expected_split_count = min(8, (2 * sm_count + 2) // 3)
        assert split_count == expected_split_count
    else:
        assert split_count > 1
    torch.testing.assert_close(output, reference, rtol=3e-2, atol=3e-2)


def test_finished_request_publishes_neutral_partials():
    case = _make_case(d=128, p=8, hq=4, requested_splits=8)
    finished = torch.tensor([False, True, False], device='cuda')
    output, reference, partial_o, partial_ml, split_count = _run(
        case, finished)
    assert torch.count_nonzero(output[8:16]) == 0
    assert not torch.isnan(output).any()
    torch.testing.assert_close(output[:8], reference[:8], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(output[16:],
                               reference[16:],
                               rtol=3e-2,
                               atol=3e-2)
    slots = partial_o[:24 * split_count].reshape(24, split_count, 4, 128)[8:16]
    ml = partial_ml[:24 * split_count].reshape(24, split_count, 4, 2)[8:16]
    assert torch.count_nonzero(slots) == 0
    assert torch.isneginf(ml[..., 0]).all()
    assert torch.count_nonzero(ml[..., 1]) == 0

    direct_cases = (
        _make_case(d=128, p=8, hq=4, requested_splits=1),
        _make_case(d=256, p=8, hq=4, requested_splits=1),
        _make_case(dtype=torch.float16,
                   d=256,
                   p=8,
                   hq=4,
                   requested_splits=1),
    )
    for direct_case in direct_cases:
        (direct_output, direct_reference, direct_partial_o, direct_partial_ml,
         direct_split_count) = _run(direct_case, finished)
        assert direct_split_count == 1
        assert torch.count_nonzero(direct_output[8:16]) == 0
        assert not torch.isnan(direct_output).any()
        torch.testing.assert_close(
            direct_output[:8], direct_reference[:8], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            direct_output[16:], direct_reference[16:], rtol=3e-2, atol=3e-2)
        assert torch.isnan(direct_partial_o).all()
        assert torch.isnan(direct_partial_ml).all()


@pytest.mark.parametrize('case', [
    _make_case(dtype=torch.float16,
               d=128,
               p=8,
               hq=4,
               requested_splits=1,
               rope_type=1,
               rope_dim=128),
    _make_case(d=128,
               p=8,
               hq=4,
               rope_type=1,
               rope_dim=128,
               mrope_mode=1,
               mrope_section=(16, 24, 24)),
    _make_case(d=128,
               p=8,
               hq=4,
               rope_type=1,
               rope_dim=128,
               mrope_mode=2,
               mrope_section=(16, 24, 24)),
])
def test_verification_attention_query_transforms(case):
    output, reference, _, _, split_count = _run(case)
    if case['requested_splits'] == 1:
        assert split_count == 1
    torch.testing.assert_close(output, reference, rtol=4e-2, atol=4e-2)
