# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        'CUDA is required for the static FP8 linear backend',
        allow_module_level=True,
    )

from lmdeploy.pytorch.backends.cuda.static_fp8_modules import (
    TritonLinearStaticF8Impl,
    _per_tensor_quant_fp8_e4m3fn_inductor,
)
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import (
    per_tensor_quant_fp8,
)

_REQUIRES_FP8_GPU = pytest.mark.skipif(
    torch.cuda.get_device_capability() < (8, 9),
    reason='Static FP8 optimization tests require SM89 or SM90+',
)


def _make_case(
    *,
    num_tokens,
    in_features,
    out_features,
    vector_weight_scale,
    with_bias,
):
    generator = torch.Generator(device='cuda')
    generator.manual_seed(
        20260730
        + num_tokens
        + in_features
        + out_features
    )
    output_dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(quant_dtype).max

    x = (
        torch.randn(
            (num_tokens, in_features),
            dtype=output_dtype,
            device='cuda',
            generator=generator,
        )
        * 0.2
    )
    weight_float = (
        torch.randn(
            (out_features, in_features),
            dtype=output_dtype,
            device='cuda',
            generator=generator,
        )
        * 0.1
    )
    input_scale = (
        x.float().abs().amax() / fp8_max
    ).clamp_min(1e-8).reshape(1)

    if vector_weight_scale:
        weight_scale = (
            weight_float.float().abs().amax(dim=1)
            / fp8_max
        ).clamp_min(1e-8)
        weight = torch.clamp(
            weight_float.float()
            / weight_scale[:, None],
            min=-fp8_max,
            max=fp8_max,
        ).to(quant_dtype)
    else:
        weight_scale = (
            weight_float.float().abs().amax() / fp8_max
        ).clamp_min(1e-8).reshape(1)
        weight = torch.clamp(
            weight_float.float() / weight_scale,
            min=-fp8_max,
            max=fp8_max,
        ).to(quant_dtype)

    bias = None
    if with_bias:
        bias = torch.randn(
            (out_features,),
            dtype=output_dtype,
            device='cuda',
            generator=generator,
        )

    return (
        x,
        weight.contiguous(),
        input_scale.contiguous(),
        weight_scale.contiguous(),
        bias,
    )


def _make_impl(
    monkeypatch,
    *,
    in_features,
    out_features,
    use_scaled_mm,
    use_compiled_quant,
    compiled_token_counts='1,2,3,8,16,29,32,128',
):
    monkeypatch.setenv(
        'LMDEPLOY_STATIC_FP8_USE_SCALED_MM',
        '1' if use_scaled_mm else '0',
    )
    monkeypatch.setenv(
        'LMDEPLOY_STATIC_FP8_USE_COMPILED_QUANT',
        '1' if use_compiled_quant else '0',
    )
    monkeypatch.setenv(
        'LMDEPLOY_STATIC_FP8_COMPILED_QUANT_TOKEN_COUNTS',
        compiled_token_counts,
    )
    return TritonLinearStaticF8Impl(
        in_features,
        out_features,
        out_dtype=torch.bfloat16,
    )


def _assert_outputs_match(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-2,
        rtol=2e-2,
    )


def test_static_fp8_linear_feature_gates_are_default_off(
    monkeypatch,
):
    monkeypatch.delenv(
        'LMDEPLOY_STATIC_FP8_USE_SCALED_MM',
        raising=False,
    )
    monkeypatch.delenv(
        'LMDEPLOY_STATIC_FP8_USE_COMPILED_QUANT',
        raising=False,
    )
    impl = TritonLinearStaticF8Impl(
        128,
        256,
        out_dtype=torch.bfloat16,
    )
    assert not impl.use_scaled_mm
    assert not impl.use_compiled_quant


@pytest.mark.parametrize(
    'shape',
    [
        (1, 4096, 2560),
        (3, 2048, 4096),
        (32, 384, 4096),
    ],
)
@pytest.mark.parametrize(
    ('vector_weight_scale', 'with_bias'),
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
@_REQUIRES_FP8_GPU
@torch.inference_mode()
def test_static_fp8_scaled_mm_matches_triton(
    monkeypatch,
    shape,
    vector_weight_scale,
    with_bias,
):
    num_tokens, in_features, out_features = shape
    case = _make_case(
        num_tokens=num_tokens,
        in_features=in_features,
        out_features=out_features,
        vector_weight_scale=vector_weight_scale,
        with_bias=with_bias,
    )
    reference = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=False,
        use_compiled_quant=False,
    )
    expected = reference.forward(*case)

    scaled_mm = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=True,
        use_compiled_quant=False,
    )
    actual = scaled_mm.forward(*case)
    _assert_outputs_match(actual, expected)


@pytest.mark.parametrize(
    'num_tokens',
    [1, 2, 3, 8, 16, 29, 32, 128],
)
@pytest.mark.parametrize('in_features', [384, 2048, 4096])
@_REQUIRES_FP8_GPU
@torch.inference_mode()
def test_static_fp8_compiled_quant_is_exact(
    num_tokens,
    in_features,
):
    generator = torch.Generator(device='cuda')
    generator.manual_seed(
        20260730 + num_tokens + in_features,
    )
    x = torch.randn(
        (num_tokens, in_features),
        dtype=torch.bfloat16,
        device='cuda',
        generator=generator,
    )
    scale = torch.tensor(
        [0.0125],
        dtype=torch.float32,
        device='cuda',
    )

    expected = per_tensor_quant_fp8(
        x,
        scale,
        quant_dtype=torch.float8_e4m3fn,
    )
    actual = _per_tensor_quant_fp8_e4m3fn_inductor(
        x,
        scale,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('num_tokens', [1, 3, 32])
@_REQUIRES_FP8_GPU
@torch.inference_mode()
def test_static_fp8_compiled_quant_dispatch_matches_uncompiled(
    monkeypatch,
    num_tokens,
):
    in_features = 4096
    out_features = 2560
    case = _make_case(
        num_tokens=num_tokens,
        in_features=in_features,
        out_features=out_features,
        vector_weight_scale=True,
        with_bias=True,
    )
    uncompiled = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=True,
        use_compiled_quant=False,
    )
    expected = uncompiled.forward(*case)

    compiled = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=True,
        use_compiled_quant=True,
    )
    actual = compiled.forward(*case)
    assert torch.equal(actual, expected)


@_REQUIRES_FP8_GPU
@torch.inference_mode()
def test_static_fp8_compiled_quant_respects_token_allowlist(
    monkeypatch,
):
    in_features = 4096
    out_features = 2560
    case = _make_case(
        num_tokens=3,
        in_features=in_features,
        out_features=out_features,
        vector_weight_scale=True,
        with_bias=False,
    )
    uncompiled = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=True,
        use_compiled_quant=False,
    )
    expected = uncompiled.forward(*case)

    allowlist_excludes_three = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=True,
        use_compiled_quant=True,
        compiled_token_counts='1,2,4,8,16,32',
    )
    actual = allowlist_excludes_three.forward(*case)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('num_tokens', [1, 32])
@_REQUIRES_FP8_GPU
@torch.inference_mode()
def test_static_fp8_scaled_mm_compiled_quant_cuda_graph(
    monkeypatch,
    num_tokens,
):
    in_features = 4096
    out_features = 2560
    (
        static_x,
        weight,
        input_scale,
        weight_scale,
        bias,
    ) = _make_case(
        num_tokens=num_tokens,
        in_features=in_features,
        out_features=out_features,
        vector_weight_scale=True,
        with_bias=True,
    )
    reference = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=False,
        use_compiled_quant=False,
    )
    optimized = _make_impl(
        monkeypatch,
        in_features=in_features,
        out_features=out_features,
        use_scaled_mm=True,
        use_compiled_quant=True,
    )

    # Compile every kernel before graph capture.
    optimized.forward(
        static_x,
        weight,
        input_scale,
        weight_scale,
        bias,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = optimized.forward(
            static_x,
            weight,
            input_scale,
            weight_scale,
            bias,
        )

    for replay_id in range(1, 4):
        replay_x = torch.randn_like(static_x)
        replay_x.mul_(replay_id / 4)
        static_x.copy_(replay_x)
        graph.replay()
        torch.cuda.synchronize()

        expected = reference.forward(
            replay_x,
            weight,
            input_scale,
            weight_scale,
            bias,
        )
        torch.cuda.synchronize()
        _assert_outputs_match(graph_output, expected)
