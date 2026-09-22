import pytest
import torch

GROUP_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn


def _has_gluon() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        from lmdeploy.pytorch.backends.cuda.blockedf8_modules import _has_gluon as has_gluon
    except (AttributeError, ImportError):
        return False
    return has_gluon()


pytestmark = pytest.mark.skipif(not _has_gluon(), reason='Gluon WGMMA kernel requires supported Triton on Hopper')


def _quantize_a(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize A with one scale per row and 128-wide K block."""
    m, k = a.shape
    assert k % GROUP_SIZE == 0

    blocks = a.float().view(m, k // GROUP_SIZE, GROUP_SIZE)
    fp8_info = torch.finfo(FP8_DTYPE)
    scale = blocks.abs().amax(dim=-1).clamp_min(1e-6) / fp8_info.max
    quant = (blocks / scale[..., None]).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)

    # Exercise the column-major logical [M, K / 128] scale layout used by the
    # DeepGEMM-facing LMDeploy quantization path. TMA requires the physical
    # stride between K-block columns to be 16-byte aligned.
    scale_alignment = 16 // scale.element_size()
    aligned_m = (m + scale_alignment - 1) // scale_alignment * scale_alignment
    aligned_scale = torch.empty_strided(scale.shape, (1, aligned_m), dtype=scale.dtype, device=scale.device)
    aligned_scale.copy_(scale)
    scale = aligned_scale
    return quant.view(m, k), scale


def _quantize_b_nt(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize B[N, K] with one scale per 128x128 (N, K) block."""
    n, k = b.shape
    assert k % GROUP_SIZE == 0

    padded_n = (n + GROUP_SIZE - 1) // GROUP_SIZE * GROUP_SIZE
    padded = torch.zeros((padded_n, k), dtype=torch.float32, device=b.device)
    padded[:n] = b

    blocks = padded.view(padded_n // GROUP_SIZE, GROUP_SIZE, k // GROUP_SIZE, GROUP_SIZE)
    fp8_info = torch.finfo(FP8_DTYPE)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-6) / fp8_info.max
    quant = (blocks / scale[:, None, :, None]).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    return quant.view(padded_n, k)[:n].contiguous(), scale


def _dequantized_reference(a: tuple[torch.Tensor, torch.Tensor],
                           b: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    a_quant, a_scale = a
    b_quant, b_scale = b
    m, k = a_quant.shape
    n = b_quant.size(0)
    k_blocks = k // GROUP_SIZE

    a_dequant = a_quant.float().view(m, k_blocks, GROUP_SIZE) * a_scale.float()[..., None]
    b_scale_per_row = b_scale.float().repeat_interleave(GROUP_SIZE, dim=0)[:n]
    b_dequant = b_quant.float().view(n, k_blocks, GROUP_SIZE) * b_scale_per_row[..., None]
    return (a_dequant.view(m, k) @ b_dequant.view(n, k).T).to(torch.bfloat16)


def _make_inputs(m: int, n: int, k: int):
    torch.manual_seed(42)
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

    # Make scale products clearly different across K and N blocks. This catches
    # kernels that accumulate all raw WGMMA partials and apply only one scale.
    num_k_blocks = k // GROUP_SIZE
    num_n_blocks = (n + GROUP_SIZE - 1) // GROUP_SIZE
    k_factors = torch.logspace(-1, 1, num_k_blocks, device='cuda')
    n_factors = torch.logspace(-0.5, 0.5, num_n_blocks, device='cuda')
    k_factors = k_factors.repeat_interleave(GROUP_SIZE)
    n_factors = n_factors.repeat_interleave(GROUP_SIZE)[:n]
    a = (a.float() * k_factors[None, :]).to(torch.bfloat16)
    b = (b.float() * n_factors[:, None] * k_factors[None, :]).to(torch.bfloat16)

    return _quantize_a(a), _quantize_b_nt(b)


def _calc_diff(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual = actual.double()
    expected = expected.double()
    return 1 - 2 * (actual * expected).sum() / (actual.square() + expected.square()).sum()


@torch.inference_mode()
def _run_case(m: int, n: int, k: int, repeats: int = 1, num_sms: int | None = None):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import fp8_gemm_nt

    a, b = _make_inputs(m, n, k)
    expected = _dequantized_reference(a, b)
    output = torch.full((m, n), torch.nan, device='cuda', dtype=torch.bfloat16)

    fp8_gemm_nt(a, b, output, None, num_sms=num_sms)
    if repeats > 1:
        first_output = output.clone()
    for repeat in range(1, repeats):
        fp8_gemm_nt(a, b, output, None, num_sms=num_sms)
        assert torch.equal(output, first_output), f'output changed on repeat {repeat + 1}'

    assert torch.isfinite(output).all()
    assert _calc_diff(output, expected) < 1e-3


def test_fp8_gemm_nt_one_tile_one_k_block():
    """Milestone 1: cover exactly one output tile and one scale block."""
    _run_case(m=64, n=32, k=128)


def test_fp8_gemm_nt_multiple_k_scale_blocks():
    """Milestone 2: promote every K partial with the correct A/B scales."""
    _run_case(m=64, n=128, k=384)


def test_fp8_gemm_nt_ragged_output_tiles():
    """Milestone 3: zero-pad loads and bound the M/N output edges."""
    _run_case(m=37, n=200, k=256)


def test_fp8_gemm_nt_multiple_mn_tiles():
    """Milestone 4: cover nonzero M/N program ids and ragged final tiles."""
    _run_case(m=137, n=320, k=384)


@pytest.mark.parametrize('m', [128, 129, 256, 257])
def test_fp8_gemm_nt_dispatch_boundaries(m):
    """Keep all three schedule families correct at their boundaries."""
    _run_case(m=m, n=192, k=384, repeats=3)


@pytest.mark.parametrize(
    ('m', 'n', 'expected'),
    [
        (128, 2048, True),
        (129, 2048, True),
        (256, 2048, True),
        (160, 4096, False),
        (192, 4096, True),
        (224, 4096, False),
        (256, 4096, True),
        (256, 8192, False),
        (257, 2048, False),
    ],
)
def test_fp8_gemm_nt_single_partition_policy(m, n, expected):
    """Use the single-partition schedule within its measured wave budget."""
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import _prefer_single_partition

    assert _prefer_single_partition(m, n, num_sms=132) is expected


@pytest.mark.parametrize(
    ('m', 'k'),
    [
        (37, 1152),
        (128, 2176),
        (256, 640),
        (256, 1152),
    ],
)
def test_fp8_gemm_nt_reuses_pipeline_stages_and_phases(m, k):
    """Exercise stage reuse and phase changes in the small and mid paths."""
    _run_case(m=m, n=192, k=k, repeats=5)


def test_fp8_gemm_nt_mid_m_compiles_at_one_pipeline_turn():
    """Guard the four-block scale queue that exposed a loop-carried type
    error."""
    _run_case(m=256, n=192, k=512, repeats=5)


def test_fp8_gemm_nt_persistent_grid_is_correct_and_deterministic():
    """Exercise persistent tile reuse and detect CTA-concurrency races."""
    num_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
    n = 4096
    num_tiles_n = n // GROUP_SIZE
    num_tiles_m = max(2, num_sms // num_tiles_n + 1)
    m = num_tiles_m * 256
    _run_case(m=m, n=n, k=1152, repeats=10)


def test_fp8_gemm_nt_persistent_grid_honors_sm_budget():
    """Redistribute every logical tile across a deliberately smaller grid."""
    device_num_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
    n = 4096
    num_tiles_n = n // GROUP_SIZE
    num_tiles_m = max(2, device_num_sms // num_tiles_n + 1)
    m = num_tiles_m * 256
    _run_case(m=m, n=n, k=1152, repeats=3, num_sms=max(1, device_num_sms - 8))


def test_fp8_gemm_nt_tiny_m_reuses_eight_pipeline_stages():
    """Exercise reuse and a full phase wrap in the tuned M<=64 path."""
    _run_case(m=4, n=192, k=2176, repeats=5)


@pytest.mark.parametrize(('m', 'k'), [(1, 4096), (32, 4096), (24, 6144), (8, 8192), (16, 8192)])
def test_fp8_gemm_nt_transposed_tiny_m(m, k):
    """Cover the K-dependent transposed schedule and both stage depths."""
    _run_case(m=m, n=192, k=k, repeats=5)


@pytest.mark.parametrize(('k', 'expected'), [(4096, 32), (5120, 32), (6144, 24), (8192, 16), (15360, 16),
                                              (16384, 8), (17408, 8)])
def test_fp8_gemm_nt_transpose_m_limit(k, expected):
    """Keep long-K B-reload regressions outside the transposed schedule."""
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import _get_transpose_m_limit

    assert _get_transpose_m_limit(k) == expected


def test_fp8_gemm_nt_rejects_invalid_dtype_and_layout():
    """Reject invalid contracts during specialization or descriptor setup."""
    from triton.compiler.errors import CompileTimeAssertionFailure

    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import fp8_gemm_nt

    a, b = _make_inputs(m=64, n=128, k=256)
    output = torch.empty((64, 128), device='cuda', dtype=torch.bfloat16)

    # Triton 3.7 validates the descriptor element width before Gluon reaches
    # the equivalent compile-time dtype assertion used by Triton 3.6.
    with pytest.raises((CompileTimeAssertionFailure, AssertionError)):
        fp8_gemm_nt((a[0].to(torch.float16), a[1]), b, output, None)
    with pytest.raises(CompileTimeAssertionFailure, match='A scales must be column-major'):
        fp8_gemm_nt((a[0], a[1].contiguous()), b, output, None)
    with pytest.raises(CompileTimeAssertionFailure, match='B scales must use FP32'):
        fp8_gemm_nt(a, (b[0], b[1].to(torch.float16)), output, None)
    with pytest.raises(CompileTimeAssertionFailure, match='output must use BF16'):
        fp8_gemm_nt(a, b, output.to(torch.float16), None)

    ragged_a, ragged_b = _make_inputs(m=64, n=129, k=256)
    misaligned_output = torch.empty((64, 129), device='cuda', dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match='strides must be 16-byte aligned'):
        fp8_gemm_nt(ragged_a, ragged_b, misaligned_output, None)

    device_num_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
    with pytest.raises(TypeError, match='num_sms must be an integer or None'):
        fp8_gemm_nt(a, b, output, None, num_sms=1.5)
    with pytest.raises(ValueError, match='num_sms must be between 1 and'):
        fp8_gemm_nt(a, b, output, None, num_sms=0)
    with pytest.raises(ValueError, match='num_sms must be between 1 and'):
        fp8_gemm_nt(a, b, output, None, num_sms=device_num_sms + 1)
