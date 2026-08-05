import pytest
import torch

GROUP_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn


def _has_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


pytestmark = pytest.mark.skipif(not _has_hopper(), reason='Gluon WGMMA kernel requires Hopper')


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
def _run_case(m: int, n: int, k: int, repeats: int = 1):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import fp8_gemm_nt

    a, b = _make_inputs(m, n, k)
    expected = _dequantized_reference(a, b)
    output = torch.full((m, n), torch.nan, device='cuda', dtype=torch.bfloat16)
    first_output = None

    for repeat in range(repeats):
        fp8_gemm_nt(a, b, output, None)
        if first_output is None:
            first_output = output.clone()
        else:
            assert torch.equal(output, first_output), f'output changed on repeat {repeat + 1}'

    assert torch.isfinite(output).all()
    assert _calc_diff(output, expected) < 1e-3


def test_fp8_gemm_nt_one_tile_one_k_block():
    """Milestone 1: get one synchronous TMA/WGMMA tile correct."""
    _run_case(m=64, n=128, k=128)


def test_fp8_gemm_nt_multiple_k_scale_blocks():
    """Milestone 2: promote every K partial with the correct A/B scales."""
    _run_case(m=64, n=128, k=384)


def test_fp8_gemm_nt_ragged_output_tiles():
    """Milestone 3: zero-pad loads and bound the M/N output edges."""
    _run_case(m=37, n=192, k=256)


def test_fp8_gemm_nt_multiple_mn_tiles():
    """Milestone 4: cover nonzero M/N program ids and ragged final tiles."""
    _run_case(m=137, n=320, k=384)


@pytest.mark.parametrize('m', [128, 129, 256, 257])
def test_fp8_gemm_nt_dispatch_boundaries(m):
    """Keep all three schedule families correct at their boundaries."""
    _run_case(m=m, n=192, k=384, repeats=3)


@pytest.mark.parametrize(
    ('m', 'k'),
    [
        (37, 640),
        (37, 1152),
        (128, 2176),
        (256, 512),
        (256, 640),
        (256, 1152),
        (256, 2176),
    ],
)
def test_fp8_gemm_nt_reuses_pipeline_stages_and_phases(m, k):
    """Exercise stage reuse and phase changes in the small and mid paths."""
    _run_case(m=m, n=192, k=k, repeats=5)


@torch.inference_mode()
def test_fp8_gemm_nt_large_grid_is_deterministic():
    """Catch CTA-concurrency bugs hidden by aggregate error tolerances."""
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import fp8_gemm_nt

    a, b = _make_inputs(m=1024, n=4096, k=1152)
    output = torch.empty((1024, 4096), device='cuda', dtype=torch.bfloat16)
    fp8_gemm_nt(a, b, output, None)
    expected = output.clone()

    for repeat in range(10):
        fp8_gemm_nt(a, b, output, None)
        assert torch.equal(output, expected), f'output changed on repeat {repeat + 1}'


def test_fp8_gemm_nt_tiny_m_reuses_eight_pipeline_stages():
    """Exercise reuse and a full phase wrap in the tuned M<=64 path."""
    _run_case(m=4, n=192, k=2176, repeats=5)


@pytest.mark.parametrize(('m', 'k'), [(1, 4096), (8, 8192)])
def test_fp8_gemm_nt_transposed_tiny_m(m, k):
    """Cover both stage depths of the transposed M<=8 schedule."""
    _run_case(m=m, n=192, k=k, repeats=5)
