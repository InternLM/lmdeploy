# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Build request for the TurboMind float-quantized linear route (NVFP4 and
compressed-tensors FP8)."""

from dataclasses import dataclass

from .base import BuildSpec


@dataclass(frozen=True)
class LinearTmQuantBuildSpec(BuildSpec):
    """Immutable requirements for a float-quantized linear served by the
    TurboMind GEMM channel.

    ``lane`` selects the checkpoint format and the serving query:

    - ``'nvfp4'``: NVFP4 (E2M1 codes, group-16 e4m3 scales, per-tensor
      global scale), fp16 compute.
    - ``'ct_fp8'``: compressed-tensors float-quantized FP8 weights
      (per-channel or 128-blocked scales), bf16 compute with in-kernel
      dynamic fp8 activation quantization.

    There is no non-TurboMind fallback for either lane: a declined route
    is a loud error naming the reason.
    """

    in_features: int
    out_features: int
    lane: str
    bias: bool
    # ct_fp8 only: 1 for per-channel scales, 128 for blocked scales.
    fp8_block_out: int | None = None
    # Module path carried for routing logs only.
    prefix: str = ''
