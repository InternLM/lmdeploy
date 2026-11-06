# Copyright (c) OpenMMLab. All rights reserved.
"""H200 schedules passing cold-weight CUDA-graph A/B and FP64 checks.

Measured on 2026-11-05 for GLM-5.3 TP1 KDA and KPool projections. Unlisted token counts use the default linear. Enable
with LMDEPLOY_ENABLE_SKINNY_GEMM=1.
"""

from lmdeploy.pytorch.kernels.cuda.skinny_gemm import SkinnyGemmConfig

SM90_SKINNY_GEMM_PLANS = {
    (64, 4096): {
        1: SkinnyGemmConfig(1, 4096, 4, 1),
        2: SkinnyGemmConfig(1, 4096, 4, 1),
        4: SkinnyGemmConfig(1, 4096, 4, 1),
        6: SkinnyGemmConfig(1, 1024, 8, 1),
        8: SkinnyGemmConfig(1, 1024, 8, 1),
    },
    (128, 4096): {
        1: SkinnyGemmConfig(1, 4096, 4, 1),
        2: SkinnyGemmConfig(1, 4096, 4, 1),
        4: SkinnyGemmConfig(1, 4096, 4, 1),
        6: SkinnyGemmConfig(1, 1024, 8, 1),
        8: SkinnyGemmConfig(1, 1024, 8, 1),
    },
    (256, 4096): {
        1: SkinnyGemmConfig(1, 4096, 4, 1),
        2: SkinnyGemmConfig(1, 4096, 4, 1),
        4: SkinnyGemmConfig(1, 4096, 4, 1),
        6: SkinnyGemmConfig(1, 1024, 4, 1),
    },
}
