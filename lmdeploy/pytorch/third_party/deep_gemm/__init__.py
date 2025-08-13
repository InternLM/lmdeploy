# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

try:
    import deep_gemm  # noqa: F401
except ImportError:
    logger.exception('DeepGemm is not installed. Please install https://github.com/deepseek-ai/DeepGEMM.')

from deep_gemm import ceil_div, get_m_alignment_for_contiguous_layout  # noqa: F401, E402

try:
    from deep_gemm import fp8_gemm_nt
except Exception:
    from deep_gemm.jit_kernels.gemm import gemm_fp8_fp8_bf16_nt

    @contextmanager
    def _log_jit_build(M: int, N: int, K: int):
        from deep_gemm.jit.runtime import RuntimeCache

        if hasattr(RuntimeCache, 'get'):
            func_name = 'get'
        else:
            func_name = '__getitem__'
        origin_func = getattr(RuntimeCache, func_name)

        def __patched_func(self, *args, **kwargs):
            ret = origin_func(self, *args, **kwargs)
            if ret is None:
                logger.warning(f'DeepGemm build <gemm_fp8_fp8_bf16_nt>: M={M}, N={N}, K={K}. Please waiting.')
            return ret

        setattr(RuntimeCache, func_name, __patched_func)
        yield
        setattr(RuntimeCache, func_name, origin_func)

    def fp8_gemm_nt(a, b, d, c, recipe=None, compiled_dim='nk', disable_ue8m0_cast=False):
        M, K = a[0].shape
        N, _ = b[0].shape
        with _log_jit_build(M, N, K):
            gemm_fp8_fp8_bf16_nt(a, b, d)
