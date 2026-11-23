# Copyright (c) OpenMMLab. All rights reserved.
"""Eager Ascend mHC reductions matching the common backend contract."""

import glob
import os

import torch

from ...hc_prepost import HCPrePostImpl

_NATIVE_LIB_LOADED = False


def _ops_registered():
    try:
        getattr(torch.ops._C_ascend, "npu_hc_pre_v2")
        getattr(torch.ops._C_ascend, "npu_hc_post")
        return True
    except (AttributeError, RuntimeError):
        return False


def _candidate_lib_paths():
    # Explicit override first, then a vllm-ascend checkout, then any
    # installed vllm_ascend package.
    override = os.environ.get("VLLM_ASCEND_C_PATH")
    if override:
        yield override
    yield ('/vllm-workspace/vllm-ascend/vllm_ascend/'
           'vllm_ascend_C.cpython-312-aarch64-linux-gnu.so')
    try:
        import importlib.util
        spec = importlib.util.find_spec('vllm_ascend')
        if spec is not None and spec.submodule_search_locations:
            pkg = spec.submodule_search_locations[0]
            for p in sorted(glob.glob(os.path.join(pkg, 'vllm_ascend_C*.so'))):
                yield p
    except (ImportError, AttributeError, ValueError):
        pass


def _load_native_ops():
    """Register npu_hc_pre_v2/npu_hc_post from vllm-ascend's C++ extension.

    The fused mHC kernels live in the ``_C_ascend`` torch library namespace;
    their AscendC implementations resolve from the custom OPP package shipped
    inside vllm_ascend, whose vendor dir must be on ASCEND_CUSTOM_OPP_PATH
    when the library is dlopened. Both can be wired up lazily from a running
    process (verified: setting the env var after torch_npu init works), so
    eager stays the fallback and no launch-script change is required.
    """
    global _NATIVE_LIB_LOADED
    if _NATIVE_LIB_LOADED or _ops_registered():
        _NATIVE_LIB_LOADED = True
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return
    for lib in _candidate_lib_paths():
        if not os.path.isfile(lib):
            continue
        pkg_dir = os.path.dirname(os.path.abspath(lib))
        for vendor in sorted(glob.glob(
                os.path.join(pkg_dir, '_cann_ops_custom', 'vendors', '*'))):
            if not os.path.isdir(vendor):
                continue
            cur = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
            if vendor not in cur.split(':'):
                os.environ['ASCEND_CUSTOM_OPP_PATH'] = (
                    vendor + (':' + cur if cur else ''))
        try:
            torch.ops.load_library(lib)
        except Exception:  # noqa: BLE001
            continue
        if _ops_registered():
            _NATIVE_LIB_LOADED = True
            return


class AscendHCPrePostImpl(HCPrePostImpl):
    def __init__(self, hc_mult, sinkhorn_iters, eps):
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        _load_native_ops()

    def _native_available(self):
        if not _NATIVE_LIB_LOADED:
            return False
        return _ops_registered()

    def pre_native(self, x, hc_fn, hc_scale, hc_base, out_dtype,
                   norm_weight=None, norm_eps=1e-6):
        # Keep the exact vLLM-Ascend operator ordering for Ascend.
        layer_input, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
            x, hc_fn, hc_scale, hc_base, self.hc_mult,
            self.sinkhorn_iters, norm_eps, self.eps)
        if layer_input.dtype != out_dtype:
            layer_input = layer_input.to(out_dtype)
        if norm_weight is not None:
            import torch_npu
            layer_input = torch_npu.npu_rms_norm(
                layer_input, norm_weight, epsilon=norm_eps)[0]
        return layer_input, post, comb

    def pre(self, x, mixes, hc_scale, hc_base, out_dtype,
            norm_weight=None, norm_eps=1e-6):
        pre, post, comb = mixes.float().split(
            [self.hc_mult, self.hc_mult, self.hc_mult ** 2], dim=-1)
        pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:self.hc_mult]) + self.eps
        post = 2 * torch.sigmoid(
            post * hc_scale[1] + hc_base[self.hc_mult:2 * self.hc_mult])
        comb = (comb * hc_scale[2] + hc_base[2 * self.hc_mult:]).unflatten(
            -1, (self.hc_mult, self.hc_mult))
        comb = torch.softmax(comb, dim=-1) + self.eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
        # The contract casts the reduced state before applying RMSNorm.
        y = self.pre_reduce(x, pre, out_dtype)
        if norm_weight is not None:
            import torch_npu
            # Same fused op as pre_native: fp32 accumulation inside the
            # kernel replaces the eager float() soup
            # (Pow/Mean/Rsqrt/Mul/Mul/Cast per call).
            y = torch_npu.npu_rms_norm(y, norm_weight, epsilon=norm_eps)[0]
        return y, post, comb

    def pre_reduce(self, x, pre, out_dtype):
        return (x.float() * pre.float().unsqueeze(-1)).sum(-2).to(out_dtype)

    def post_expand(self, x, residual, post, comb):
        if self._native_available():
            rank2 = x.dim() == 2
            if rank2:
                x = x.unsqueeze(0)
                residual = residual.unsqueeze(0)
                post = post.unsqueeze(0)
                comb = comb.unsqueeze(0)
            out = torch.ops._C_ascend.npu_hc_post(
                x, residual, post, comb)
            return out.squeeze(0) if rank2 else out
        # comb[..., src, out] mixes the source residual streams.
        mixed = torch.matmul(comb.float().transpose(-1, -2), residual.float())
        return (mixed + post.float().unsqueeze(-1) * x.float().unsqueeze(-2)).to(residual.dtype)
