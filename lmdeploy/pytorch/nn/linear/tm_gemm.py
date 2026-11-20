# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""W4A16 linear layers backed by the TurboMind GEMM channel.

The shapes mirror the AWQ family (plain, merged, QKV). A shared mixin swaps the backend operator and the post-load
weight update; everything else -- weight layout, loaders, TP splitting, forward plumbing -- stays inherited from the AWQ
classes.
"""

from contextlib import nullcontext

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl
from lmdeploy.pytorch.backends.tm_gemm_modules import LinearTmGemmBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context

from .awq import AwqLinear, MergedAwqLinear, QKVAwqLinear


class TmGemmLinearMixin:
    """TurboMind GEMM adaptation shared by the plain, merged, and QKV linears.

    The TurboMind route packs CPU-direct: at module build time the packed
    quant storages (qweight/scales/qzeros) are routed to the host, so the
    loader fills CPU buffers and ``update_weights`` packs from host memory.
    The raw weights never occupy device memory, so the former spill (copy to
    CPU, free the GPU storage) has nothing left to free.

    ``release_torch_weights`` (default true) selects this CPU-direct route.
    Modules that feed ``get_unquantized_weight`` (MLA absorption such as
    kv_b_proj) set it to False: they keep device-resident registered copies
    because the absorption path reads them through the module.

    Two seams of the AWQ shapes are overridden:

    - ``_build_impl`` requests ``LinearTmGemmBuildSpec``. The backend probe
      then picks the TurboMind kernel per module and falls back to the AWQ
      Triton kernel otherwise (which keeps CUDA-resident weights), mirroring
      the ``LMDEPLOY_W4A16_GEMM_BACKEND`` semantics.
    - ``update_weights`` packs into the TurboMind channel from the
      registered storages (host-resident under the CPU-direct route).
    """

    def _build_impl(self, in_features: int, out_features: int, w_bit: int, group_size: int, bias: bool,
                    output_dtype: torch.dtype, device: torch.device):
        """Build the backend operator with the TurboMind GEMM spec."""
        # Provider checks must inspect the device that owns these weights.
        device_guard = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
        with device_guard:
            return get_backend().build_op(
                LinearTmGemmBuildSpec(
                    in_features=in_features,
                    out_features=out_features,
                    w_bit=w_bit,
                    group_size=group_size,
                    bias=bias,
                    output_dtype=output_dtype,
                    prefix=self.tm_prefix,
                ),
                enable_deterministic=get_build_model_context().enable_deterministic,
            )

    @property
    def cpu_resident(self) -> bool:
        """Whether the packed quant storages live on the host (CPU-direct
        TurboMind route)."""
        return (self.release_torch_weights
                and isinstance(getattr(self, 'impl', None), TurbomindGemmW4A16Impl))

    def register_all_parameters(self, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
                                bias: torch.Tensor | None = None):
        """Register the packed storages as parameters.

        At module build time the storages are still the untouched CUDA
        empties ``create_weights`` produced. When the probe picked the
        TurboMind channel (and the module is not MLA-exempt), they are
        replaced with host empties so the loader fills CPU-direct buffers.
        Later calls re-register the loaded storages verbatim: the channel
        has already packed them, and a fresh allocation would only discard
        data. Bias always stays on the module device — forward reads it
        python-side on every call.
        """
        if self.cpu_resident and qweight.is_cuda:
            qweight = torch.empty(tuple(qweight.shape), dtype=qweight.dtype, device='cpu')
            scales = torch.empty(tuple(scales.shape), dtype=scales.dtype, device='cpu')
            qzeros = torch.empty(tuple(qzeros.shape), dtype=qzeros.dtype, device='cpu')
        super().register_all_parameters(qweight, scales, qzeros, bias)

    def _apply(self, fn, recurse=True):
        """Keep the host-resident quant storages out of module moves.

        They are the CPU-direct pack source: the engine's moves (the
        post-load ``Module.to(cuda)`` and wakeup's ``to(cuda)``) must not
        re-upload them. Bias and any other registered tensors still travel.
        """
        if not self.cpu_resident:
            return super()._apply(fn, recurse=recurse)
        pinned = [(name, self._parameters.pop(name)) for name in ('qweight', 'scales', 'qzeros')
                  if name in self._parameters]
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            for name, param in pinned:
                self._parameters[name] = param

    def update_weights(self):
        """Pack into the TurboMind channel.

        Under the CPU-direct route the registered storages are host-resident
        and the channel's ``prepare_weight`` performs the H2D transform; no
        device-side raw copy exists to spill. The Triton fallback keeps its
        torch weights, and the MLA-exempt modules keep theirs by design.
        """
        if self.qweight.numel() == 0:
            # A dummy/skipped module has empty storages; packing would read
            # them pointlessly.
            return
        qweight, scales, qzeros, bias = self.impl.update_weights(self.qweight, self.scales, self.qzeros, self.bias)
        self.register_all_parameters(qweight, scales, qzeros, bias)

    def sleep_release(self):
        """Release the native packed handle ahead of ``Module.to``.

        The packed weight is not a registered tensor, so the model move cannot free it. CPU-direct modules keep their
        host-resident storages; MLA-exempt modules keep the registered CUDA copies, which travel with the model move.
        """
        if not isinstance(self.impl, TurbomindGemmW4A16Impl):
            return
        self.impl.sleep_release()

    def sleep_wakeup(self):
        """Re-pack after the model move.

        CPU-direct modules re-pack from the unchanged host storages; MLA-exempt modules re-pack from their registered
        CUDA copies. No spill/restore exists in either path.
        """
        if not isinstance(self.impl, TurbomindGemmW4A16Impl):
            return
        if self.qweight.numel() == 0:
            raise RuntimeError('TurboMind W4A16 weights were never loaded; '
                               'wakeup cannot re-pack an empty module.')
        self.impl.sleep_wakeup(self.qweight, self.scales, self.qzeros, self.bias)


class TmGemmLinear(TmGemmLinearMixin, AwqLinear):
    """W4a16 linear that prefers the TurboMind GEMM kernel.

    Falls back to the AWQ Triton kernel per layer when the registry probe
    rejects the shape, mirroring the ``LMDEPLOY_W4A16_GEMM_BACKEND``
    semantics.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_bit: int,
        group_size: int,
        bias: bool,
        device: torch.device | None = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        layer_type: str = 'attn',
        release_torch_weights: bool = True,
        prefix: str = '',
    ):
        # The mixin reads these while the AWQ base class builds the operator.
        self.release_torch_weights = release_torch_weights
        self.tm_prefix = prefix
        super().__init__(
            in_features,
            out_features,
            w_bit=w_bit,
            group_size=group_size,
            bias=bias,
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
            layer_type=layer_type,
        )


class MergedTmGemmLinear(TmGemmLinearMixin, MergedAwqLinear):
    """Merged AWQ linear that prefers the TurboMind GEMM kernel.

    Gate/up style merged projections issue one TurboMind GEMM call over the concatenated output columns.
    """

    def __init__(self,
                 in_features: int,
                 all_out_features: list[int],
                 w_bit: int,
                 group_size: int,
                 bias: bool,
                 device: torch.device | None = None,
                 is_tp: bool = True,
                 out_names: list[int] | None = None,
                 layer_type: str = 'attn',
                 release_torch_weights: bool = True,
                 prefix: str = ''):
        self.release_torch_weights = release_torch_weights
        self.tm_prefix = prefix
        super().__init__(
            in_features,
            all_out_features=all_out_features,
            w_bit=w_bit,
            group_size=group_size,
            bias=bias,
            device=device,
            is_tp=is_tp,
            out_names=out_names,
            layer_type=layer_type,
        )


class QKVTmGemmLinear(TmGemmLinearMixin, QKVAwqLinear):
    """QKV AWQ linear that prefers the TurboMind GEMM kernel."""

    def __init__(self,
                 in_features: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 w_bit: int,
                 group_size: int,
                 bias: bool = False,
                 device: torch.device | None = None,
                 is_tp: bool = True,
                 num_replicate_kv_heads: int = 1,
                 release_torch_weights: bool = True,
                 prefix: str = ''):
        self.release_torch_weights = release_torch_weights
        self.tm_prefix = prefix
        super().__init__(
            in_features,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            w_bit=w_bit,
            group_size=group_size,
            bias=bias,
            device=device,
            is_tp=is_tp,
            num_replicate_kv_heads=num_replicate_kv_heads,
        )
