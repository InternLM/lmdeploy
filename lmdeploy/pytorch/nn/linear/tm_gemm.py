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
from lmdeploy.pytorch.backends.cuda.tm_weight_release import TorchWeightSpill
from lmdeploy.pytorch.backends.tm_gemm_modules import LinearTmGemmBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context

from .awq import AwqLinear, MergedAwqLinear, QKVAwqLinear


class TmGemmLinearMixin:
    """TurboMind GEMM adaptation shared by the plain, merged, and QKV linears.

    ``release_torch_weights`` (default true) lets ``update_weights`` spill the
    packed torch weights to an exact CPU copy and free the GPU storage.
    Modules that feed ``get_unquantized_weight`` (MLA absorption such as
    kv_b_proj) set it to False; they keep their torch copies.

    Two seams of the AWQ shapes are overridden:

    - ``_build_impl`` requests ``LinearTmGemmBuildSpec``. The backend probe
      then picks the TurboMind kernel per module and falls back to the AWQ
      Triton kernel otherwise, mirroring the ``LMDEPLOY_W4A16_GEMM_BACKEND``
      semantics.
    - ``update_weights`` packs into the TurboMind channel and then spills the
      raw torch weights to exact CPU copies, freeing GPU storage.
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

    def update_weights(self):
        """Pack into the TurboMind channel, then spill the torch weights."""
        if self.qweight.numel() == 0:
            # The module was not part of this update bucket and its weights
            # are spilled; packing would read the emptied storages.
            return
        qweight, scales, qzeros, bias = self.impl.update_weights(self.qweight, self.scales, self.qzeros, self.bias)
        self.register_all_parameters(qweight, scales, qzeros, bias)
        if not isinstance(self.impl, TurbomindGemmW4A16Impl):
            # Triton fallback keeps its torch weights; nothing to spill.
            return
        if not self.release_torch_weights:
            # MLA absorption reads the torch copies through
            # get_unquantized_weight; keep them resident.
            return
        if self._spill is None:
            self._spill = TorchWeightSpill()
        # Bias is read in python on every forward, so it stays on the GPU.
        self._spill.spill(
            dict(
                qweight=self.qweight,
                scales=self.scales,
                qzeros=self.qzeros,
            )
        )


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
        self._spill: TorchWeightSpill | None = None
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
        self._spill: TorchWeightSpill | None = None
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
        self._spill: TorchWeightSpill | None = None
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
