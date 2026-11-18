# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""W4A16 linear implementation driven by the TurboMind Linear API.

Differences from the AWQ Triton W4A16 implementation:

- Activation quantization happens inside the kernel (``ConvertInput``); the
  python side only fixes shape and contiguity.
- The initial load publishes packed weights without a per-module sync. Hot
  updates keep the sync because a concurrent forward may still run on engine
  streams.
- Optional spill of the packed torch weights frees the GPU copies (see
  ``tm_weight_release``).
- Communication stays on the python side: ``all_reduce`` plus
  ``reduce_scatter_by_tp_sizes`` when a scatter split is requested.
"""

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch import envs as _envs
from lmdeploy.turbomind.linear import Linear, Weight, get_linear
from lmdeploy.turbomind.weight_format import AWQFormat
from lmdeploy.utils import get_logger

from ..awq_modules import LinearW4A16Impl
from ..tm_gemm_modules import LinearTmGemmBuildSpec
from .awq_modules import AwqLinearW4A16Impl
from .tm_probe import turbomind_support_reason

logger = get_logger('lmdeploy')


class TurbomindGemmW4A16Impl(LinearW4A16Impl):
    """Adapt AWQ W4A16 parameters to the TurboMind Linear API with spill
    support and SiLU gate/up fusion."""

    # The channel has no one-input-many-weights entry. Keep batched
    # projections on their per-projection fallback.
    supports_batched = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        fusion_type: str | None = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.fusion_type = fusion_type
        self._linear: Linear | None = None
        self._weight: Weight | None = None
        self._plan = None
        self._published = False

    def _release(self):
        weight, self._weight = self._weight, None
        if weight is None:
            self._linear = None
            return
        # Forward is asynchronous and may use any engine stream. Release the
        # prepared weight only after all device work completes.
        torch.cuda.synchronize()
        weight.close()
        self._linear = None

    def __del__(self):
        try:
            linear = self._linear
            if linear is None:
                return
            with torch.cuda.device(linear.device):
                self._release()
        except Exception:
            pass

    def update_weights(
        self,
        qweight,
        scales,
        qzeros,
        bias: torch.Tensor | None = None,
    ):
        """Pack the loader products into the TurboMind channel.

        With ``fusion_type='silu'`` the arguments are ``(gate, up)`` tuples
        and the packed handle computes ``silu(x @ gate.T) * (x @ up.T)`` in
        one kernel call.
        """
        self._release()

        linear = get_linear()
        if self._plan is None:
            # The registry query is deterministic for a fixed format and
            # dtype. Cache it so hot updates skip the family walk.
            self._plan = linear.get_weight_plan(
                weight_format=AWQFormat(block_in=self.group_size),
                dtype=scales[0].dtype if self.fusion_type == 'silu' else scales.dtype,
                fusion_type=self.fusion_type,
            )

        if self.fusion_type == 'silu':
            weight = linear.fuse_weight(
                (qweight[0], qweight[1]),
                plan=self._plan,
                scales=(scales[0], scales[1]),
                zeros=(qzeros[0], qzeros[1]),
            )
        else:
            weight = linear.prepare_weight(qweight, plan=self._plan, scales=scales, zeros=qzeros)

        if self._published:
            # Hot update: a concurrent forward may execute on any engine
            # stream. Publish only after packing completes.
            torch.cuda.current_stream().synchronize()
        else:
            # Initial load: no forward is in flight while weights load, so
            # the per-module sync is dropped. The transient peak stays pinned
            # to the single-layer staging upper bound.
            self._published = True

        self._linear = linear
        self._weight = weight
        return qweight, scales, qzeros, bias

    def forward(
        self,
        x,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        all_reduce: bool = False,
        group: torch.distributed.ProcessGroup | None = None,
        rank: int = 0,
        scatter_size: list[int] | None = None,
    ):
        linear = self._linear
        weight = self._weight
        if weight is None:
            raise RuntimeError('update_weights() must run before forward (TM packed weight missing)')

        # The channel consumes a [m, k] row-major contiguous view. DLPack
        # exports without strides, so a non-contiguous input would be read
        # with silent layout corruption.
        in_shape = x.shape
        x = x.flatten(0, -2)
        if not x.is_contiguous():
            x = x.contiguous()

        exec_plan = linear.get_exec_plan(x, weight)
        out, _ = linear(x, weight, exec_plan=exec_plan)

        # The kernel has no epilogue bias. Add it per layer in python.
        if bias is not None:
            out = out + bias

        out = out.reshape(in_shape[:-1] + (self.out_features,))
        if out.ndim == 2:
            out = out.unsqueeze(0)

        if all_reduce:
            if scatter_size is not None:
                from lmdeploy.pytorch.distributed import reduce_scatter_by_tp_sizes

                out = reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out

    def forward_batched(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            'The TurboMind linear channel does not support batched projections; '
            'BatchedLinear must keep this implementation out of its gate.'
        )


def _build_linear_tm_gemm(spec: LinearTmGemmBuildSpec) -> LinearW4A16Impl:
    """Build the W4A16 linear implementation, preferring TurboMind GEMM.

    Reuses the shared registry probe and env semantics: ``auto`` falls
    back to the AWQ Triton kernel, a forced ``turbomind`` raises on an
    unsupported layer, any other value keeps the Triton kernel.
    """
    if spec.fusion_type == 'silu' and spec.bias:
        raise ValueError('fused gate/up projections reject bias: the epilogue '
                         'activation leaves no place for a per-half bias add')
    provider = _envs.w4a16_gemm_backend
    label = spec.prefix or 'w4a16_linear'
    shapes = f'{spec.in_features}x{spec.out_features}'
    if provider in ('auto', 'turbomind'):
        reason = turbomind_support_reason(
            spec.in_features, spec.out_features, spec.w_bit, spec.group_size, spec.output_dtype
        )
        if reason is None:
            logger.info(f'W4A16 GEMM routing: {label} [{shapes}] -> turbomind (backend={provider}).')
            return TurbomindGemmW4A16Impl(
                spec.in_features,
                spec.out_features,
                spec.group_size,
                fusion_type=spec.fusion_type,
            )
        if provider == 'turbomind':
            from lmdeploy import turbomind

            raise RuntimeError(
                f"TurboMind GEMM linear was requested but is unavailable or incompatible: {reason}."
            ) from turbomind._import_error
        detail = f'fallback: {reason}'
    else:
        # Any value other than auto/turbomind keeps the historical Triton path.
        detail = f'backend={provider}'
    logger.info(f'W4A16 GEMM routing: {label} [{shapes}] -> awq triton ({detail}).')
    return AwqLinearW4A16Impl(spec.in_features, spec.out_features, spec.w_bit, spec.group_size)
