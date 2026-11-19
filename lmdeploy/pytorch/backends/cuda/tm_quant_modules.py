# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Float-quantized linear implementations driven by the TurboMind Linear API
(NVFP4 and compressed-tensors FP8 lanes).

Both lanes are weight-only float formats with no native fallback: the
TurboMind channel is the only serving path, so a declined route is a loud
error. Merged projections (gate/up, QKV) keep one packed handle per shard
because NVFP4 carries a per-tensor global scale per checkpoint module;
forward concatenates the shard outputs before the module-level activation.

Serving queries (registry-verified on SM86):

- ``nvfp4``: NVFP4Format, fp16 compute; inputs are widened/narrowed to
  half in python like the W4A16 route.
- ``ct_fp8``: FP8Format, bf16 compute with ``input_dtype`` fp8; the
  kernel quantizes activations dynamically (CT `token` strategy), so the
  python side passes bf16 activations untouched.
"""

from typing import TYPE_CHECKING

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import get_logger

from ..tm_gemm_modules import LinearW4A16Impl
from ..tm_quant_modules import LinearTmQuantBuildSpec

logger = get_logger('lmdeploy')

if TYPE_CHECKING:
    from lmdeploy.turbomind.linear import Linear, Weight


def _load_turbomind_channel(lane: str):
    """Import the TurboMind linear channel plus the lane's format, or return
    the blocker."""
    from lmdeploy import turbomind
    if not turbomind.is_available():
        return None, turbomind._import_error
    from lmdeploy.turbomind.linear import get_linear
    from lmdeploy.turbomind.weight_format import FP8Format, NVFP4Format

    fmt = NVFP4Format if lane == 'nvfp4' else FP8Format
    return (get_linear, fmt), None


def _lane_plan_kwargs(lane: str, fp8_block_out: int | None):
    """Registry query parameters and the python-side activation dtype of one
    lane."""
    if lane == 'nvfp4':
        return dict(block_in=16), None, torch.float16
    return dict(block_out=fp8_block_out if fp8_block_out is not None else 1), torch.float8_e4m3fn, torch.bfloat16


class TurbomindQuantLinearImpl(LinearW4A16Impl):
    """Adapt float-quantized parameters to the TurboMind Linear API."""

    # The channel has no one-input-many-weights entry. Keep batched
    # projections on their per-projection fallback.
    supports_batched = False

    def __init__(self, in_features: int, out_features: int, lane: str, fp8_block_out: int | None = None):
        if lane not in ('nvfp4', 'ct_fp8'):
            raise ValueError(f'Unknown TurboMind quant lane: {lane!r}')
        self.in_features = in_features
        self.out_features = out_features
        self.lane = lane
        self.fp8_block_out = fp8_block_out
        self._linear: Linear | None = None
        self._weights: list[Weight] = []
        self._plan = None
        self._published = False

    def _release(self):
        weights, self._weights = self._weights, []
        if not weights:
            self._linear = None
            return
        # Forward is asynchronous and may use any engine stream. Release the
        # prepared weights only after all device work completes.
        torch.cuda.synchronize()
        for weight in weights:
            weight.close()
        self._linear = None

    def __del__(self):
        try:
            linear = self._linear
            if linear is None and not self._weights:
                return
            if linear is not None:
                with torch.cuda.device(linear.device):
                    self._release()
            else:
                self._release()
        except Exception:
            pass

    def update_weights(
        self,
        weights: list[torch.Tensor],
        scales: list[torch.Tensor],
        global_scales: list[torch.Tensor | None] | None = None,
        biases: list[torch.Tensor | None] | None = None,
    ):
        """Pack one weight per shard into the TurboMind channel.

        ``weights``/``scales`` hold the raw checkpoint-layout tensors of
        each shard (already TP-sliced). The format's ``finalize`` runs
        inside ``prepare_weight``: NVFP4 consumes the checkpoint global
        scale (a divisor) as-is, FP8 per-channel scales arrive as
        ``[N, 1]`` and expand to the kernel's K-grouped layout.
        """
        self._release()

        loaded, err = _load_turbomind_channel(self.lane)
        if loaded is None:
            raise RuntimeError(
                'TurboMind linear channel is unavailable on this installation.'
            ) from err
        get_linear, fmt_cls = loaded

        linear = get_linear()
        if self._plan is None:
            # The registry query is deterministic for a fixed format and
            # dtype. Cache it so hot updates skip the family walk.
            fmt_kwargs, input_dtype, compute_dtype = _lane_plan_kwargs(self.lane, self.fp8_block_out)
            self._plan = linear.get_weight_plan(
                weight_format=fmt_cls(**fmt_kwargs),
                dtype=compute_dtype,
                input_dtype=input_dtype,
            )

        if global_scales is None:
            global_scales = [None] * len(weights)
        prepared = []
        with linear._activate():
            for shard_weight, shard_scales, shard_global_scale in zip(weights, scales, global_scales, strict=True):
                prepared.append(
                    linear.prepare_weight(
                        shard_weight,
                        plan=self._plan,
                        scales=shard_scales,
                        global_scale=shard_global_scale,
                    ))

        if self._published:
            # Hot update: a concurrent forward may execute on any engine
            # stream. Publish only after packing completes.
            torch.cuda.current_stream().synchronize()
        else:
            # Initial load: no forward is in flight while weights load, so
            # the per-module sync is dropped.
            self._published = True

        self._linear = linear
        self._weights = prepared

    def sleep_wakeup(self, weights, scales, global_scales=None, biases=None):
        """Re-pack the restored torch weights after wakeup."""
        return self.update_weights(weights, scales, global_scales, biases)

    def forward(
        self,
        x,
        biases: list[torch.Tensor | None] | None = None,
        all_reduce: bool = False,
        group: torch.distributed.ProcessGroup | None = None,
        rank: int = 0,
        scatter_size: list[int] | None = None,
    ):
        linear = self._linear
        if not self._weights:
            raise RuntimeError('update_weights() must run before forward (TM packed weight missing)')

        in_shape = x.shape
        input_dtype = x.dtype
        _fmt, _in, activation_dtype = _lane_plan_kwargs(self.lane, self.fp8_block_out)
        x = x.flatten(0, -2)
        if input_dtype != activation_dtype:
            x = x.to(activation_dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        outs = []
        for weight in self._weights:
            exec_plan = linear.get_exec_plan(x, weight)
            out, _ = linear(x, weight, exec_plan=exec_plan)
            outs.append(out)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)

        # The kernel has no epilogue bias. Add it per layer in python.
        if biases is not None and any(bias is not None for bias in biases):
            bias = torch.cat([bias for bias in biases])
            out = out + bias

        out = out.reshape(in_shape[:-1] + (self.out_features,))
        if out.ndim == 2:
            out = out.unsqueeze(0)
        if input_dtype != activation_dtype:
            out = out.to(dtype=input_dtype)

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


def _quant_route_reason(lane: str, fp8_block_out: int | None) -> str | None:
    """Why the registry declines this lane's plan, or None when accepted."""
    try:
        from lmdeploy.turbomind.linear import get_linear
        from lmdeploy.turbomind.weight_format import FP8Format, NVFP4Format
    except Exception as e:  # pragma: no cover - import failure path
        return f'TurboMind linear channel failed to import: {e}'
    fmt_kwargs, input_dtype, compute_dtype = _lane_plan_kwargs(lane, fp8_block_out)
    fmt_cls = NVFP4Format if lane == 'nvfp4' else FP8Format
    try:
        get_linear().get_weight_plan(
            weight_format=fmt_cls(**fmt_kwargs),
            dtype=compute_dtype,
            input_dtype=input_dtype,
        )
    except NotImplementedError as e:
        return f'no GEMM family accepts the {lane} plan: {e}'
    return None


def _build_linear_tm_quant(spec: LinearTmQuantBuildSpec):
    """Build the float-quantized linear implementation on the TurboMind
    channel.

    Neither lane has a native fallback: ``off`` names the offline-dequant
    alternative, and an unsupported device (registry rejection) is a loud
    error under both ``auto`` and ``on``.
    """
    lane = spec.lane
    state = _envs.tm_kernel_state(lane)
    label = spec.prefix or lane
    shapes = f'{spec.in_features}x{spec.out_features}'
    if state == 'off':
        raise RuntimeError(
            f'{lane} route is disabled by LMDEPLOY_TM_KERNELS={lane}=off; there is no native fallback '
            'for this checkpoint format. Dequantize the checkpoint offline (weight-only float '
            'formats have no Triton path) or clear the switch.')
    reason = _quant_route_reason(lane, spec.fp8_block_out)
    if reason is not None:
        raise RuntimeError(
            f'{lane} route is unavailable on this device: {reason}. '
            'The TurboMind kernel family is the only serving path for this format.')
    logger.info(f'Quant GEMM routing: {label} [{shapes}] -> turbomind {lane} (state={state}).')
    return TurbomindQuantLinearImpl(spec.in_features, spec.out_features, lane, spec.fp8_block_out)
