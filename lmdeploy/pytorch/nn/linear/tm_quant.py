# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Float-quantized linear layers backed by the TurboMind GEMM channel.

Two lanes served by one implementation family:

- ``nvfp4``: compressed-tensors ``nvfp4-pack-quantized`` checkpoints. The
  module registers the checkpoint suffixes as parameter names
  (``weight_packed``/``weight_scale``/``weight_global_scale``), so raw
  checkpoint keys load without any stream-level renaming. Compute dtype is
  fp16; inputs are converted like the W4A16 route.
- ``ct_fp8``: compressed-tensors ``float-quantized`` FP8 checkpoints
  (per-channel ``weight_scale`` ``[N, 1]`` or 128-blocked). Compute dtype
  is bf16; the kernel quantizes activations dynamically.

Merged projections (gate/up, QKV) keep one packed handle per shard:
NVFP4 carries a per-tensor global scale per checkpoint module, so shard
outputs are computed separately and concatenated before the module-level
activation. The torch parameter copies stay resident (no spill): one
extra packed-weight copy is bounded and keeps hot updates and
sleep/wakeup trivial.
"""

from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.tm_quant_modules import LinearTmQuantBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader

from ..utils import chunk_aligned, get_distribute_size
from .base import LinearBase
from .utils import QKVMixin

_LANE_COMPUTE_DTYPE = {'nvfp4': torch.float16, 'ct_fp8': torch.bfloat16}
_FP8_DTYPE = torch.float8_e4m3fn


class TmQuantLinearMixin:
    """Shared construction/pack plumbing for the TurboMind float lanes."""

    lane: str

    @property
    def _fp8_block_out(self):
        """ct_fp8 scale layout: 1 = per-channel, 128 = blocked."""
        return 128 if getattr(self, 'fp8_blocked', False) else 1

    def _create_lane_weights(self, in_features: int, out_features: int, bias: bool, device: torch.device):
        """Allocate the checkpoint-layout parameters of one lane."""
        compute_dtype = _LANE_COMPUTE_DTYPE[self.lane]
        params: dict[str, torch.Tensor | None] = {}
        if self.lane == 'nvfp4':
            assert in_features % 16 == 0, f'NVFP4 requires in_features % 16 == 0, got {in_features}'
            assert out_features % 2 == 0, f'NVFP4 requires out_features % 2 == 0, got {out_features}'
            params['weight_packed'] = torch.empty((out_features, in_features // 2), dtype=torch.uint8,
                                                  device=device)
            params['weight_scale'] = torch.empty((out_features, in_features // 16), dtype=_FP8_DTYPE,
                                                 device=device)
            params['weight_global_scale'] = torch.empty((self.num_shards, ), dtype=torch.float32, device=device)
        else:
            assert in_features % 128 == 0, f'CT FP8 requires in_features % 128 == 0, got {in_features}'
            assert out_features % 2 == 0, f'CT FP8 requires out_features % 2 == 0, got {out_features}'
            params['weight'] = torch.empty((out_features, in_features), dtype=_FP8_DTYPE, device=device)
            if self._fp8_block_out == 1:
                # Per-channel checkpoint scales: [N, 1] (FP8Format(block_out=1)
                # expands them to the kernel's [K//128, N] layout in
                # finalize).
                scale_shape = (out_features, 1)
            else:
                # 128-blocked checkpoint scales follow the format's accepts
                # convention: [ceil(N/128), ceil(K/128)] (one scale per
                # output block per input group).
                scale_shape = ((out_features + 127) // 128, (in_features + 127) // 128)
            params['weight_scale'] = torch.empty(scale_shape, dtype=compute_dtype, device=device)
        params['bias'] = torch.empty((out_features, ), dtype=compute_dtype, device=device) if bias else None
        return params

    def _register_lane_parameters(self, params: dict[str, torch.Tensor | None]):
        """Register parameters under their checkpoint suffix names."""
        for name, tensor in params.items():
            if tensor is None:
                self.register_parameter(name, None)
                continue
            tensor = nn.Parameter(tensor, requires_grad=False)
            self.register_parameter(name, tensor)
            if name == 'weight_global_scale':
                # Per-shard per-tensor scales load by shard id.
                tensor.weight_loader = self.global_scale_loader
                tensor._weight_type = name
            else:
                tensor.weight_loader = self.weight_loader
                tensor._weight_type = name

    def _update_all_out_features(self, all_out_features: list[int]):
        """TP-distribute each fused section, keeping kernel N alignment."""
        world_size, rank = self.get_tp_world_rank()
        align = 8  # the channel's dense families align N to 8 on both lanes
        return [get_distribute_size(out_feat, world_size, rank, align) for out_feat in all_out_features]

    def _shard_slices(self):
        """Row ranges of each shard inside the fused parameters."""
        ends = []
        total = 0
        for size in self.shard_sections:
            total += size
            ends.append(total)
        starts = [0] + ends[:-1]
        return list(zip(starts, ends, strict=True))

    def _lane_shards(self):
        """Per-shard views of the fused parameters for packing."""
        slices = self._shard_slices()
        if self.lane == 'nvfp4':
            weight_key, scale_key = 'weight_packed', 'weight_scale'
            gs = getattr(self, 'weight_global_scale')
            global_scales = [gs[i:i + 1] for i in range(self.num_shards)]
            scale_slices = slices
        else:
            weight_key, scale_key = 'weight', 'weight_scale'
            global_scales = None
            if self._fp8_block_out == 1:
                scale_slices = slices
            else:
                # Blocked scales are [ceil(N/128), ceil(K/128)]: one row per
                # output block, so shard boundaries move in 128 blocks.
                scale_slices = [((start + 127) // 128, (end + 127) // 128) for start, end in slices]
        weights = [getattr(self, weight_key)[start:end] for start, end in slices]
        scales = [getattr(self, scale_key)[start:end] for start, end in scale_slices]
        biases = ([getattr(self, 'bias')[start:end] for start, end in slices]
                  if getattr(self, 'bias', None) is not None else None)
        return weights, scales, global_scales, biases

    def _build_impl(self, device: torch.device):
        """Build the TurboMind float-lane operator.

        No native fallback
        exists: a declined route raises with the routing reason.
        """
        device = torch.device(device) if not isinstance(device, torch.device) else device
        device_guard = (torch.cuda.device(device) if device is not None and device.type == 'cuda' else None)
        with device_guard:
            return get_backend().build_op(
                LinearTmQuantBuildSpec(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    lane=self.lane,
                    bias=getattr(self, 'bias_flag', False),
                    fp8_block_out=self._fp8_block_out if self.lane == 'ct_fp8' else None,
                    prefix=getattr(self, 'tm_prefix', ''),
                ),
                enable_deterministic=get_build_model_context().enable_deterministic,
            )

    def update_weights(self):
        """Pack the checkpoint-layout parameters into the TurboMind channel."""
        weights, scales, global_scales, biases = self._lane_shards()
        self.impl.update_weights(weights, scales, global_scales, biases)

    def sleep_release(self):
        """Release the native packed handles ahead of ``Module.to``."""
        self.impl._release()

    def sleep_wakeup(self):
        """Re-pack the torch weights after wakeup."""
        self.update_weights()

    def _forward_default(self, x, all_reduce, tp_sizes):
        """Default forward implement."""
        biases = None
        bias = getattr(self, 'bias', None)
        if bias is not None:
            biases = [bias[s:e] for s, e in self._shard_slices()]
        return self.impl.forward(
            x,
            biases=biases,
            all_reduce=all_reduce,
            group=self.tp_group,
            rank=self.tp_rank,
            scatter_size=tp_sizes,
        )


class TmQuantLinear(TmQuantLinearMixin, LinearBase):
    """Single-shard float-quantized linear served by the TurboMind channel."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lane: str,
        bias: bool,
        fp8_blocked: bool = False,
        device: torch.device | None = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        layer_type: str = 'attn',
        prefix: str = '',
    ):
        self.lane = lane
        self.fp8_blocked = fp8_blocked
        self.tm_prefix = prefix
        self.bias_flag = bias
        if fp8_blocked and is_tp:
            # Builders coerce is_tp down before construction, so a module
            # asked for TP sharding genuinely has tp > 1.
            raise NotImplementedError('Blocked CT FP8 TP sharding is not implemented; '
                                      'serve blocked float-quantized checkpoints with tp=1.')
        self.init_tp_args(is_tp, all_reduce, colwise, layer_type)
        self.num_shards = getattr(self, 'num_shards', 1)
        if self.is_tp:
            world_size, rank = self.get_tp_world_rank()
            if colwise:
                out_features = get_distribute_size(out_features, world_size, rank, align=8)
            else:
                in_features = get_distribute_size(in_features,
                                                  world_size,
                                                  rank,
                                                  align=16 if lane == 'nvfp4' else 128)
        self.in_features = in_features
        self.out_features = out_features
        self.shard_sections = [out_features]
        params = self._create_lane_weights(in_features, out_features, bias, device or torch.device('cpu'))
        LinearBase.__init__(
            self,
            dtype=_LANE_COMPUTE_DTYPE[lane],
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
            layer_type=layer_type,
        )
        self.impl = self._build_impl(device)
        self._register_lane_parameters(params)

    def global_scale_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Plain variants keep one global scale per rank, whole."""
        default_weight_loader(param, loaded_weight.reshape(1))

    def setup_loaders(self):
        """Rebind the parameter loaders to the module's current loaders.

        Model patches that override ``weight_loader`` (e.g. non-uniform QKV
        TP splits) re-run this so the parameters pick up the patched
        loader.
        """
        for name, param in self.named_parameters(recurse=False):
            if param is None:
                continue
            param.weight_loader = self.global_scale_loader if name == 'weight_global_scale' else self.weight_loader

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader for the plain variants."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = self.get_tp_world_rank()
        weight_type = param._weight_type
        if self.colwise:
            if weight_type == 'bias':
                weight = chunk_aligned(loaded_weight, world_size, 0, 8)[rank]
                return default_weight_loader(param, weight)
            # weight / packed weight / scale rows split by output channels
            weight = chunk_aligned(loaded_weight, world_size, 0, 1)[rank]
            return default_weight_loader(param, weight)

        # rowwise: the input dim splits, per-channel scales stay whole.
        if weight_type == 'bias':
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)
        if weight_type == 'weight_scale' and loaded_weight.size(-1) == 1:
            return default_weight_loader(param, loaded_weight)
        if weight_type == 'weight_packed':
            # packed rows: 8 bytes hold one 16-code group
            weight = chunk_aligned(loaded_weight, world_size, -1, 8)[rank]
            return default_weight_loader(param, weight)
        if weight_type == 'weight_scale':
            weight = chunk_aligned(loaded_weight, world_size, -1, 1)[rank]
            return default_weight_loader(param, weight)
        if weight_type == 'weight':
            # fp8 weight: [N, K], split K with the kernel's 128 alignment
            weight = chunk_aligned(loaded_weight, world_size, -1, 128)[rank]
            return default_weight_loader(param, weight)
        return default_weight_loader(param, loaded_weight)


class MergedTmQuantLinear(TmQuantLinear):
    """Fused-output float-quantized linear (gate/up style).

    Checkpoint modules keep per-shard global scales; the packed channel holds one handle per shard and forward
    concatenates the outputs.
    """

    def __init__(
        self,
        in_features: int,
        all_out_features: list[int],
        lane: str,
        bias: bool,
        fp8_blocked: bool = False,
        device: torch.device | None = None,
        is_tp: bool = True,
        out_names: list[Any] | None = None,
        layer_type: str = 'attn',
        prefix: str = '',
    ):
        self.init_tp_args(is_tp, all_reduce=False, colwise=True, layer_type=layer_type)
        if out_names is None:
            out_names = torch.arange(len(all_out_features)).tolist()
        self.out_names_map = dict((name, idx) for idx, name in enumerate(out_names))
        self.num_shards = len(all_out_features)
        self.all_out_features = self._update_all_out_features(all_out_features)
        out_features = sum(self.all_out_features)
        super().__init__(
            in_features,
            out_features,
            lane,
            bias,
            fp8_blocked=fp8_blocked,
            device=device,
            colwise=True,
            is_tp=is_tp,
            layer_type=layer_type,
            prefix=prefix,
        )
        self.shard_sections = list(self.all_out_features)

    def global_scale_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any = 0):
        """Load one shard's per-tensor global scale."""
        idx = self.out_names_map[shard_id]
        default_weight_loader(param[idx:idx + 1], loaded_weight.reshape(1))

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any = 0):
        """Weight loader splitting the fused parameter rows by shard."""
        idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[idx]
        if param._weight_type == 'bias':
            weight = chunk_aligned(loaded_weight, self.tp, 0, 8)[self.tp_rank]
        else:
            weight = chunk_aligned(loaded_weight, self.tp, 0, 1)[self.tp_rank]
        param_w.copy_(weight)


class QKVTmQuantLinear(MergedTmQuantLinear, QKVMixin):
    """QKV float-quantized linear served by the TurboMind channel."""

    def __init__(
        self,
        in_features: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        head_size_v: int,
        lane: str,
        bias: bool = False,
        fp8_blocked: bool = False,
        device: torch.device | None = None,
        is_tp: bool = True,
        num_replicate_kv_heads: int = 1,
        prefix: str = '',
    ):
        self.init_tp_args(is_tp, all_reduce=False, colwise=True, layer_type='attn')
        QKVMixin.__init__(
            self,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            num_replicate_kv_heads=num_replicate_kv_heads,
            is_tp=is_tp,
            tp=self.tp,
            tp_rank=self.tp_rank,
        )
        super().__init__(
            in_features,
            self.get_qkv_out_feautures(),
            lane,
            bias,
            fp8_blocked=fp8_blocked,
            device=device,
            is_tp=is_tp,
            out_names=('q', 'k', 'v'),
            layer_type='attn',
            prefix=prefix,
        )

    def _update_all_out_features(self, all_out_features: list[int]):
        """Head-derived sections distribute exactly; alignment is inherited
        from the head geometry."""
        return all_out_features

    def global_scale_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any = 'q'):
        """Load one shard's per-tensor global scale."""
        idx = self.out_names_map[shard_id]
        default_weight_loader(param[idx:idx + 1], loaded_weight.reshape(1))

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any = 'q'):
        """Weight loader with non-replicated k/v head chunking."""
        world_size, rank = self.get_tp_world_rank()
        chunk_size, chunk_idx = world_size, rank
        idx = self.out_names_map[shard_id]

        if self.num_replicate_kv_heads > 1 and shard_id in ('k', 'v'):
            chunk_size = world_size // self.num_replicate_kv_heads
            chunk_idx = rank // self.num_replicate_kv_heads

        param_w = param.data.split(self.all_out_features, 0)[idx]
        if param._weight_type == 'bias':
            weight = chunk_aligned(loaded_weight, chunk_size, 0, 8)[chunk_idx]
        else:
            weight = chunk_aligned(loaded_weight, chunk_size, 0, 1)[chunk_idx]
        param_w.copy_(weight)
