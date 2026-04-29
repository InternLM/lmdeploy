# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.nn.quant_utils import quant_blocked_fp8

from .base import split_size as _split_size
from .blocked_fp8 import FusedMoEBlockedF8

_FP4_TABLE = torch.tensor([
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
],
                          dtype=torch.float32)


def _v4_swiglu(intermediate: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """Match DeepSeek-V4 routed-expert activation semantics.

    Keep the activation hook in `nn/moe` so the V4 fused MoE wrapper does not depend on the legacy CUDA backend
    implementation file.
    """
    hidden = intermediate.size(-1) // 2
    gate = intermediate[..., :hidden].float()
    up = intermediate[..., hidden:].float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    return (torch.nn.functional.silu(gate) * up).to(intermediate.dtype)


def _get_v4_moe_runtime_kind(device: torch.device) -> str:
    """Select the routed-expert runtime path for the current GPU.

    Hopper/SM90 does not have a grouped FP8xFP4 expert kernel in the current DeepGEMM integration, so we convert
    checkpoint-native FP4 experts into blocked FP8 weights once at load time and run the fused blocked FP8 path.
    Blackwell/SM100 keeps the original grouped FP8xFP4 path.
    """
    if device.type == 'cuda' and torch.cuda.is_available():
        return 'fp4' if torch.cuda.get_device_capability(device)[0] >= 10 else 'fp8'
    return 'fp8'


def _dequantize_fp4_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize one checkpoint-native FP4 expert weight tensor to BF16.

    DeepSeek-V4 experts are stored as packed E2M1 values with per-row 1x32 scales. Hopper blocked fused MoE does not
    consume that native layout, so we first recover the dense BF16 matrix and then re-quantize with lmdeploy's blocked-
    FP8 scheme.
    """
    assert weight.dtype == torch.int8
    assert weight.ndim == 2
    out_dim, packed_in_dim = weight.shape
    in_dim = packed_in_dim * 2
    fp4_block_size = 32
    assert scale.shape == (out_dim, in_dim // fp4_block_size)

    fp4_table = _FP4_TABLE.to(weight.device)
    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    x = torch.stack([fp4_table[low.long()], fp4_table[high.long()]], dim=-1).flatten(1, 2)
    x = x * scale.float().repeat_interleave(fp4_block_size, dim=1)
    return x.to(torch.bfloat16)


def _convert_fp4_to_blocked_fp8(weight: torch.Tensor,
                                scale: torch.Tensor,
                                block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed FP4 expert weights into blocked FP8 expert weights.

    Important: lmdeploy's blocked fused MoE kernel expects the same weight/scale
    layout produced by `quant_blocked_fp8`, not DeepSeek's offline lossless
    E4M3 export format. We therefore:
    1. recover the exact BF16 weight matrix from checkpoint-native FP4
    2. quantize it again with lmdeploy's blocked FP8 recipe

    The conversion is done expert-by-expert to keep peak memory bounded during
    model loading on Hopper.
    """
    num_experts, out_features, packed_in_features = weight.shape
    in_features = packed_in_features * 2
    fp8_weight = torch.empty((num_experts, out_features, in_features),
                             dtype=torch.float8_e4m3fn,
                             device=weight.device)
    fp8_scale = torch.empty((num_experts, out_features // block_size, in_features // block_size),
                            dtype=torch.float32,
                            device=weight.device)

    for expert_id in range(num_experts):
        dense_weight = _dequantize_fp4_weight(weight[expert_id], scale[expert_id])
        expert_weight, expert_scale = quant_blocked_fp8(dense_weight,
                                                        torch.float8_e4m3fn,
                                                        block_size,
                                                        scale_fmt='ue8m0')
        fp8_weight[expert_id].copy_(expert_weight)
        fp8_scale[expert_id].copy_(expert_scale)
    return fp8_weight, fp8_scale


class V4ExpertWeights(nn.Module):
    """Local expert-sharded V4 expert weights.

    The checkpoint-native storage is packed FP4. `update_weights()` may replace the parameters with blocked FP8 tensors
    on Hopper, but the loading contract stays anchored on the original checkpoint layout so the SM100 FP4 path can be
    restored without rewriting the weight mapping again.
    """

    def __init__(self,
                 num_local_experts: int,
                 in_features: int,
                 out_features: int,
                 expert_offset: int,
                 weight_type: str,
                 device: torch.device):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.expert_offset = expert_offset
        self.weight_type = weight_type
        self.half_out = out_features // 2

        weight = torch.empty((num_local_experts, out_features, in_features // 2), dtype=torch.int8, device=device)
        scale = torch.empty((num_local_experts, out_features, in_features // 32),
                            dtype=torch.float8_e8m0fnu,
                            device=device)
        self.register_parameter('weight', nn.Parameter(weight, requires_grad=False))
        self.register_parameter('scale', nn.Parameter(scale, requires_grad=False))

        self.weight.weight_loader = self.weight_loader
        self.scale.weight_loader = self.scale_loader

    def _get_local_param(self, param: nn.Parameter, expert_id: int, shard_id: str):
        local_id = expert_id - self.expert_offset
        if local_id < 0 or local_id >= self.num_local_experts:
            return None
        if self.weight_type == 'gate_up':
            if shard_id == 'gate':
                return param.data[local_id, :self.half_out]
            if shard_id == 'up':
                return param.data[local_id, self.half_out:]
        elif self.weight_type == 'down' and shard_id == 'down':
            return param.data[local_id]
        raise RuntimeError(f'Unsupported shard_id={shard_id} for weight_type={self.weight_type}')

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        param_data = self._get_local_param(param, expert_id, shard_id)
        if param_data is None:
            return
        param_data.copy_(loaded_weight.to(param_data.device, dtype=param_data.dtype))

    def scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        param_data = self._get_local_param(param, expert_id, shard_id)
        if param_data is None:
            return
        param_data.copy_(loaded_weight.to(param_data.device, dtype=param_data.dtype))

    def update(self, weight: torch.Tensor, scale: torch.Tensor):
        weight_loader = self.weight.weight_loader
        scale_loader = self.scale.weight_loader
        weight = nn.Parameter(weight, requires_grad=False)
        scale = nn.Parameter(scale, requires_grad=False)
        weight.weight_loader = weight_loader
        scale.weight_loader = scale_loader
        self.register_parameter('weight', weight)
        self.register_parameter('scale', scale)


class ExpertParallelFusedMoEV4FP4(nn.Module):
    """TP-only fused MoE for DeepSeek-V4 routed experts.

    Public name is kept for backward compatibility. Runtime behavior is
    architecture-dependent:
    - SM90/Hopper: convert checkpoint FP4 experts to blocked FP8 and run the
      graph-friendly blocked FP8 fused MoE path.
    - SM100/Blackwell: keep the original grouped FP8xFP4 path.
    """

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 swiglu_limit: float = 0.0,
                 device: torch.device | None = None):
        super().__init__()
        device = device or torch.device('cpu')

        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        if dist_config.ep > 1:
            raise RuntimeError('DeepSeek-V4 fused MoE currently supports TP only; EP is not implemented.')

        tp, rank = get_tp_world_rank('moe')
        if num_experts % tp != 0:
            raise RuntimeError(f'Number of experts ({num_experts}) must be divisible by moe tp ({tp}).')

        self.tp = tp
        self.tp_rank = rank
        self.tp_group = dist_ctx.moe_tp_group.gpu_group
        self.num_experts = num_experts
        self.num_local_experts = num_experts // tp
        self.expert_offset = rank * self.num_local_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.top_k = top_k
        self.block_size = 128
        self.runtime_kind = _get_v4_moe_runtime_kind(device)

        self.gate_up = V4ExpertWeights(self.num_local_experts,
                                       hidden_dim,
                                       ffn_dim * 2,
                                       expert_offset=self.expert_offset,
                                       weight_type='gate_up',
                                       device=device)
        self.down = V4ExpertWeights(self.num_local_experts,
                                    ffn_dim,
                                    hidden_dim,
                                    expert_offset=self.expert_offset,
                                    weight_type='down',
                                    device=device)

        impl_builder = get_backend().get_layer_impl_builder(
            OpType.FusedMoEV4FP4 if self.runtime_kind == 'fp4' else OpType.FusedMoEV4BlockedF8)
        build_kwargs = dict(top_k=top_k,
                            num_experts=self.num_local_experts,
                            hidden_dim=hidden_dim,
                            ffn_dim=ffn_dim,
                            expert_offset=self.expert_offset,
                            swiglu_limit=swiglu_limit)
        if self.runtime_kind == 'fp8':
            build_kwargs['scale_fmt'] = 'ue8m0'
        self.impl = impl_builder.build(**build_kwargs)

    def update_weights(self):
        if self.runtime_kind == 'fp8':
            gate_up_weight, gate_up_scale = _convert_fp4_to_blocked_fp8(self.gate_up.weight,
                                                                        self.gate_up.scale,
                                                                        block_size=self.block_size)
            down_weight, down_scale = _convert_fp4_to_blocked_fp8(self.down.weight,
                                                                  self.down.scale,
                                                                  block_size=self.block_size)
        else:
            gate_up_weight, gate_up_scale, down_weight, down_scale = self.impl.update_weights(self.gate_up.weight,
                                                                                              self.gate_up.scale,
                                                                                              self.down.weight,
                                                                                              self.down.scale)
        self.gate_up.update(gate_up_weight, gate_up_scale)
        self.down.update(down_weight, down_scale)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        return self.impl.forward(hidden_states,
                                 topk_weights,
                                 topk_ids,
                                 self.gate_up.weight,
                                 self.gate_up.scale,
                                 self.down.weight,
                                 self.down.scale,
                                 group=self.tp_group if self.tp > 1 else None)


class V4ExpertTPWeights(nn.Module):
    """Checkpoint-native V4 expert weights for TP-sharded fused MoE.

    Unlike the older experimental implementation, these tensors keep all routed experts and shard only the expert FFN
    dimension across TP ranks. That matches lmdeploy's generic fused-MoE runtime and the vLLM integration strategy we
    are aligning to.
    """

    def __init__(self,
                 num_experts: int,
                 hidden_dim: int,
                 ffn_dim: int,
                 weight_type: str,
                 device: torch.device):
        super().__init__()
        tp, rank = get_tp_world_rank('moe')
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.weight_type = weight_type
        self.tp = tp
        self.rank = rank
        self.local_ffn_dim = _split_size(ffn_dim, tp, 128)[rank]
        self.ffn_split_sizes = _split_size(ffn_dim, tp, 128)

        if weight_type == 'gate_up':
            weight = torch.empty((num_experts, self.local_ffn_dim * 2, hidden_dim // 2),
                                  dtype=torch.int8, device=device)
            scale = torch.empty((num_experts, self.local_ffn_dim * 2, hidden_dim // 32),
                                dtype=torch.float8_e8m0fnu,
                                device=device)
        elif weight_type == 'down':
            weight = torch.empty((num_experts, hidden_dim, self.local_ffn_dim // 2),
                                 dtype=torch.int8, device=device)
            scale = torch.empty((num_experts, hidden_dim, self.local_ffn_dim // 32),
                                dtype=torch.float8_e8m0fnu,
                                device=device)
        else:
            raise RuntimeError(f'Unsupported V4 expert weight_type: {weight_type}')

        self.register_parameter('weight', nn.Parameter(weight, requires_grad=False))
        self.register_parameter('scale', nn.Parameter(scale, requires_grad=False))
        self.weight.weight_loader = self.weight_loader
        self.scale.weight_loader = self.scale_loader

    def _split_loaded_weight(self, loaded_weight: torch.Tensor, dim: int, sizes: list[int]):
        return loaded_weight.split(sizes, dim=dim)[self.rank].contiguous()

    def _chunk_loaded_weight(self, loaded_weight: torch.Tensor, shard_id: str, is_scale: bool):
        if self.tp == 1:
            return loaded_weight
        if self.weight_type == 'gate_up':
            assert shard_id in ('gate', 'up')
            return self._split_loaded_weight(loaded_weight, dim=0, sizes=self.ffn_split_sizes)
        assert shard_id == 'down'
        if is_scale:
            sizes = [size // 32 for size in self.ffn_split_sizes]
        else:
            sizes = [size // 2 for size in self.ffn_split_sizes]
        return self._split_loaded_weight(loaded_weight, dim=1, sizes=sizes)

    def _get_target_slice(self, param: nn.Parameter, expert_id: int, shard_id: str):
        if self.weight_type == 'gate_up':
            half_out = self.local_ffn_dim
            if shard_id == 'gate':
                return param.data[expert_id, :half_out]
            if shard_id == 'up':
                return param.data[expert_id, half_out:]
        elif self.weight_type == 'down' and shard_id == 'down':
            return param.data[expert_id]
        raise RuntimeError(f'Unsupported shard_id={shard_id} for weight_type={self.weight_type}')

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        loaded_weight = self._chunk_loaded_weight(loaded_weight, shard_id, is_scale=False)
        target = self._get_target_slice(param, expert_id, shard_id)
        target.copy_(loaded_weight.to(target.device, dtype=target.dtype))

    def scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        loaded_weight = self._chunk_loaded_weight(loaded_weight, shard_id, is_scale=True)
        target = self._get_target_slice(param, expert_id, shard_id)
        target.copy_(loaded_weight.to(target.device, dtype=target.dtype))


class FusedMoEV4(nn.Module):
    """DeepSeek-V4 routed experts on top of lmdeploy's generic fused MoE.

    Default path on Hopper/SM90 is blocked FP8 fused MoE with TP sharding on the expert intermediate dimension. The
    older expert-parallel FP4 runtime is kept only as an opt-in/reference path for future SM100 work.
    """

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 swiglu_limit: float = 0.0,
                 device: torch.device | None = None):
        super().__init__()
        device = device or torch.device('cpu')
        self.runtime_kind = _get_v4_moe_runtime_kind(device)
        self._impl_update_weights = None

        if self.runtime_kind == 'fp4':
            self.impl = ExpertParallelFusedMoEV4FP4(hidden_dim,
                                                    ffn_dim,
                                                    num_experts,
                                                    top_k,
                                                    swiglu_limit=swiglu_limit,
                                                    device=device)
            self.ckpt_gate_up = None
            self.ckpt_down = None
            self._take_update_weights_ownership()
            return

        self.impl = FusedMoEBlockedF8(hidden_dim=hidden_dim,
                                      ffn_dim=ffn_dim,
                                      num_experts=num_experts,
                                      top_k=top_k,
                                      renormalize=False,
                                      fp8_dtype=torch.float8_e4m3fn,
                                      scale_fmt='ue8m0',
                                      dtype=torch.bfloat16,
                                      device=device,
                                      all_reduce=True,
                                      act_func=lambda x: _v4_swiglu(x, swiglu_limit))
        self.ckpt_gate_up = V4ExpertTPWeights(num_experts, hidden_dim, ffn_dim, weight_type='gate_up', device=device)
        self.ckpt_down = V4ExpertTPWeights(num_experts, hidden_dim, ffn_dim, weight_type='down', device=device)
        self._take_update_weights_ownership()

    def _take_update_weights_ownership(self):
        """Prevent model_weight_loader from re-running nested fused-MoE update.

        `load_model_weights()` walks every named submodule and calls `update_weights()` if present. This wrapper owns
        the load-time V4 conversion and must invoke the inner fused-MoE update exactly once; letting the child module
        run again later would rebuild already-converted weights with the wrong contract.
        """
        self._impl_update_weights = self.impl.update_weights
        self.impl.update_weights = lambda: None

    def update_weights(self):
        if self.runtime_kind == 'fp4':
            self._impl_update_weights()
            return

        gate_up_weight, gate_up_scale = _convert_fp4_to_blocked_fp8(self.ckpt_gate_up.weight,
                                                                    self.ckpt_gate_up.scale,
                                                                    block_size=self.impl.block_size)
        down_weight, down_scale = _convert_fp4_to_blocked_fp8(self.ckpt_down.weight,
                                                              self.ckpt_down.scale,
                                                              block_size=self.impl.block_size)
        self.impl.gate_up.update_weight(gate_up_weight, gate_up_scale)
        self.impl.down.update_weight(down_weight, down_scale)
        self._impl_update_weights()

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        return self.impl(hidden_states, topk_weights, topk_ids)


FusedMoEV4FP4 = ExpertParallelFusedMoEV4FP4
