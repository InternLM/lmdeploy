# Copyright (c) OpenMMLab. All rights reserved.
"""Compressed-tensors W4A16 routed-expert weights and distributed runtime.

The checkpoint layout is kept intact: INT4 codes are packed along the logical
K dimension into int32 words and every 32 values share one BF16 scale.  This
module only takes exact TP or EP-local views; it never materializes persistent
BF16 expert weights.
"""

from itertools import product

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank
from lmdeploy.pytorch.models.patch import get_build_model_context

from .base import FusedMoEBase, MoeType, moe_gather_inputs, moe_reduce, update_dims

_PART_DTYPES = {
    'weight_packed': torch.int32,
    'weight_scale': torch.bfloat16,
    'weight_shape': torch.int32,
}


class CompressedTensorsMoEWeights(nn.Module):
    """Rank-local checkpoint-layout parameters for one fused MoE projection."""

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        ffn_dim: int,
        weight_type: str,
        num_bits: int,
        group_size: int,
        device: torch.device,
    ):
        super().__init__()
        if weight_type not in {'gate_up', 'down'}:
            raise ValueError(
                f'Unknown compressed-tensors MoE weight type: {weight_type}')
        if num_bits != 4 or group_size != 32:
            raise ValueError(
                f'Only compressed-tensors INT4 group-size 32 is supported, got bits={num_bits}, '
                f'group_size={group_size}')
        if hidden_dim <= 0 or ffn_dim <= 0 or num_experts <= 0:
            raise ValueError(
                'MoE dimensions and number of experts must be positive')

        tp_world, tp_rank = get_tp_world_rank('moe')
        ep_world, ep_rank = get_ep_world_rank()
        if ep_world <= 0 or not 0 <= ep_rank < ep_world:
            raise ValueError(
                f'Invalid EP world/rank pair: world={ep_world}, rank={ep_rank}')
        if num_experts % ep_world != 0:
            raise ValueError(
                f'num_experts={num_experts} is not divisible by EP={ep_world}')
        num_local_experts = num_experts // ep_world
        first_local_expert = ep_rank * num_local_experts
        expert_list = list(
            range(first_local_expert,
                  first_local_expert + num_local_experts))

        if ffn_dim % tp_world != 0:
            raise ValueError(
                f'ffn_dim={ffn_dim} is not divisible by MoE TP={tp_world}')
        local_ffn_dim = ffn_dim // tp_world
        pack_factor = 32 // num_bits
        if hidden_dim % group_size != 0:
            raise ValueError(
                f'hidden_dim={hidden_dim} must preserve group_size={group_size} boundaries'
            )
        if local_ffn_dim % group_size != 0:
            raise ValueError(
                f'TP-local ffn_dim={local_ffn_dim} must preserve group_size={group_size} boundaries'
            )
        if hidden_dim % pack_factor != 0 or local_ffn_dim % pack_factor != 0:
            raise ValueError(
                f'MoE dimensions must preserve INT4 pack-factor={pack_factor} boundaries'
            )

        # Keep ``num_experts`` as the global router domain for compatibility.
        # Parameters, loader destinations and completeness are rank-local.
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.expert_list = expert_list
        self.expert_map = {
            global_expert_id: local_slot
            for local_slot, global_expert_id in enumerate(expert_list)
        }
        self.hidden_dim = hidden_dim
        self.global_ffn_dim = ffn_dim
        self.local_ffn_dim = local_ffn_dim
        self.weight_type = weight_type
        self.num_bits = num_bits
        self.group_size = group_size
        self.pack_factor = pack_factor
        self.tp_world = tp_world
        self.tp_rank = tp_rank
        self.ep_world = ep_world
        self.ep_rank = ep_rank
        self._loaded_parts: set[tuple[int, str, str]] = set()

        if weight_type == 'gate_up':
            packed_shape = (num_local_experts, 2 * local_ffn_dim,
                            hidden_dim // pack_factor)
            scale_shape = (num_local_experts, 2 * local_ffn_dim,
                           hidden_dim // group_size)
            logical_shape = (num_local_experts, 2, 2)
        else:
            packed_shape = (num_local_experts, hidden_dim,
                            local_ffn_dim // pack_factor)
            scale_shape = (num_local_experts, hidden_dim,
                           local_ffn_dim // group_size)
            logical_shape = (num_local_experts, 2)

        self._register_checkpoint_parameter('weight_packed', packed_shape,
                                            torch.int32, device)
        self._register_checkpoint_parameter('weight_scale', scale_shape,
                                            torch.bfloat16, device)
        self._register_checkpoint_parameter('weight_shape',
                                            logical_shape,
                                            torch.int32,
                                            device,
                                            fill_value=-1)

    def _register_checkpoint_parameter(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        fill_value: int | None = None,
    ):
        data = torch.empty(shape, dtype=dtype, device=device)
        if fill_value is not None:
            data.fill_(fill_value)
        parameter = nn.Parameter(data, requires_grad=False)
        parameter.weight_loader = self.weight_loader
        parameter._weight_type = name
        self.register_parameter(name, parameter)

    def _expected_shards(self) -> tuple[str, ...]:
        if self.weight_type == 'gate_up':
            return ('gate', 'up')
        return ('down', )

    def _validate_loader_key(self, expert_id: int, shard_id: str, part: str):
        if not isinstance(expert_id, int) or isinstance(
                expert_id, bool) or not 0 <= expert_id < self.num_experts:
            raise ValueError(
                f'expert_id must be in [0, {self.num_experts}), got {expert_id!r}'
            )
        if shard_id not in self._expected_shards():
            raise ValueError(
                f'{self.weight_type} cannot load projection shard {shard_id!r}'
            )
        if part not in _PART_DTYPES:
            raise ValueError(
                f'Unknown compressed-tensors checkpoint part: {part!r}')

    def _expected_full_shape(self, shard_id: str,
                             part: str) -> tuple[int, ...]:
        if shard_id in {'gate', 'up'}:
            logical = (self.global_ffn_dim, self.hidden_dim)
        else:
            logical = (self.hidden_dim, self.global_ffn_dim)
        if part == 'weight_packed':
            return logical[0], logical[1] // self.pack_factor
        if part == 'weight_scale':
            return logical[0], logical[1] // self.group_size
        return (2, )

    def _local_logical_shape(self, shard_id: str) -> tuple[int, int]:
        if shard_id in {'gate', 'up'}:
            return self.local_ffn_dim, self.hidden_dim
        return self.hidden_dim, self.local_ffn_dim

    def _destination(self, param: nn.Parameter, local_expert_id: int,
                     shard_id: str, part: str):
        if self.weight_type == 'gate_up':
            if part == 'weight_shape':
                shape_slot = 0 if shard_id == 'gate' else 1
                return param.data[local_expert_id, shape_slot]
            shard_offset = 0 if shard_id == 'gate' else self.local_ffn_dim
            return param.data[local_expert_id,
                              shard_offset:shard_offset + self.local_ffn_dim]
        return param.data[local_expert_id]

    def _take_tp_shard(self, loaded_weight: torch.Tensor, shard_id: str,
                       part: str):
        if part == 'weight_shape':
            return loaded_weight.new_tensor(
                self._local_logical_shape(shard_id))
        if shard_id in {'gate', 'up'}:
            start = self.tp_rank * self.local_ffn_dim
            return loaded_weight.narrow(0, start, self.local_ffn_dim)

        divisor = self.pack_factor if part == 'weight_packed' else self.group_size
        local_width = self.local_ffn_dim // divisor
        start = self.tp_rank * local_width
        return loaded_weight.narrow(1, start, local_width)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ):
        """Validate and copy one checkpoint tensor owned by this rank."""
        part = getattr(param, '_weight_type', None)
        self._validate_loader_key(expert_id, shard_id, part)
        local_expert_id = self.expert_map.get(expert_id)
        if local_expert_id is None:
            return

        if loaded_weight.dtype != _PART_DTYPES[part]:
            raise ValueError(
                f'{shard_id}.{part} must have dtype {_PART_DTYPES[part]}, got {loaded_weight.dtype}'
            )

        expected_shape = self._expected_full_shape(shard_id, part)
        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f'{shard_id}.{part} shape mismatch: expected {expected_shape}, got {tuple(loaded_weight.shape)}'
            )
        if part == 'weight_shape':
            expected_logical = self._expected_full_shape(
                shard_id, 'weight_packed')
            expected_logical = (
                expected_logical[0],
                expected_logical[1] * self.pack_factor,
            )
            observed_logical = tuple(
                int(dim) for dim in loaded_weight.tolist())
            if observed_logical != expected_logical:
                raise ValueError(
                    f'{shard_id}.weight_shape value mismatch: expected {expected_logical}, '
                    f'got {observed_logical}')

        key = (expert_id, shard_id, part)
        if key in self._loaded_parts:
            raise RuntimeError(
                f'Duplicate compressed-tensors checkpoint tensor for expert={expert_id}, '
                f'shard={shard_id}, part={part}')

        local_weight = self._take_tp_shard(loaded_weight, shard_id, part)
        destination = self._destination(param, local_expert_id, shard_id, part)
        if tuple(local_weight.shape) != tuple(destination.shape):
            raise RuntimeError(
                f'Internal TP shard shape mismatch: source={tuple(local_weight.shape)}, '
                f'destination={tuple(destination.shape)}')
        destination.copy_(local_weight)
        self._loaded_parts.add(key)

    def validate_complete(self):
        """Reject inference unless every local projection triplet was
        loaded."""
        expected = set(
            product(self.expert_list, self._expected_shards(), _PART_DTYPES))
        missing = sorted(expected - self._loaded_parts)
        unexpected = sorted(self._loaded_parts - expected)
        if missing or unexpected:
            missing_preview = ', '.join(map(str, missing[:8]))
            unexpected_preview = ', '.join(map(str, unexpected[:8]))
            raise RuntimeError(
                f'Incomplete compressed-tensors {self.weight_type} weights: missing={len(missing)} '
                f'[{missing_preview}], unexpected={len(unexpected)} [{unexpected_preview}]'
            )


class FusedMoEW4A16(FusedMoEBase):
    """TP or DP-attention/DeepEP MoE over packed INT4 weights."""

    # The Kimi reference combines router-weighted expert outputs in FP32.
    # Preserve that accumulator through the single outer TP reduction.
    tp_reduce_dtype = torch.float32

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        renormalize: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        all_reduce: bool = True,
        num_bits: int = 4,
        group_size: int = 32,
        layer_idx: int = 0,
    ):
        device = device or torch.device('cpu')
        dtype = dtype or torch.bfloat16
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f'Compressed-tensors W4A16 requires a 16-bit floating dtype, got {dtype}'
            )

        self.init_dist_args(all_reduce)
        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        if dist_config.dp > 1:
            if dist_config.attn_tp != 1:
                raise RuntimeError(
                    'Compressed-tensors W4A16 DP attention currently requires '
                    f'attn_tp=1, got attn_tp={dist_config.attn_tp}.')
            if self.ep <= 1:
                raise RuntimeError(
                    'Compressed-tensors W4A16 DP attention requires expert '
                    f'parallelism, got ep={self.ep}.')
            expected_world_size = dist_config.world_size
            if (dist_config.dp != expected_world_size
                    or dist_config.ep != expected_world_size
                    or dist_config.mlp_tp != expected_world_size
                    or dist_config.moe_tp != 1):
                raise RuntimeError(
                    'Compressed-tensors W4A16 DP attention currently requires '
                    'dp=ep=mlp_tp=world_size and moe_tp=1, got '
                    f'dp={dist_config.dp}, ep={dist_config.ep}, '
                    f'mlp_tp={dist_config.mlp_tp}, '
                    f'moe_tp={dist_config.moe_tp}, '
                    f'world_size={expected_world_size}.')
        if dist_config.enable_eplb:
            raise RuntimeError(
                'Compressed-tensors W4A16 does not support EPLB.')
        if dist_config.enable_microbatch:
            raise RuntimeError(
                'Compressed-tensors W4A16 does not support microbatch execution.'
            )
        if self.tp_mode != TPMode.DEFAULT:
            raise RuntimeError(
                'Compressed-tensors W4A16 only supports eager default TP mode.'
            )

        super().__init__(tp=self.tp,
                         tp_mode=self.tp_mode,
                         do_renormalize=renormalize)
        impl_builder = get_backend().get_layer_impl_builder(
            OpType.FusedMoEW4A16)
        deep_ep_max_tokens_per_rank = (
            get_build_model_context().deep_ep_max_tokens_per_rank)
        self.impl = impl_builder.build(
            top_k=top_k,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            ep_size=self.ep,
            ep_group=dist_ctx.ep_gpu_group,
            renormalize=renormalize,
            num_bits=num_bits,
            group_size=group_size,
            out_dtype=dtype,
            num_max_dispatch_tokens_per_rank=
            deep_ep_max_tokens_per_rank,
            layer_idx=layer_idx,
        )

        global_ffn_dim = ffn_dim
        hidden_dim, local_ffn_dim = update_dims(hidden_dim, ffn_dim)
        self.gate_up = CompressedTensorsMoEWeights(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            ffn_dim=global_ffn_dim,
            weight_type='gate_up',
            num_bits=num_bits,
            group_size=group_size,
            device=device,
        )
        self.expert_list = (
            self.gate_up.expert_list if self.ep > 1 else None)
        self.down = CompressedTensorsMoEWeights(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            ffn_dim=global_ffn_dim,
            weight_type='down',
            num_bits=num_bits,
            group_size=group_size,
            device=device,
        )
        self.hidden_dim = hidden_dim
        self.ffn_dim = local_ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device

    def update_weights(self):
        """Finish loading only after every packed expert triplet is present."""
        self.gate_up.validate_complete()
        self.down.validate_complete()

    def dispatch(self, state: dict):
        """Gather eager default-TP inputs."""
        moe_type = state['moe_type']
        if moe_type != MoeType.Default:
            raise NotImplementedError(
                f'Compressed-tensors W4A16 does not support MoE mode: {moe_type}'
            )
        hidden_states, topk_weights, topk_idx = moe_gather_inputs(
            state['hidden_states'],
            state['topk_weights'],
            state['topk_idx'],
            group=self.gather_group,
        )
        return {
            'hidden_states': hidden_states,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'moe_type': moe_type,
        }

    def gemm(self, state: dict):
        """Run the direct-packed eager implementation."""
        output = self.impl.forward(
            state['hidden_states'],
            state['topk_weights'],
            state['topk_idx'],
            self.gate_up.weight_packed,
            self.gate_up.weight_scale,
            self.down.weight_packed,
            self.down.weight_scale,
        )
        return {'hidden_states': output, 'moe_type': state['moe_type']}

    def combine(self, state: dict):
        """Apply the existing MoE TP reduction contract exactly once."""
        moe_type = state['moe_type']
        if moe_type != MoeType.Default:
            raise NotImplementedError(
                f'Compressed-tensors W4A16 does not support MoE mode: {moe_type}'
            )
        if self.all_reduce:
            state['hidden_states'] = moe_reduce(
                state['hidden_states'],
                rank=self.tp_rank,
                tp_mode=self.tp_mode,
                group=self.tp_group,
            )
        return {'hidden_states': state['hidden_states'], 'moe_type': moe_type}

    def wait(self, state: dict):
        """Async execution is outside the synchronous eager path."""
        raise NotImplementedError(
            'Compressed-tensors W4A16 only supports synchronous eager execution.'
        )
