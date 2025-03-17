# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank, get_ep_world_rank
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from ..backends import OpType, get_backend
from .utils import div_up


class SoftmaxTopK(nn.Module):
    """softmax topk."""

    def __init__(self, top_k: int, dim: int = -1):
        super().__init__()
        self.top_k = top_k
        impl_builder = get_backend().get_layer_impl_builder(OpType.SoftmaxTopK)
        self.impl = impl_builder.build(top_k, dim)

    def forward(self, x: torch.Tensor):
        """forward."""
        return self.impl.forward(x)


def create_mlp_weights(hidden_dim: int, ffn_dim: int, num_experts: int, dtype: torch.dtype, device: torch.device):
    """create weights."""
    gate_up_weights = torch.empty((num_experts, ffn_dim * 2, hidden_dim), dtype=dtype, device=device)
    down_weights = torch.empty((num_experts, hidden_dim, ffn_dim), dtype=dtype, device=device)
    return gate_up_weights, down_weights


def _update_args(hidden_dim: int, ffn_dim: int):
    """update args."""
    world_size, _ = get_tp_world_rank()
    assert ffn_dim % world_size == 0
    ffn_dim = ffn_dim // world_size
    return hidden_dim, ffn_dim


class LinearWeights(nn.Module):
    """fused moe linear weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 dtype: torch.dtype,
                 device: torch.device,
                 expert_list: List[int] = None,
                 ep: bool = False):
        super().__init__()
        weight = torch.empty((num_experts, out_features, in_features), dtype=dtype, device=device)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_parameter('weight', weight)
        self.ep = ep
        self.expert_list = expert_list
        self.weight_type = weight_type
        self.half_out = out_features // 2

        if False: # zcx self.ep:
            self.expert_map = dict((eid, idx) for idx, eid in enumerate(expert_list))
            self.weight.weight_loader = self.weight_loader_ep
        else:
            self.weight.weight_loader = self.weight_loader_tp

    def update_weight(self, weight: torch.Tensor):
        """update weight."""
        weight_loader = self.weight.weight_loader
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight.weight_loader = weight_loader
        self.register_parameter('weight', weight)

    def weight_loader_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        # zcx begin
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return

        expert_map = dict((eid, idx) for idx, eid in enumerate(expert_list))
        expert_id = expert_map[expert_id]
        # zcx end
        """weight loader."""
        world_size, rank = get_tp_world_rank()
        if shard_id == 'gate':
            param_data = param.data[expert_id, :self.half_out]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, self.half_out:]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            # weight is not contiguous, chunk and copy in cpu is slow
            weight = loaded_weight.to(param_data.device)
            weight = weight.chunk(world_size, dim=1)[rank]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)

    def weight_loader_ep(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """weight loader."""
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return

        expert_map = self.expert_map
        param_id = expert_map[expert_id]
        if shard_id == 'gate':
            param_data = param.data[param_id, :self.half_out]
        elif shard_id == 'up':
            param_data = param.data[param_id, self.half_out:]
        elif shard_id == 'down':
            param_data = param.data[param_id]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(loaded_weight)

   


def _gather_input(x: torch.Tensor, tp_sizes: List[int]):
    """gather input."""
    shape0 = x.shape[:-2]
    shape1 = x.shape[-1:]
    shapes = [shape0 + (size, ) + shape1 for size in tp_sizes]
    new_x = [x.new_empty(shape) for shape in shapes]
    dist.all_gather(new_x, x)
    x = torch.cat(new_x, dim=-2)
    return x


def _reduce_scatter_input(out: torch.Tensor, tp_sizes: List[int]):
    """reduce scatter."""
    _, rank = get_tp_world_rank()
    out = out.transpose(0, -2)
    if not out.is_contiguous():
        out = out.contiguous()
    outs = out.split(tp_sizes, 0)
    outs = list(outs)
    out = outs[rank]
    dist.reduce_scatter(out, outs)
    out = out.transpose(0, -2)
    return out


def _moe_gather_inputs(hidden_states, topk_weights, topk_ids, enable_ep):
    dist_ctx = get_dist_manager().current_context()
    dp = dist_ctx.dp
    if dp <= 1:
        return hidden_states, topk_weights, topk_ids

    step_ctx = get_step_ctx_manager().current_context()
    dp_meta = step_ctx.dp_meta
    if not enable_ep:
        if dist_ctx.tp == 1:
            return hidden_states, topk_weights, topk_ids
        tp_sizes = dp_meta.tp_sizes
        hidden_states = _gather_input(hidden_states, tp_sizes)
        topk_weights = _gather_input(topk_weights, tp_sizes)
        topk_ids = _gather_input(topk_ids, tp_sizes)
    else:
        raise RuntimeError('Not supported.')
    return hidden_states, topk_weights, topk_ids


def _moe_reduce(ret, enable_ep):
    dist_ctx = get_dist_manager().current_context()
    dp = dist_ctx.dp
    if dp > 1:
        step_ctx = get_step_ctx_manager().current_context()
        dp_meta = step_ctx.dp_meta
        if not enable_ep:
            if dist_ctx.tp == 1:
                return ret
            tp_sizes = dp_meta.tp_sizes
            ret = _reduce_scatter_input(ret, tp_sizes)
        else:
            raise RuntimeError('Not supported.')
    else:
        dist.all_reduce(ret)
    return ret


class FusedMoE(nn.Module):
    """fused moe."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 renormalize: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 enable_ep: bool = False):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16

        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoE)
        self.impl = impl_builder.build(top_k, num_experts, renormalize)

        enable_ep = enable_ep and self.impl.support_ep()
        if enable_ep:
            world_size, rank = get_tp_world_rank()
            expert_list = self.impl.ep_expert_list(world_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = _update_args(hidden_dim, ffn_dim)
            expert_list = None
        self.expert_list = expert_list
        self.gate_up = LinearWeights(num_experts,
                                     hidden_dim,
                                     ffn_dim * 2,
                                     weight_type='gate_up',
                                     dtype=dtype,
                                     device=device,
                                     expert_list=expert_list,
                                     ep=enable_ep)
        self.down = LinearWeights(
            num_experts,
            ffn_dim,
            hidden_dim,
            weight_type='down',
            dtype=dtype,
            device=device,
            expert_list=expert_list,
            ep=enable_ep,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        world_size, _ = get_tp_world_rank()
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce
        self.enable_ep = enable_ep

    def update_weights(self):
        """update weights."""
        gate_up_weights, down_weights = self.impl.update_weights(self.gate_up.weight, self.down.weight)
        self.gate_up.update_weight(gate_up_weights)
        self.down.update_weight(down_weights)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        hidden_states, topk_weights, topk_ids = _moe_gather_inputs(hidden_states, topk_weights, topk_ids,
                                                                   self.enable_ep)

        ret = self.impl.forward(hidden_states, topk_weights, topk_ids, self.gate_up.weight, self.down.weight,
                                self.expert_list)
        if self.all_reduce:
            ret = _moe_reduce(ret, self.enable_ep)
        return ret


class LinearWeightsW8A8(LinearWeights):
    """fused moe linear w8a8 weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 device: torch.device,
                 expert_list: List[int] = None,
                 ep: bool = False,
                 quant_dtype: torch.dtype = torch.int8):
        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            weight_type=weight_type,
            dtype=quant_dtype,
            device=device,
            expert_list=expert_list,
            ep=ep,
        )
        scale = torch.empty((num_experts, out_features, 1), dtype=torch.float32, device=device)
        scale = torch.nn.Parameter(scale, requires_grad=False)
        self.register_parameter('scale', scale)

        if self.ep:
            self.scale.weight_loader = self.weight_loader_ep
        else:
            self.scale.weight_loader = self.weight_loader_scale_tp

    def update_weight(self, weight: torch.Tensor, scale: torch.Tensor):
        """update weight."""
        super().update_weight(weight=weight)
        weight_loader = self.scale.weight_loader
        scale = torch.nn.Parameter(scale, requires_grad=False)
        scale.weight_loader = weight_loader
        self.register_parameter('scale', scale)

    def weight_loader_scale_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                               shard_id: str):
        """weight loader scale tp."""
        world_size, rank = get_tp_world_rank()
        if shard_id == 'gate':
            param_data = param.data[expert_id, :self.half_out]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, self.half_out:]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            weight = loaded_weight
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)


class FusedMoEW8A8(nn.Module):
    """fused moe w8a8."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 renormalize: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 quant_dtype: Optional[torch.dtype] = torch.int8,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 enable_ep: bool = False):
        super().__init__()

        if device is None:
            device = torch.device('cpu')
        dtype = torch.float16 if dtype is None else dtype

        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoEW8A8)
        self.impl = impl_builder.build(top_k, num_experts, renormalize, dtype, quant_dtype=quant_dtype)

        enable_ep = enable_ep and self.impl.support_ep()
        if enable_ep:
            world_size, rank = get_tp_world_rank()
            expert_list = self.impl.ep_expert_list(world_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = _update_args(hidden_dim, ffn_dim)
            expert_list = None
        self.expert_list = expert_list

        self.gate_up = LinearWeightsW8A8(num_experts,
                                         hidden_dim,
                                         ffn_dim * 2,
                                         weight_type='gate_up',
                                         device=device,
                                         expert_list=expert_list,
                                         ep=enable_ep,
                                         quant_dtype=quant_dtype)
        self.down = LinearWeightsW8A8(num_experts,
                                      ffn_dim,
                                      hidden_dim,
                                      weight_type='down',
                                      device=device,
                                      expert_list=expert_list,
                                      ep=enable_ep,
                                      quant_dtype=quant_dtype)

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        world_size, _ = get_tp_world_rank()
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce

    def update_weights(self):
        """update weights."""
        (gate_up_weights, down_weights, gate_up_scale,
         down_scale) = self.impl.update_weights(self.gate_up.weight, self.down.weight, self.gate_up.scale,
                                                self.down.scale)
        self.gate_up.update_weight(gate_up_weights, gate_up_scale)
        self.down.update_weight(down_weights, down_scale)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
        ret = self.impl.forward(hidden_states, topk_weights, topk_ids, self.gate_up.weight, self.gate_up.scale,
                                self.down.weight, self.down.scale, self.expert_list)
        if self.all_reduce:
            dist.all_reduce(ret)
        return ret


class LinearWeightsBlockedF8(LinearWeights):
    """fused moe linear blocked fp8 weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 weight_type: str,
                 block_size: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 expert_list: List[int] = None,
                 ep: bool = False):
        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            weight_type=weight_type,
            dtype=dtype,
            device=device,
            expert_list=expert_list,
            ep=ep,
        )
        self.block_size = block_size
        scale = torch.empty((num_experts, div_up(out_features, block_size), div_up(in_features, block_size)),
                            dtype=torch.float32,
                            device=device)
        scale = torch.nn.Parameter(scale, requires_grad=False)
        self.register_parameter('scale', scale)

        if False:#zcx self.ep:
            self.scale.weight_loader = self.weight_loader_ep
        else:
            self.scale.weight_loader = self.weight_loader_scale_tp

    def update_weight(self, weight: torch.Tensor, scale: torch.Tensor):
        """update weight."""
        super().update_weight(weight=weight)
        weight_loader = self.scale.weight_loader
        scale = torch.nn.Parameter(scale, requires_grad=False)
        scale.weight_loader = weight_loader
        self.register_parameter('scale', scale)

    def weight_loader_scale_tp(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int,
                               shard_id: str):
        # zcx begin
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return

        expert_map = dict((eid, idx) for idx, eid in enumerate(expert_list))
        expert_id = expert_map[expert_id]
        # zcx end
        """weight loader scale tp."""
        world_size, rank = get_tp_world_rank()
        block_size = self.block_size
        half_out = self.half_out // block_size
        if shard_id == 'gate':
            param_data = param.data[expert_id, :half_out]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, half_out:]
            weight = loaded_weight.chunk(world_size, dim=0)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            loaded_weight = loaded_weight.to(param_data.device)
            weight = loaded_weight.chunk(world_size, dim=1)[rank]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)


class FusedMoEBlockedF8(nn.Module):
    """fused moe blocked f8."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 renormalize: bool = False,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 all_reduce: bool = True,
                 enable_ep: bool = False):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        dtype = torch.float16 if dtype is None else dtype
        self.block_size = 128
        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoEBlockedF8)
        self.impl = impl_builder.build(top_k, num_experts, renormalize, block_size=self.block_size, out_dtype=dtype)

        enable_ep = enable_ep and self.impl.support_ep()
        # zcx
        self.ep_size, rank = get_ep_world_rank()
        self.deepep_moe = DeepEPMoE(num_experts, self.ep_size)
        expert_list = self.impl.ep_expert_list(self.ep_size, rank)
        num_experts = len(expert_list)
        
        # print(f"zcx:ep_rank={rank}, expert_list={expert_list}")
        # if enable_ep:
        #     world_size, rank = get_tp_world_rank()
        #     expert_list = self.impl.ep_expert_list(world_size, rank)
        #     num_experts = len(expert_list)
        # else:
        #     hidden_dim, ffn_dim = _update_args(hidden_dim, ffn_dim)
        #     expert_list = None
        self.expert_list = expert_list

        self.gate_up = LinearWeightsBlockedF8(num_experts,
                                              hidden_dim,
                                              ffn_dim * 2,
                                              weight_type='gate_up',
                                              block_size=self.block_size,
                                              dtype=fp8_dtype,
                                              device=device,
                                              expert_list=expert_list,
                                              ep=False)# zcx
        self.down = LinearWeightsBlockedF8(
            num_experts,
            ffn_dim,
            hidden_dim,
            weight_type='down',
            block_size=self.block_size,
            dtype=fp8_dtype,
            device=device,
            expert_list=expert_list,
            ep=False,#zcx
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.device = device
        self.enable_ep = enable_ep
        world_size, _ = get_tp_world_rank()
        if world_size == 1:
            all_reduce = False
        self.all_reduce = all_reduce
        

    def update_weights(self):
        """update weights."""
        (gate_up_weights, down_weights, gate_up_scale,
         down_scale) = self.impl.update_weights(self.gate_up.weight, self.down.weight, self.gate_up.scale,
                                                self.down.scale)
        self.gate_up.update_weight(gate_up_weights, gate_up_scale)
        self.down.update_weight(down_weights, down_scale)

    # def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.LongTensor):
    #     hidden_states, topk_weights, topk_ids = _moe_gather_inputs(hidden_states, topk_weights, topk_ids,
    #                                                                self.enable_ep)

    #     ret = self.impl.forward(hidden_states, topk_weights, topk_ids, self.gate_up.weight, self.gate_up.scale,
    #                             self.down.weight, self.down.scale, self.expert_list)
    #     if self.all_reduce:
    #         ret = _moe_reduce(ret, self.enable_ep)
    #     return ret
    def forward(self, hidden_states: torch.Tensor, tokens_per_expert:torch.Tensor):
        return self.deepep_moe.forward(hidden_states, tokens_per_expert, self.gate_up.weight, self.gate_up.scale,
                                self.down.weight, self.down.scale)


class GroupedGemmRunner(torch.nn.Module):
    flashinfer_gemm_warpper = None

    def __init__(self, device, use_flashinfer: bool = False):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        if self.use_flashinfer and GroupedGemmRunner.flashinfer_gemm_warpper is None:
            GroupedGemmRunner._init_flashinfer_wrapper(device)

    @classmethod
    def _init_flashinfer_wrapper(cls, device):
        from flashinfer import SegmentGEMMWrapper

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        cls.flashinfer_gemm_warpper = SegmentGEMMWrapper(workspace_buffer)

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
    ):
        if self.use_flashinfer:
            # TODO: flashinfer
            assert False
            assert GroupedGemmRunner.flashinfer_gemm_warpper is not None
            c = GroupedGemmRunner.flashinfer_gemm_warpper.run(
                x=a,
                weights=b,
                batch_size=batch_size,
                weight_column_major=weight_column_major,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
            )
        else:
            assert weight_column_major == True
            c = grouped_gemm_triton(
                a,
                b,
                c,
                batch_size,
                weight_column_major,
                seg_indptr,
                weight_indices,
                use_fp8_w8a8,
                scale_a,
                scale_b,
                block_shape=block_shape,
            )
        return c


import triton
import triton.language as tl
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8


@triton.jit
def compute_m_range(
    pid,
    batch_size,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    BLOCK_SIZE_M: tl.constexpr,
):
    idx = 0
    for bs in range(batch_size):
        tiles = tl.load(m_num_tiles_indptr + bs)
        if pid >= tiles:
            idx = bs

    idx_start = tl.load(m_num_tiles_indptr + idx)

    m_range_start = tl.load(seg_indptr + idx) + (pid - idx_start) * BLOCK_SIZE_M
    m_range_end = min(tl.load(seg_indptr + idx + 1), m_range_start + BLOCK_SIZE_M)
    expert_id = tl.load(weight_indices + idx)
    return m_range_start, m_range_end, expert_id

@triton.jit
def grouped_gemm_triton_kernel(
    a,
    b,
    c,
    batch_size,
    N,
    K,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    scale_a,
    scale_b,
    use_fp8_w8a8: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    a_stride_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_stride_1: tl.constexpr,
    as_stride_0: tl.constexpr,
    as_stride_1: tl.constexpr,
    bs_stride_0: tl.constexpr,
    bs_stride_2: tl.constexpr,
    bs_stride_1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    c_dtype = c.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m_block = tl.load(m_num_tiles_indptr + batch_size)
    if pid_m >= total_m_block:
        return

    m_range_start, m_range_end, expert_id = compute_m_range(
        pid_m, batch_size, seg_indptr, weight_indices, m_num_tiles_indptr, BLOCK_SIZE_M
    )
    if m_range_end - m_range_start == 0:
        return

    n_range_start = pid_n * BLOCK_SIZE_N
    n_range_end = min(n_range_start + BLOCK_SIZE_N, N)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    offs_am = tl.where(offs_am < m_range_end - m_range_start, offs_am, 0)
    offs_bn = tl.where(offs_bn < n_range_end - n_range_start, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptr = a + (m_range_start + offs_am[:, None]) * a_stride_0 + offs_k[None, :]
    b_ptr = b + (
        (expert_id * b_stride_0)
        + (n_range_start + offs_bn[:, None]) * b_stride_1
        + offs_k[None, :]
    )

    if group_k > 0 and group_n > 0:
        a_scale_ptrs = scale_a + (m_range_start + offs_am[:, None]) * as_stride_0
        offs_bsn = (n_range_start + offs_bn) // group_n
        b_scale_ptrs = scale_b + (expert_id * bs_stride_0) + offs_bsn * bs_stride_1

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(
            a_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )
        b_tile = tl.load(
            b_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )

        if group_k > 0 and group_n > 0:
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_scale = tl.load(a_scale_ptrs + offs_ks * as_stride_1)
            b_scale = tl.load(b_scale_ptrs + offs_ks * bs_stride_2)
            accumulator += tl.dot(a_tile, b_tile.T) * a_scale * b_scale[None, :]
        else:
            accumulator = tl.dot(a_tile, b_tile.T, accumulator)
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K

    if use_fp8_w8a8 and not (group_k > 0 and group_n > 0):
        scale_a_value = tl.load(scale_a + expert_id)
        scale_b_value = tl.load(scale_b + expert_id)
        accumulator *= scale_a_value * scale_b_value

    c_tile = accumulator.to(c_dtype)

    offs_cm = m_range_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_range_start + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m_range_end) & (offs_cn[None, :] < n_range_end)
    tl.store(c_ptr, c_tile, mask=c_mask)

@triton.jit
def compute_m_num_tiles_indptr(
    m_num_tiles_indptr, seg_indptr, batch_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    for bs in range(batch_size):
        m = tl.load(seg_indptr + bs + 1) - tl.load(seg_indptr + bs)
        cur_num_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        pre_num_tiles = tl.load(m_num_tiles_indptr + bs)
        tl.store(m_num_tiles_indptr + bs + 1, pre_num_tiles + cur_num_tiles)



def grouped_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    batch_size: int,
    weight_column_major: bool,
    seg_indptr: Optional[torch.Tensor] = None,
    weight_indices: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
    block_shape: Optional[List[int]] = None,
):
    assert weight_column_major == True  # TODO: more
    if use_fp8_w8a8 and block_shape is None:
        assert scale_a is not None and scale_b is not None

    if block_shape is not None:
        assert len(block_shape) == 2
        block_n, block_k = block_shape[0], block_shape[1]
        # a, scale_a = sglang_per_token_group_quant_fp8(a, block_k)
        assert triton.cdiv(a.shape[-1], block_k) == scale_a.shape[-1]
        assert triton.cdiv(b.shape[-2], block_n) == scale_b.shape[-2]
        assert triton.cdiv(b.shape[-1], block_k) == scale_b.shape[-1]

    # TODO: adjust config or tune kernel
    # Reduce block size to prevent L40 shared memory overflow.
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
    }

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1,)](
        m_num_tiles_indptr, seg_indptr, batch_size, config["BLOCK_SIZE_M"]
    )

    grid = lambda META: (
        triton.cdiv(a.size(0), META["BLOCK_SIZE_M"]) + batch_size,
        triton.cdiv(b.size(1), META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_triton_kernel[grid](
        a,
        b,
        c,
        batch_size,
        b.size(1),
        b.size(2),
        seg_indptr,
        weight_indices,
        m_num_tiles_indptr,
        scale_a,
        scale_b,
        use_fp8_w8a8,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        a.stride(0),
        b.stride(0),
        b.stride(1),
        scale_a.stride(0) if scale_a is not None and scale_a.ndim == 2 else 0,
        scale_a.stride(1) if scale_a is not None and scale_a.ndim == 2 else 0,
        scale_b.stride(0) if scale_b is not None and scale_b.ndim >= 2 else 0,
        scale_b.stride(2) if scale_b is not None and scale_b.ndim == 3 else 0,
        scale_b.stride(1) if scale_b is not None and scale_b.ndim >= 2 else 0,
        **config,
    )
    return c


@triton.jit
def silu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # silu & mul & quantize
            gate_output = gate_output * tl.sigmoid(gate_output)
            gate_output = gate_output.to(InDtype)

            silu_mul_output = gate_output * up_output * scale
            silu_mul_output = silu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, silu_mul_output, mask=mask)


class DeepEPMoE:
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        # top_k: int,
        # hidden_size: int,
        # intermediate_size: int,
        # params_dtype: Optional[torch.dtype] = None,
        # renormalize: bool = True,
        # use_grouped_topk: bool = False,
        # num_expert_group: Optional[int] = None,
        # topk_group: Optional[int] = None,
        # quant_config: Optional[QuantizationConfig] = None,
        ep_size: Optional[int] = None,
        # prefix: str = "",
        # correction_bias: Optional[torch.Tensor] = None,
        # custom_routing_function: Optional[Callable] = None,
        # activation: str = "silu",
    ):
        # super().__init__(
        #     num_experts,
        #     top_k,
        #     hidden_size,
        #     intermediate_size,
        #     params_dtype,
        #     renormalize,
        #     use_grouped_topk,
        #     num_expert_group,
        #     topk_group,
        #     # quant_config,
        #     tp_size,
        #     prefix,
        #     correction_bias,
        #     custom_routing_function,
        #     activation,
        # )
        self.num_experts = num_experts
        self.ep_size = ep_size
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.block_shape = [128, 128]
        self.use_fp8_w8a8 = True

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     tokens_per_expert: torch.Tensor,
    # ):
        # Todo @sleepcoo: use m_grouped_gemm_fp8_fp8_bf16_nt_masked after low_latency dispatch (decode)
        # return self.forward_normal(hidden_states, tokens_per_expert)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        gate_up_weight:torch.Tensor,
        gate_up_scale:torch.Tensor,
        gate_down_weight:torch.Tensor,
        gate_down_scale:torch.Tensor
    ):
        # assert self.quant_method is not None
        # assert self.activation == "silu"
        # hidden_states = torch.load("/nvme1/zhaochaoxing/hidden_states.pt")
        # tokens_per_expert = torch.load("/nvme1/zhaochaoxing/tokens_per_expert.pt")
        # print(f"zcx:gate_up_weight:{gate_up_weight},gate_up_scale:{gate_up_scale}")
        # print(f"zcx:gate_up_weight:{gate_up_weight.shape},gate_up_scale:{gate_up_scale.shape}")
        # print(f"zcx:gate_down_weight:{gate_down_weight},gate_down_scale:{gate_down_scale}")
        # raise RuntimeError()
        input_size = hidden_states.shape
        
        self.grouped_gemm_runner = GroupedGemmRunner(
            hidden_states.device, use_flashinfer=False  # TODO: use flashinfer
        )
        seg_indptr_cur_rank = torch.cat(
            [
                torch.zeros(
                    1, device=tokens_per_expert.device, dtype=tokens_per_expert.dtype
                ),
                torch.cumsum(tokens_per_expert, dim=0),
            ]
        )
        reorder_topk_ids = torch.repeat_interleave(tokens_per_expert)
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        gateup_output = torch.empty(
            hidden_states.shape[0],
            gate_up_weight.shape[1],
            #self.w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if hidden_states.shape[0] > 0:
            input, input_scale = quant_fp8(hidden_states, 128, dtype=gate_up_weight.dtype)
            # input = input.unflatten(0, input_size[:-1])
            gateup_output = self.grouped_gemm_runner(
                a=input,
                b=gate_up_weight,
                #b=self.w13_weight,
                c=gateup_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=input_scale,
                scale_b=gate_up_scale,#(
                    # self.w13_weight_scale_inv
                    #if self.use_block_quant
                    #else self.w13_weight_scale
                # ),
                block_shape=self.block_shape,
            )
        # print(f"zcx:gateup_output:{gateup_output}, {gateup_output.shape},{gateup_output.dtype}")
        # raise RuntimeError()
        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(hidden_states.dtype
                #self.fp8_dtype
                #if (self.use_fp8_w8a8 and not self.use_block_quant)
                #else hidden_states.dtype
            ),
        )
        silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            None,
            # self.w2_input_scale,
            0,
            self.num_experts_per_partition - 1,
            BLOCK_SIZE=512,
        )
        
        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            gate_down_weight.shape[1],
            #self.w2_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if down_input.shape[0] > 0:
            down_input, down_input_scale = quant_fp8(down_input, 128, dtype=gate_down_weight.dtype)
            down_output = self.grouped_gemm_runner(
                a=down_input,
                # b=self.w2_weight,
                b=gate_down_weight,
                c=down_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=down_input_scale,
                scale_b=gate_down_scale, #(
                    # self.w2_weight_scale_inv
                    #if self.use_block_quant
                    #else self.w2_weight_scale
                # ),
                block_shape=self.block_shape,
            )
        return down_output

def build_fused_moe(
    hidden_dim: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    renormalize: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    all_reduce: bool = True,
    enable_ep: bool = False,
    quant_config: Any = None,
):
    """fused moe builder."""

    if quant_config is None:
        return FusedMoE(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
        )

    quant_method = quant_config['quant_method']
    if quant_method == 'smooth_quant':
        quant_dtype = eval('torch.' + quant_config.get('quant_dtype', 'int8'))
        return FusedMoEW8A8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            quant_dtype=quant_dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
        )
    elif quant_method == 'fp8':
        fmt = quant_config.get('fmt', 'e4m3')
        if fmt == 'e4m3':
            fp8_dtype = torch.float8_e4m3fn
        elif fmt == 'e5m2':
            fp8_dtype = torch.float8_e5m2
        else:
            raise TypeError(f'Unsupported fp8 fmt: {fmt}')
        return FusedMoEBlockedF8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            fp8_dtype=fp8_dtype,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            enable_ep=enable_ep,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')
