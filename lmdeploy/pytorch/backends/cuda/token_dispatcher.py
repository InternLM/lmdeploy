# Copyright (c) OpenMMLab. All rights reserved.
try:
    from deep_ep import Buffer
    use_deepep = True
except ImportError:
    use_deepep = False

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.cuda.ep_moe import (deepep_permute_triton_kernel, deepep_post_reorder_triton_kernel,
                                                  deepep_run_moe_deep_preprocess)

from ..default.token_dispatcher import AlltoAllTokenDispatcher
from ..token_dispatcher import TokenDispatcherImpl

_buffer_normal = None
_buffer_low_latency = None
_buffer_common = None


def get_buffer_common(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
    hidden_bytes: int,
):
    global _buffer_common
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    num_rdma_bytes = max(
        Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts),
        num_rdma_bytes)

    if (_buffer_common is None or _buffer_common.group != group or _buffer_common.num_nvl_bytes < num_nvl_bytes
            or _buffer_common.num_rdma_bytes < num_rdma_bytes):
        _buffer_common = Buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // group.size(),
        )
    return _buffer_common


def get_buffer_normal(group: dist.ProcessGroup, hidden_bytes: int):
    """Copy from DeepEP example usage in model inference prefilling.

    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-model-training-or-inference-prefilling
    """
    global _buffer_normal
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    if (_buffer_normal is None or _buffer_normal.group != group or _buffer_normal.num_nvl_bytes < num_nvl_bytes
            or _buffer_normal.num_rdma_bytes < num_rdma_bytes):
        _buffer_normal = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer_normal


def get_buffer_low_latency(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
):
    """Copy from DeepEP example usage in model inference decoding.

    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
    """

    global _buffer_low_latency
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(),
                                                           num_experts)

    if (_buffer_low_latency is None or _buffer_low_latency.group != group or not _buffer_low_latency.low_latency_mode
            or _buffer_low_latency.num_rdma_bytes < num_rdma_bytes):
        assert num_experts % group.size() == 0, f'num_experts:{num_experts} must be divisible by ep_size:{group.size()}'
        _buffer_low_latency = Buffer(
            group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // group.size(),
        )
    return _buffer_low_latency


class DeepEPTokenDispatcher(TokenDispatcherImpl):
    """Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-
    LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py."""

    def __init__(self,
                 group: torch.distributed.ProcessGroup,
                 top_k: int,
                 num_experts: int = None,
                 num_local_experts: int = None,
                 hidden_size: int = None,
                 params_dtype: torch.dtype = None,
                 num_max_dispatch_tokens_per_rank=128,
                 async_finish: bool = True):
        self.group = group
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize
        # Handle used for combine operation
        self.handle = None
        self.async_finish = async_finish
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.buffer_normal = get_buffer_common(self.group,
                                               num_max_dispatch_tokens_per_rank,
                                               self.hidden_size,
                                               self.num_experts,
                                               hidden_bytes=self.hidden_size * self.params_bytes)

    def permute(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        fp8_dtype: Optional[torch.dtype] = None,
        use_fp8_w8a8: bool = False,
        use_block_quant: bool = False,
    ):
        reorder_topk_ids, self.src2dst, seg_indptr = deepep_run_moe_deep_preprocess(topk_idx, self.num_experts)
        num_total_tokens = reorder_topk_ids.numel()
        gateup_input = torch.empty(
            (int(num_total_tokens), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(fp8_dtype if (use_fp8_w8a8 and not use_block_quant) else hidden_states.dtype),
        )
        deepep_permute_triton_kernel[(hidden_states.shape[0], )](
            hidden_states,
            gateup_input,
            self.src2dst,
            topk_idx,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )
        return reorder_topk_ids, seg_indptr, gateup_input

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_list: List[int] = None,
        previous_event=None,
    ):
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        (
            hidden_states,
            topk_idx,
            topk_weights,
            _,
            event,
        ) = self.dispatch_normal(hidden_states, topk_idx, topk_weights, self.num_experts, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        if hidden_states.shape[0] > 0:
            reorder_topk_ids, seg_indptr, hidden_states = self.permute(hidden_states,
                                                                       topk_idx,
                                                                       fp8_dtype=hidden_states.dtype)
        return hidden_states, topk_idx, topk_weights, reorder_topk_ids, seg_indptr

    def dispatch_normal(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        previous_event=None,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer_normal.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            self.handle,
            event,
        ) = self.buffer_normal.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights.to(torch.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
        )
        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            event,
        )

    def combine(self, hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            num_tokens = self.src2dst.shape[0] // self.top_k
            output = torch.empty(
                (num_tokens, hidden_states.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            deepep_post_reorder_triton_kernel[(num_tokens, )](
                hidden_states,
                output,
                self.src2dst,
                topk_idx,
                topk_weights,
                self.top_k,
                hidden_states.shape[1],
                BLOCK_SIZE=512,
            )
        else:
            output = torch.zeros(
                (0, hidden_states.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        previous_event = Buffer.capture() if self.async_finish else None
        output, _, event = self.buffer_normal.combine(
            output,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
        )
        event.current_stream_wait() if self.async_finish else ()
        return output.view(self.hidden_shape)


class DeepEPTokenDispatcherLowLatency(TokenDispatcherImpl):

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        return_recv_hook: bool = False,
    ):
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize
        self.handle = None
        self.num_max_dispatch_tokens_per_rank = 128
        self.buffer_low_latency = get_buffer_common(self.group,
                                                    self.num_max_dispatch_tokens_per_rank,
                                                    self.hidden_size,
                                                    self.num_experts,
                                                    hidden_bytes=self.hidden_size * self.params_bytes)
        self.return_recv_hook = return_recv_hook

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx = topk_idx.to(torch.int64)
        expected_m = (hidden_states.shape[0] * self.buffer_low_latency.group_size * topk_idx.shape[1] +
                      num_experts) // num_experts

        packed_recv_hidden, masked_m, self.handle, event, hook = (self.buffer_low_latency.low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=True,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        ))
        hook() if self.return_recv_hook else event.current_stream_wait()
        return (
            packed_recv_hidden,
            topk_idx,
            topk_weights,
            masked_m,
            expected_m,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        combined_hidden_states, event, hook = (self.buffer_low_latency.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights.to(torch.float32),
            self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        ))
        hook() if self.return_recv_hook else event.current_stream_wait()
        return combined_hidden_states


class TokenDispatcherBuilder:
    """token dispatcher builder."""

    @staticmethod
    def build(
        group,
        top_k,
        num_experts,
        num_local_experts,
        hidden_size,
        params_dtype,
    ) -> TokenDispatcherImpl:
        """build."""
        if use_deepep is True:
            return DeepEPTokenDispatcher(
                group,
                top_k,
                num_experts,
                num_local_experts,
                hidden_size,
                params_dtype,
            )
        else:
            return AlltoAllTokenDispatcher(
                group,
                num_experts,
                num_local_experts,
            )
