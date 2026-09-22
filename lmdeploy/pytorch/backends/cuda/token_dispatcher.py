# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from enum import Enum

from lmdeploy.pytorch.envs import deep_ep_buffer_num_sms, env_to_int

try:
    from deep_ep import Buffer

    Buffer.set_num_sms(deep_ep_buffer_num_sms)
    use_deepep = True
except ImportError:
    Buffer = None
    use_deepep = False


import torch
import torch.distributed as dist

from ..default.token_dispatcher import AlltoAllTokenDispatcher
from ..token_dispatcher import TokenDispatcherImpl


class DeepEPMode(Enum):
    """DeepEP communication mode."""

    NORMAL = 'normal'
    LOW_LATENCY = 'low_latency'
    AUTO = 'auto'


class DisposibleTensor:
    """Tensor wrapper that allows eager disposal while preserving metadata."""

    def __init__(self, value: torch.Tensor):
        self._value = value
        self._backup_metadata = None

    @property
    def value(self):
        assert not self.is_disposed
        return self._value

    @property
    def is_disposed(self):
        return self._value is None

    def dispose(self, backup_metadata: bool = True):
        assert not self.is_disposed
        if not torch.compiler.is_compiling() and sys.getrefcount(self._value) != 2:
            return
        if backup_metadata:
            self._backup_metadata = {key: getattr(self._value, key) for key in ('shape', 'device', 'dtype')}
        self._value = None

    @staticmethod
    def maybe_unwrap(value):
        return value.value if isinstance(value, DisposibleTensor) else value

    @staticmethod
    def maybe_dispose(value):
        if isinstance(value, DisposibleTensor):
            value.dispose()

    @property
    def shape(self):
        return self._get_metadata('shape')

    @property
    def device(self):
        return self._get_metadata('device')

    @property
    def dtype(self):
        return self._get_metadata('dtype')

    def _get_metadata(self, name: str):
        if not self.is_disposed:
            return getattr(self._value, name)
        assert self._backup_metadata is not None
        return self._backup_metadata[name]


class DeepEPBuffer:
    """LMDeploy-owned DeepEP buffer facade."""

    _buffer_normal = None
    _buffer_low_latency = None
    _buffer_common = None
    _deepep_mode = DeepEPMode.AUTO
    _deepep_sms = deep_ep_buffer_num_sms if use_deepep else 20
    _num_max_dispatch_tokens_per_rank = 128
    _allow_mnnvl = True
    _latest_mode = DeepEPMode.AUTO
    _hidden_size = -1
    _num_experts = -1
    _explicitly_destroy = False

    @classmethod
    def _build_buffer(cls, *args, **kwargs):
        """Build a DeepEP Buffer while tolerating older constructor
        signatures."""
        try:
            return Buffer(*args, **kwargs)
        except TypeError:
            kwargs.pop('allow_mnnvl', None)
            kwargs.pop('explicitly_destroy', None)
            return Buffer(*args, **kwargs)

    @classmethod
    def set_explicitly_destroy(cls):
        if cls._buffer_common is not None or cls._buffer_normal is not None or cls._buffer_low_latency is not None:
            return False
        if not cls._explicitly_destroy:
            cls._explicitly_destroy = True
            return True
        return False

    @classmethod
    def get_explicitly_destroy(cls):
        return cls._explicitly_destroy

    @classmethod
    def destroy(cls):
        if not cls._explicitly_destroy:
            return False
        if cls._buffer_common is not None:
            cls._buffer_common.destroy()
            cls._buffer_common = None
            return True
        if cls._buffer_low_latency is not None:
            cls._buffer_low_latency.destroy()
            cls._buffer_low_latency = None
            return True
        if cls._buffer_normal is not None:
            cls._buffer_normal.destroy()
            cls._buffer_normal = None
            return True
        return False

    @classmethod
    def update_parameters(cls, hidden_size: int, num_experts: int):
        cls._hidden_size = hidden_size
        cls._num_experts = num_experts
        cls._deepep_sms = env_to_int('DEEPEP_BUFFER_NUM_SMS', cls._deepep_sms)
        cls._allow_mnnvl = os.getenv('DEEPEP_ENABLE_MNNVL', '1') != '0'
        env_mode = os.getenv('DEEPEP_MODE', 'auto').strip().lower()
        if env_mode == 'normal':
            cls._deepep_mode = DeepEPMode.NORMAL
        elif env_mode == 'low_latency':
            cls._deepep_mode = DeepEPMode.LOW_LATENCY
        else:
            cls._deepep_mode = DeepEPMode.AUTO

    @classmethod
    def set_deepep_mode(cls, mode: DeepEPMode):
        low_latency_buffer_cleaned = False
        if (cls._deepep_mode == DeepEPMode.AUTO and mode == DeepEPMode.LOW_LATENCY
                and cls._latest_mode == DeepEPMode.NORMAL):
            cls.clean_low_latency_buffer(cls._buffer_common)
            low_latency_buffer_cleaned = True
        cls._latest_mode = mode
        return cls._latest_mode, low_latency_buffer_cleaned

    @classmethod
    def clean_low_latency_buffer(cls, buffer=None):
        if buffer is None:
            buffer = cls._buffer_common
        if use_deepep and isinstance(buffer, Buffer):
            buffer.clean_low_latency_buffer(cls._num_max_dispatch_tokens_per_rank, cls._hidden_size, cls._num_experts)

    @classmethod
    def get_buffer_common(
        cls,
        group: dist.ProcessGroup,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        hidden_bytes: int,
    ):
        if cls._buffer_common is not None:
            # Match dlblas/DeepEP's process-wide common buffer lifetime.
            return cls._buffer_common

        cls.update_parameters(hidden, num_experts)
        num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank or cls._num_max_dispatch_tokens_per_rank
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank

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

        assert num_experts % group.size(
        ) == 0, f'num_experts: {num_experts} must be divisible by ep_size: {group.size()}'
        cls._buffer_common = cls._build_buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=max(num_experts // group.size(), cls._deepep_sms),
            allow_mnnvl=cls._allow_mnnvl,
            explicitly_destroy=cls._explicitly_destroy,
        )
        cls._buffer_common.set_num_sms(cls._deepep_sms)
        return cls._buffer_common

    @classmethod
    def get_buffer_normal(cls, group: dist.ProcessGroup, hidden_bytes: int):
        num_nvl_bytes, num_rdma_bytes = 0, 0
        for config in (
                Buffer.get_dispatch_config(group.size()),
                Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

        if (cls._buffer_normal is None or cls._buffer_normal.group != group
                or cls._buffer_normal.num_nvl_bytes < num_nvl_bytes
                or cls._buffer_normal.num_rdma_bytes < num_rdma_bytes):
            cls._buffer_normal = cls._build_buffer(group,
                                                   num_nvl_bytes,
                                                   num_rdma_bytes,
                                                   explicitly_destroy=cls._explicitly_destroy)
        return cls._buffer_normal

    @classmethod
    def get_buffer_low_latency(
        cls,
        group: dist.ProcessGroup,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ):
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(),
                                                               num_experts)

        if (cls._buffer_low_latency is None or cls._buffer_low_latency.group != group
                or not cls._buffer_low_latency.low_latency_mode
                or cls._buffer_low_latency.num_rdma_bytes < num_rdma_bytes):
            assert num_experts % group.size(
            ) == 0, f'num_experts: {num_experts} must be divisible by ep_size: {group.size()}'
            cls._buffer_low_latency = cls._build_buffer(
                group,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=max(num_experts // group.size(), Buffer.num_sms // 2),
                explicitly_destroy=cls._explicitly_destroy,
            )
        return cls._buffer_low_latency


def get_buffer_common(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
    hidden_bytes: int,
):
    return DeepEPBuffer.get_buffer_common(group, num_max_dispatch_tokens_per_rank, hidden, num_experts, hidden_bytes)


def get_buffer_normal(group: dist.ProcessGroup, hidden_bytes: int):
    """Copy from DeepEP example usage in model inference prefilling.

    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-model-training-or-inference-prefilling
    """
    return DeepEPBuffer.get_buffer_normal(group, hidden_bytes)


def get_buffer_low_latency(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
):
    """Copy from DeepEP example usage in model inference decoding.

    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
    """

    return DeepEPBuffer.get_buffer_low_latency(group, num_max_dispatch_tokens_per_rank, hidden, num_experts)


class DeepEPTokenDispatcher(TokenDispatcherImpl):
    """Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-
    LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        num_max_dispatch_tokens_per_rank=128,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize
        # Handle used for combine operation
        self.handle = None
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.buffer_normal = get_buffer_common(self.group,
                                               num_max_dispatch_tokens_per_rank,
                                               self.hidden_size,
                                               self.num_experts,
                                               hidden_bytes=self.hidden_size * self.params_bytes)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_list: list[int] = None,
        previous_event=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        (
            hidden_states,
            topk_idx,
            topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.dispatch_normal(hidden_states, topk_idx, topk_weights, self.num_experts, previous_event)
        self.tokens_per_expert = torch.tensor(
            recv_tokens_per_expert,
            device=hidden_states.device,
            dtype=torch.int64,
        )
        tokens_per_expert = self.get_number_of_tokens_per_expert()
        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        if hidden_states.shape[0] > 0:
            hidden_states, _, _, _, _ = self.get_permuted_hidden_states_by_experts(hidden_states)
        return hidden_states, topk_idx, topk_weights, tokens_per_expert

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
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
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
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        )

    def dispatch_normal_async(self,
                              x: torch.Tensor,
                              topk_idx: torch.Tensor,
                              topk_weights: torch.Tensor,
                              num_experts: int | None = None,
                              previous_event=None,
                              async_finish=True):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer_normal.get_dispatch_layout(
            topk_idx,
            num_experts=self.num_experts if num_experts is None else num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.buffer_normal.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        )

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            hidden_states = self.get_restored_hidden_states_by_experts(hidden_states)
        hidden_states, event = self.combine_normal(hidden_states, self.handle)
        self.handle = None
        return hidden_states.view(self.hidden_shape)

    def combine_normal(self, x: torch.Tensor, handle: tuple, previous_event=None):
        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def combine_normal_async(self, x: torch.Tensor, handle: tuple, previous_event=None, async_finish=True):
        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )
        return combined_x, event

    def release(self):
        self.tokens_per_expert = None
        self.handle = None
        self.topk_idx = None
        self.topk_weights = None
        self.hidden_shape_before_permute = None
        self.dispatched_routing_map = None
        self.reversed_mapping_for_combine = None
        return True

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """Get the number of tokens per expert."""
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(self,
                                              hidden_states: torch.Tensor,
                                              topk_idx: torch.Tensor | None = None,
                                              topk_weights: torch.Tensor | None = None,
                                              num_experts: int | None = None) -> torch.Tensor:
        (dispatched_routing_map,
         topk_weights) = super().indices_to_multihot(self.topk_idx if topk_idx is None else topk_idx,
                                                     self.topk_weights if topk_weights is None else topk_weights,
                                                     self.num_experts if num_experts is None else num_experts)
        hidden_states_shape = hidden_states.shape
        (hidden_states, reversed_mapping_for_combine) = super().permute(
            hidden_states,
            dispatched_routing_map,
        )
        self.hidden_shape_before_permute = hidden_states_shape
        self.dispatched_routing_map = dispatched_routing_map
        self.topk_weights = topk_weights
        self.reversed_mapping_for_combine = reversed_mapping_for_combine
        return hidden_states, hidden_states_shape, dispatched_routing_map, topk_weights, reversed_mapping_for_combine

    def get_restored_hidden_states_by_experts(
        self,
        hidden_states: torch.Tensor,
        reversed_mapping_for_combine: torch.Tensor | None = None,
        hidden_shape_before_permute: torch.Size | None = None,
        dispatched_routing_map: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        assert (self.topk_weights.dtype == torch.float32), 'DeepEP only supports float32 probs'
        hidden_states = super().unpermute(
            hidden_states,
            sorted_indices=self.reversed_mapping_for_combine
            if reversed_mapping_for_combine is None else reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute
            if hidden_shape_before_permute is None else hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map if dispatched_routing_map is None else dispatched_routing_map,
            probs=self.topk_weights if topk_weights is None else topk_weights,
        )
        return hidden_states.to(input_dtype)


class DeepEPTokenDispatcherNormal(TokenDispatcherImpl):
    """DeepEP normal-mode dispatcher used by LMDeploy EP MoE."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        num_max_dispatch_tokens_per_rank: int = 128,
        expert_alignment: int = 128,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.handle = None
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.buffer_normal = DeepEPBuffer.get_buffer_common(
            self.group,
            self.num_max_dispatch_tokens_per_rank,
            self.hidden_size,
            self.num_experts,
            hidden_bytes=self.hidden_size * self.params_bytes,
        )
        self.expert_alignment = expert_alignment

    def get_buffer(self):
        return self.buffer_normal

    def dispatch(
        self,
        x,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_list: list[int] = None,
        previous_event=None,
    ):
        hidden_states = x[0] if isinstance(x, tuple) else x
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        x, topk_idx, topk_weights, recv_tokens_per_expert, handle, event = self.dispatch_normal(
            x, topk_idx, topk_weights, self.num_experts, previous_event)

        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        return x, topk_idx, topk_weights, recv_tokens_per_expert

    def dispatch_normal(
        self,
        x,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        previous_event=None,
    ):
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = (
            self.get_buffer().get_dispatch_layout(
                topk_idx,
                num_experts,
                previous_event=previous_event,
                async_finish=False,
                allocate_on_comm_stream=False,
            ))

        recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert, handle, event = self.get_buffer().dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights.to(torch.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
            expert_alignment=self.expert_alignment,
        )

        return recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert, handle, event

    def dispatch_normal_async(self,
                              x,
                              topk_idx: torch.Tensor,
                              topk_weights: torch.Tensor,
                              num_experts: int | None = None,
                              previous_event=None,
                              async_finish=True):
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = (
            self.get_buffer().get_dispatch_layout(
                topk_idx,
                num_experts=self.num_experts if num_experts is None else num_experts,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=previous_event is not None and async_finish,
            ))

        recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert, handle, event = self.get_buffer().dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
            expert_alignment=self.expert_alignment,
        )

        return recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert, handle, event

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, event = self.combine_normal(hidden_states, self.handle)
        self.handle = None
        return hidden_states.view(self.hidden_shape)

    def combine_normal(self, x: torch.Tensor, handle: tuple, previous_event=None):
        combined_x, _, event = self.get_buffer().combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def combine_normal_async(self, x: torch.Tensor, handle: tuple, previous_event=None, async_finish=True):
        combined_x, _, event = self.get_buffer().combine(
            x,
            handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )
        return combined_x, event

    def release(self):
        self.handle = None
        self.topk_idx = None
        self.topk_weights = None
        return True


class DeepEPTokenDispatcherLowLatency(TokenDispatcherImpl):

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        num_max_dispatch_tokens_per_rank: int = 128,
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
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.buffer_low_latency = DeepEPBuffer.get_buffer_common(self.group,
                                                                 self.num_max_dispatch_tokens_per_rank,
                                                                 self.hidden_size,
                                                                 self.num_experts,
                                                                 hidden_bytes=self.hidden_size * self.params_bytes)
        self.return_recv_hook = return_recv_hook

    def get_buffer(self):
        return self.buffer_low_latency

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_idx = topk_idx.to(torch.int64)
        expected_m = (hidden_states.shape[0] * self.get_buffer().group_size * topk_idx.shape[1] +
                      num_experts) // num_experts

        packed_recv_hidden, masked_m, self.handle, event, hook = (self.get_buffer().low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=True,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        ))
        hook() if self.return_recv_hook else event.current_stream_wait()
        packed_recv_hidden = [DisposibleTensor(x) for x in packed_recv_hidden]
        return (
            packed_recv_hidden,
            topk_idx,
            topk_weights,
            masked_m,
            expected_m,
        )

    def dispatch_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: int | None = None,
        use_fp8: bool = True,
        async_finish: bool = True,
    ):
        assert topk_idx.dtype == torch.int64
        recv_hidden_states, recv_expert_count, handle, event, hook = (self.get_buffer().low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            num_experts=self.num_experts if num_experts is None else num_experts,
            use_fp8=use_fp8,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        ))
        recv_hidden_states = [DisposibleTensor(x) for x in recv_hidden_states]
        return recv_hidden_states, recv_expert_count, handle, event, hook

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        combined_hidden_states, event, hook = (self.get_buffer().low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights.to(torch.float32),
            self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        ))
        hook() if self.return_recv_hook else event.current_stream_wait()
        return combined_hidden_states

    def combine_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        async_finish: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert topk_idx.dtype == torch.int64
        assert topk_weights.dtype == torch.float32
        combined_hidden_states, event, hook = self.get_buffer().low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        )
        return combined_hidden_states, event, hook


class TokenDispatcherBuilder:
    """Token dispatcher builder."""

    @staticmethod
    def build(
        group,
        num_experts,
        num_local_experts,
        hidden_size,
        params_dtype,
    ) -> TokenDispatcherImpl:
        """build."""
        if use_deepep is True:
            return DeepEPTokenDispatcher(
                group,
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
