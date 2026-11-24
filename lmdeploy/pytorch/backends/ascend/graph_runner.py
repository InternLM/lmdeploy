# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
# this file implements the cudagraph for ascend backend.
import functools
from contextlib import ExitStack
from typing import Any

import torch
from torch import Tensor
from torch.profiler import record_function

from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
BuffType = dict[str, Tensor]


def _is_sparse_attention(graph_meta: CudaGraphMeta) -> bool:
    """Whether this graph runs the device-only NSA/DSA attention path."""
    return getattr(graph_meta, 'mla_index_topk', None) is not None


# AscendCudaGraphMixin methods for cudagraph buffer management.
def support_cuda_graph(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: list[list[Tensor]],
    attn_metadata: Any = None,
    inputs_embeds: Tensor = None,
    **kwargs,
):
    """Allow decode graph after the Ascend runtime was validated at import."""
    if attn_metadata is None:
        return False

    # Graph capture wraps the MoE EP dispatch/combine collective, so all DP ranks
    # must enter or skip the graph together. Gate on the DP-global decode state
    # (any rank prefill => all prefill) rather than this rank's local is_decoding.
    if not get_step_ctx_manager().current_context().global_is_decoding():
        return False
    return getattr(attn_metadata, 'is_decoding', False) or getattr(attn_metadata, 'is_multi_token_decoding', False)


def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, *args, **kwargs) -> BuffType:
    """Make cudagraph buffers from forward inputs."""
    max_batches = graph_meta.max_batchs
    max_tokens = graph_meta.max_tokens
    num_blocks = graph_meta.num_blocks
    device = graph_meta.device
    input_buffers: BuffType = dict()
    input_buffers['input_ids'] = torch.zeros((1, max_tokens), dtype=torch.int32, device=device)

    input_buffers['position_ids'] = torch.zeros((1, max_tokens), dtype=torch.int32, device=device)

    input_buffers['block_offsets'] = torch.zeros((max_batches, num_blocks), dtype=torch.int32, device=device)

    input_buffers['kv_seqlens'] = torch.ones(max_batches, dtype=torch.int32, device=device)
    if not _is_sparse_attention(graph_meta):
        input_buffers['kv_seqlens_cpu'] = torch.ones(max_batches, dtype=torch.int32)

    input_buffers['kv_start_indices'] = -torch.ones((max_tokens), dtype=torch.int32, device=device)

    input_buffers['x_active_mask'] = torch.zeros((max_tokens), dtype=torch.bool, device=device)

    if not _is_sparse_attention(graph_meta):
        input_buffers['attention_mask'] = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=device),
            diagonal=1,
        )

    # ssm
    if graph_meta.is_ssm:
        input_buffers['state_ids'] = torch.full((max_batches,), -1, dtype=torch.int64, device=device)
        input_buffers['cache_seqlens'] = torch.zeros(max_batches, dtype=torch.int32, device=device)

    query_len = max_tokens // max_batches
    input_buffers['q_seqlens'] = torch.full((max_batches,), query_len, dtype=torch.int32, device=device)
    input_buffers['cu_seqlens_q'] = torch.arange(max_batches + 1, dtype=torch.int32, device=device) * query_len
    if not _is_sparse_attention(graph_meta):
        input_buffers['cu_seqlens_q_cpu'] = torch.arange(max_batches + 1, dtype=torch.int32) * query_len
    # Keep q_start_loc as a compatibility view with its actual [B] start-offset
    # semantics.  Attention kernels consume cu_seqlens_q directly.
    input_buffers['q_start_loc'] = input_buffers['cu_seqlens_q'][:-1]

    # mrope
    if graph_meta.use_mrope:
        input_buffers['mrope_position_ids'] = torch.zeros(3, max_tokens, dtype=torch.int64, device=device)

    return input_buffers


def fill_buffers_cudagraph(
    self,
    graph_meta: CudaGraphMeta,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: list,
    attn_metadata: Any,
    inputs_embeds: Tensor,
    **kwargs,
) -> dict[str, Tensor]:
    """Fill cudagraph buffers from forward inputs."""
    block_offsets: Tensor = attn_metadata.block_offsets
    kv_seqlens: Tensor = attn_metadata.kv_seqlens
    kv_start_indices: Tensor = attn_metadata.kv_start_indices
    moe_metadata = get_step_ctx_manager().current_context().moe_metadata
    x_active_mask: Tensor = moe_metadata.x_active_mask
    cache_seqlens: Tensor = attn_metadata.cache_seqlens

    is_multi_token_decoding = attn_metadata.is_multi_token_decoding

    input_buffers: BuffType = graph_meta.input_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # fill buffer
    max_num_tokens = input_buffers['input_ids'].size(-1)
    if num_tokens < max_num_tokens:
        # Only initialize the padding region to avoid unnecessary work and RNG advancement
        input_buffers['input_ids'][:, num_tokens:max_num_tokens].random_(0, graph_meta.vocab_size)

    input_buffers['input_ids'][:, :num_tokens] = input_ids
    input_buffers['position_ids'].zero_()
    input_buffers['position_ids'][:, :num_tokens] = position_ids
    input_buffers['block_offsets'].zero_()
    input_buffers['block_offsets'][:batch_size, :num_blocks] = block_offsets

    input_buffers['kv_seqlens'].fill_(0)
    input_buffers['kv_seqlens'][:batch_size] = kv_seqlens
    if not _is_sparse_attention(graph_meta):
        kv_seqlens_cpu: Tensor = attn_metadata.kv_seqlens_cpu
        if kv_seqlens_cpu is None:
            raise RuntimeError('Ascend paged-attention graph requires kv_seqlens_cpu')
        input_buffers['kv_seqlens_cpu'].fill_(0)
        input_buffers['kv_seqlens_cpu'][:batch_size] = kv_seqlens_cpu
    input_buffers['kv_start_indices'].fill_(-1)
    input_buffers['kv_start_indices'][: kv_start_indices.size(0)] = kv_start_indices
    if x_active_mask is not None:
        input_buffers['x_active_mask'].fill_(0)
        input_buffers['x_active_mask'][: x_active_mask.size(0)] = x_active_mask

    if graph_meta.is_ssm:
        state_ids = kwargs['state_ids']
        input_buffers['state_ids'].fill_(0)
        input_buffers['state_ids'][:batch_size].copy_(state_ids)

        if is_multi_token_decoding:
            input_buffers['cache_seqlens'].fill_(0)
            input_buffers['cache_seqlens'][:batch_size].copy_(cache_seqlens)

            attn_metadata.cache_seqlens = input_buffers['cache_seqlens']

    if is_multi_token_decoding and not _is_sparse_attention(graph_meta):
        attn_metadata.attention_mask = [input_buffers['attention_mask']]

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if 'inputs_embeds' not in input_buffers:
            max_num_tokens = input_buffers['input_ids'].size(-1)
            input_buffers['inputs_embeds'] = inputs_embeds.new_zeros(1, max_num_tokens, emb_size)
        input_buffers['inputs_embeds'][:, :num_tokens] = inputs_embeds

    attn_metadata.block_offsets = input_buffers['block_offsets']
    attn_metadata.kv_seqlens = input_buffers['kv_seqlens']
    attn_metadata.kv_seqlens_cpu = input_buffers.get('kv_seqlens_cpu')
    attn_metadata.kv_start_indices = input_buffers['kv_start_indices']
    moe_metadata.x_active_mask = input_buffers['x_active_mask']
    attn_metadata.q_start_loc = input_buffers['q_start_loc']
    attn_metadata.q_seqlens = input_buffers['q_seqlens']
    attn_metadata.cu_seqlens_q = input_buffers['cu_seqlens_q']
    attn_metadata.cu_seqlens_q_cpu = input_buffers.get('cu_seqlens_q_cpu')

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
        moe_metadata=moe_metadata,
    )

    new_inputs['input_ids'] = input_buffers['input_ids']
    new_inputs['position_ids'] = input_buffers['position_ids']

    if inputs_embeds is not None:
        new_inputs['inputs_embeds'] = input_buffers['inputs_embeds']

    new_inputs.update(kwargs)

    if graph_meta.is_ssm:
        new_inputs['state_ids'] = input_buffers['state_ids']

    # mrope
    if graph_meta.use_mrope:
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'].zero_()
            input_buffers['mrope_position_ids'][:, :num_tokens] = mrope_position_ids
        new_inputs['mrope_position_ids'] = input_buffers['mrope_position_ids']

    return new_inputs


def update_context_cudagraph(self, graph_meta, context):
    """Update step context with input buffers."""
    input_buffers = graph_meta.input_buffers
    context.block_offsets = input_buffers['block_offsets']
    context.kv_seqlens = input_buffers['kv_seqlens']
    context.kv_seqlens_cpu = input_buffers.get('kv_seqlens_cpu')
    context.q_start_loc = input_buffers['q_start_loc']
    context.q_seqlens = input_buffers['q_seqlens']
    context.cu_seqlens_q_cpu = input_buffers.get('cu_seqlens_q_cpu')
    context.kv_start_indices = input_buffers['kv_start_indices']
    context.moe_metadata.x_active_mask = input_buffers['x_active_mask']

    # ssm
    if graph_meta.is_ssm:
        context.state_offsets = input_buffers['state_ids']

    # mrope
    if graph_meta.use_mrope:
        context.mrope_position_ids = input_buffers['mrope_position_ids']


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def get_ascend_compatible_size(n: int):
    """Get ascend compatible size."""
    if n <= 16:
        n = next_power_of_2(n)
    elif n <= 256:
        n = (n + 15) & ~0xF
    else:
        n = (((n - 1) >> 8) + 1) << 8
    return n


@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size.

    Generate compatible sizes up to max_batches (not exceeding it), then add max_batches itself to ensure it can be
    handled.
    """
    ret = []
    batch_size = 1

    # Generate batch sizes and apply get_ascend_compatible_size
    # Only include sizes that do not exceed max_batches
    while batch_size <= max_batches:
        compatible_size = get_ascend_compatible_size(batch_size)
        if compatible_size > max_batches:
            break
        if not ret or compatible_size > ret[-1]:
            ret.append(compatible_size)
        batch_size = compatible_size + 1

    # Add max_batches itself to ensure it can be handled
    if max_batches not in ret:
        ret.append(max_batches)

    return ret


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False


class AscendSingleGraphRunner:
    """Cuda single graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        pool: Any,
        model_config: ModelConfig,
        device: torch.device,
        is_mla: bool,
    ):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config

        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
            mla_index_topk=getattr(model_config, 'mla_index_topk', None),
            is_ssm=len(model_config.states_shapes) > 0,
            use_mrope=model_config.use_mrope,
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.npu.NPUGraph = None
        self.is_mla = is_mla

    @record_function('capture_cudagraph')
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f"Capturing graph with meta: {self.meta}")
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.npu.current_stream()

        # warmup
        warmup_output = self.model(**padded_kwargs)
        warmup_buffers = self.model.make_output_buffers(warmup_output)

        aclgraph = torch.npu.NPUGraph()
        AscendGraphRunner.capturing = True
        try:
            with ExitStack():
                with torch.npu.graph(
                    aclgraph,
                    auto_dispatch_capture=True,
                    pool=self.pool,
                    stream=current_stream,
                ):
                    graph_output = self.model(**padded_kwargs)
        finally:
            AscendGraphRunner.capturing = False

        output_buffers = self.model.make_output_buffers(graph_output)
        self.meta.output_buffers = output_buffers
        self._graph = aclgraph
        final_output = self.model.get_outputs_cudagraph(warmup_buffers, **kwargs)
        return final_output

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """forward."""
        assert self._graph is not None
        self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        self._graph.replay()
        if not _is_sparse_attention(self.meta):
            if self.is_mla:
                cpu_update_input = [{'actual_seq_kvlen': self.meta.input_buffers['kv_seqlens_cpu']}]
            else:
                cpu_update_input = [{'actual_seq_lengths_kv': self.meta.input_buffers['kv_seqlens_cpu']}]
            self._graph.update(cpu_update_input=cpu_update_input)
        # torch.npu.synchronize()
        output_buffers = self.meta.output_buffers
        output = self.model.get_outputs_cudagraph(output_buffers, **kwargs)
        return output

    def reset(self):
        """Reset NPU graph and release its buffers."""
        if self._graph is not None:
            try:
                if hasattr(self._graph, 'reset'):
                    self._graph.reset()
            finally:
                self._graph = None

        if hasattr(self.meta, 'output_buffers') and isinstance(self.meta.output_buffers, dict):
            self.meta.output_buffers.clear()

    def __del__(self):
        """Best-effort cleanup when runner is GC-ed."""
        try:
            self.reset()
        except Exception:
            pass


class AscendGraphRunner(GraphRunner):
    """Cuda graph runner."""

    capturing = False

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        device: torch.device,
        is_mla: bool = False,
    ):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks
        self.enable_graph = self.check_enable_graph()
        self.graph_pool_handle = torch.npu.graph_pool_handle()
        self._runner_map: dict[Any, AscendSingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False
        self.is_mla = is_mla

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def _get_capture_tokens(self, batch_size: int):
        """Get capture tokens."""
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f"Unsupported batch_size={batch_size}"

    def get_graph_key(
        self,
        input_ids: torch.Tensor,
        attn_metadata: Any,
        **kwargs,
    ):
        """Get graph key."""
        context = get_step_ctx_manager().current_context()
        # Keep the graph key consistent across DP ranks (the captured graph holds the
        # MoE EP collective), mirroring the global gate used in support_cuda_graph.
        is_decoding = context.global_is_decoding()
        is_multi_token_decoding = attn_metadata.is_multi_token_decoding
        meta = self.get_meta()
        enable_microbatch = context.enable_microbatch

        if is_multi_token_decoding:
            q_seqlens = attn_metadata.q_seqlens
            max_q_seq_len = attn_metadata.max_q_seqlen
            batch_size = q_seqlens.size(0)
            if meta.padding_batch_size is None:
                new_batch_size = self._get_capture_tokens(batch_size)
            else:
                new_batch_size = self._get_capture_tokens(meta.padding_batch_size)
            return (
                new_batch_size,
                is_multi_token_decoding,
                enable_microbatch,
                max_q_seq_len,
            )

        num_tokens = input_ids.numel()
        if meta.padding_batch_size is None:
            new_num_tokens = self._get_capture_tokens(num_tokens)
        else:
            new_num_tokens = self._get_capture_tokens(meta.padding_batch_size)
        return (new_num_tokens, is_decoding, enable_microbatch, 1)

    def __call__(self, **kwargs):
        """call."""
        # Graph capture/replay holds the MoE EP collective; gate on the DP-global
        # decode state so every DP rank enters or skips the graph together.
        context = get_step_ctx_manager().current_context()
        enable_graph = context.global_is_decoding() and self.enable_graph(**kwargs)

        if not enable_graph:
            with record_function('forward_eager'):
                ret = self.model(**kwargs)
                return self.model.make_output_buffers(ret)

        graph_key = self.get_graph_key(**kwargs) + self.model.get_cudagraph_extra_key(**kwargs)
        max_batches = graph_key[0]
        is_decoding_or_multi_token_decoding = graph_key[1]
        max_q_seq_len = graph_key[3]
        max_tokens = max_batches * max_q_seq_len
        if graph_key not in self._runner_map:
            runner = AscendSingleGraphRunner(
                self.model,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding_or_multi_token_decoding,
                pool=self.graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
                is_mla=self.is_mla,
            )
            runner.capture(**kwargs)
            self._runner_map[graph_key] = runner
        else:
            runner = self._runner_map[graph_key]
        output = runner.forward(**kwargs)
        return output

    @record_function('prepare_inputs_for_generation')
    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""
        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self):
        """Remove all graphs and related resources to prevent hanging on
        exit."""
        super().reset()
        for _, runner in self._runner_map.items():
            try:
                runner.reset()
            except Exception as e:
                logger.warning(f"AscendGraphRunner.reset: runner.reset error: {e!r}")
        self._runner_map.clear()
        self.graph_pool_handle = None
        torch.npu.empty_cache()

    def __del__(self):
        """Best-effort cleanup when graph runner is GC-ed."""
        try:
            self.reset()
        except Exception:
            pass

    def update_inputs(self, inputs):
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.global_is_decoding()
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size

            batch_size = inputs.seq_length.size(0)
            query_len = inputs.input_ids.numel() // batch_size
            tp_size = self._get_capture_tokens(padding_batch_size) * query_len
            dp_meta.sync_tp_size(tp_size)
        return inputs

    def get_capture_batch_sizes(self) -> list[int]:
        """Capture batch sizes."""
        if self.cache_config.cudagraph_capture_batch_sizes is not None:
            return super().get_capture_batch_sizes()
        return _get_capture_batch_size_impl(self.cache_config.max_batches)
