# Copyright (c) OpenMMLab. All rights reserved.
import functools
import json
import os
import time
import weakref
from collections import deque
from typing import Any

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.backends.deepep_state import get_deepep_state
from lmdeploy.pytorch.backends.selector import get_backend
from lmdeploy.pytorch.config import (
    BackendConfig,
    CacheConfig,
    ModelConfig,
    normalize_cudagraph_capture_batch_sizes,
)
from lmdeploy.pytorch.envs import fake_capture
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.strategies.base import StrategyFactoryBase
from lmdeploy.utils import get_logger

from ..graph_runner import GraphRunner
from .attention import TritonAttentionMetadata

logger = get_logger('lmdeploy')


_CUDAGRAPH_RCA_RUNNERS: weakref.WeakSet = weakref.WeakSet()
_CUDAGRAPH_RCA_DUMPED = False
_CUDAGRAPH_RCA_FILE_WRITE_FAILED = False
_RCA_FATAL_PATTERNS = (
    'DeepEP error',
    'timeout (dispatch CPU)',
    'CPU recv timeout',
    'CUDA error',
    'device-side assert',
    'illegal memory access',
    'CUBLAS_STATUS_EXECUTION_FAILED',
)


def _rca_enabled():
    value = os.getenv('LMDEPLOY_CUDAGRAPH_RCA_TRACE', '0').lower()
    return value in ('1', 'true', 'yes', 'on')


def _rca_dump_on_reset_enabled():
    value = os.getenv('LMDEPLOY_CUDAGRAPH_RCA_DUMP_ON_RESET', '0').lower()
    return value in ('1', 'true', 'yes', 'on')


def _rca_record_limit():
    try:
        return min(max(int(os.getenv('LMDEPLOY_CUDAGRAPH_RCA_RECORDS', '1000')), 1), 1000)
    except ValueError:
        return 1000


def _safe_filename_part(value):
    value = str(value)
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in value)


def _rca_rank():
    for env_name in ('RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID'):
        value = os.getenv(env_name)
        if value is not None:
            return value
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return str(torch.distributed.get_rank())
    except Exception:  # pragma: no cover - debug helper only
        pass
    return 'unknown'


def _rca_local_rank():
    for env_name in ('LOCAL_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID'):
        value = os.getenv(env_name)
        if value is not None:
            return value
    return 'unknown'


def _rca_jsonl_path():
    log_file = os.getenv('LMDEPLOY_LOG_FILE')
    if not log_file:
        return None
    log_file = os.path.expanduser(log_file)
    rank = _safe_filename_part(_rca_rank())
    local_rank = _safe_filename_part(_rca_local_rank())
    return f'{log_file}.cudagraph_rca.rank{rank}.local{local_rank}.pid{os.getpid()}.jsonl'


def _safe_list(value, limit: int = 32):
    if value is None:
        return None
    items = list(value)
    if len(items) > limit:
        items = items[:limit] + [f'...({len(items) - limit} more)']
    return [_safe_value(item) for item in items]


def _safe_tensor_value(value: torch.Tensor | None):
    """Describe tensor metadata without reading tensor contents."""
    if value is None:
        return None
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _safe_value(value):
    if isinstance(value, torch.Tensor):
        return _safe_tensor_value(value)
    if isinstance(value, (list, tuple)):
        return _safe_list(value)
    return value


def _dp_meta_record(dp_meta):
    if dp_meta is None:
        return dict(
            dp_is_decoding=None,
            dp_batches=None,
            tp_sizes=None,
            moe_tp_sizes=None,
            dp_draft_num_tokens=None,
        )
    return dict(
        dp_is_decoding=getattr(dp_meta, 'dp_is_decoding', None),
        dp_batches=_safe_list(getattr(dp_meta, 'dp_batches', None)),
        tp_sizes=_safe_list(getattr(dp_meta, 'tp_sizes', None)),
        moe_tp_sizes=_safe_list(getattr(dp_meta, 'moe_tp_sizes', None)),
        dp_draft_num_tokens=_safe_list(getattr(dp_meta, 'dp_draft_num_tokens', None)),
    )


def _model_role(model: torch.nn.Module):
    cls_name = type(model).__name__
    module = type(model).__module__
    role = 'draft' if 'MTP' in cls_name or module.endswith('_mtp') else 'target'
    return role, cls_name


def _fatal_for_rca(exc: BaseException):
    message = repr(exc)
    return any(pattern in message for pattern in _RCA_FATAL_PATTERNS)


def _json_line(payload: dict[str, Any]):
    def json_default(value):
        if isinstance(value, torch.Tensor):
            return _safe_tensor_value(value)
        return str(value)

    return json.dumps(payload, sort_keys=True, separators=(',', ':'), default=json_default)


def _append_rca_jsonl(payloads: dict[str, Any] | list[dict[str, Any]]):
    global _CUDAGRAPH_RCA_FILE_WRITE_FAILED
    path = _rca_jsonl_path()
    if path is None:
        return None
    if isinstance(payloads, dict):
        payloads = [payloads]
    try:
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            for payload in payloads:
                f.write(_json_line(payload))
                f.write('\n')
    except Exception as exc:  # pragma: no cover - debug helper only
        if not _CUDAGRAPH_RCA_FILE_WRITE_FAILED:
            _CUDAGRAPH_RCA_FILE_WRITE_FAILED = True
            logger.error('[CUDAGRAPH_RCA_JSONL] failed to write %s: %r', path, exc)
    return path


def _dump_all_cudagraph_rca_records(reason: str):
    global _CUDAGRAPH_RCA_DUMPED
    if not _rca_enabled() or _CUDAGRAPH_RCA_DUMPED:
        return
    _CUDAGRAPH_RCA_DUMPED = True
    runners = list(_CUDAGRAPH_RCA_RUNNERS)
    dump_payload = dict(
        event='dump',
        reason=reason,
        runner_count=len(runners),
        pid=os.getpid(),
        rank=_rca_rank(),
        local_rank=_rca_local_rank(),
    )
    json_file = _append_rca_jsonl(dump_payload)
    logger.error(
        '[CUDAGRAPH_RCA_DUMP] %s',
        _json_line(dict(**dump_payload, json_file=json_file)),
    )
    for runner in runners:
        try:
            runner._dump_rca_records(reason)
        except Exception:  # pragma: no cover - best effort failure dump
            logger.exception('[CUDAGRAPH_RCA_DUMP] failed to dump runner records')


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


@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size."""
    ret = []
    batch_size = 1
    batch_step = 256
    # power of 2
    while batch_size <= min(batch_step, max_batches):
        ret.append(batch_size)
        batch_size *= 2

    # step
    ret += list(range(batch_size, max_batches + 1, batch_step))

    if max_batches != ret[-1]:
        ret.append(max_batches)
    return ret


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False


class CUDASingleGraphRunner:
    """Cuda single graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        decode_query_len: int,
        pool: tuple[int, int],
        model_config: ModelConfig,
        device: torch.device,
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
            use_mla_fp8_cache=getattr(self.model_config, 'use_mla_fp8_cache', False),
            use_flash_mla=getattr(self.model_config, 'use_flash_mla', False),
            mla_index_topk=getattr(self.model_config, 'mla_index_topk', None),
            use_fa3_decoding=(model_config.model_paradigm == 'ar_spec'
                              and not getattr(model_config, 'use_flash_mla', False)),
            is_ssm=len(model_config.states_shapes) > 0,
            use_mrope=model_config.use_mrope,
            block_size=model_config.block_size,
            decode_query_len=decode_query_len,
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.cuda.CUDAGraph = None
        self.USE_GRAPH = not fake_capture
        logger.info(f'Initialized CUDASingleGraphRunner with max_batches={max_batches}, max_tokens={max_tokens}, '
                    f'num_blocks={num_blocks}, is_decoding={is_decoding}, use_graph={self.USE_GRAPH}')

    @record_function('capture_cudagraph')
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f'Capturing graph with meta: {self.meta}')
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        # warmup
        warmup_output = self.model(**padded_kwargs)
        warmup_buffers = self.model.make_output_buffers(warmup_output)

        if self.USE_GRAPH:
            self._graph = torch.cuda.CUDAGraph()
            # unsafe kernel call in other thread might invalid the capture
            # so we set thread_safe capture mode here.
            with torch.cuda.graph(self._graph,
                                  pool=self.pool,
                                  stream=current_stream,
                                  capture_error_mode='thread_local'):
                output = self.model(**padded_kwargs)
        else:
            output = warmup_output

        output_buffers = self.model.make_output_buffers(output)
        self.meta.output_buffers = output_buffers
        output = self.model.get_outputs_cudagraph(warmup_buffers, **kwargs)
        return output

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """forward."""
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        if self.USE_GRAPH:
            assert self._graph is not None
            self._graph.replay()
            output_buffers = self.meta.output_buffers
        else:
            output = self.model(**padded_kwargs)
            output_buffers = self.model.make_output_buffers(output)
        output = self.model.get_outputs_cudagraph(output_buffers, **kwargs)
        return output

    def __del__(self):
        """del."""
        del self._graph


class CUDAGraphRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.num_blocks = cache_config.num_gpu_blocks

        # Speculative decoding on CUDA requires FlashAttention-3 (FA3),
        # unless the model uses FlashMLA (e.g., DeepSeek MTP) which handles
        # multi-token decoding queries natively.
        # FA3 is available on SM80+ (Ampere and above) GPUs with CUDA >= 12.3.
        # Without FA3, the Triton paged attention kernel cannot handle
        # multi-token decoding queries (max_q_seqlen > 1) used in spec decoding.
        if model_config.model_paradigm == 'ar_spec' and not getattr(model_config, 'use_flash_mla', False):
            from .attention import use_fa3
            if not use_fa3:
                sm = torch.cuda.get_device_capability()
                cuda_ver = torch.version.cuda or 'N/A'
                raise RuntimeError(
                    f'Speculative decoding on CUDA requires FlashAttention-3 (FA3), '
                    f'which needs SM80+ (Ampere and above) with CUDA >= 12.3 and '
                    f'flash-attn installed. Detected: SM{sm[0]}.{sm[1]}, CUDA {cuda_ver}. '
                    f'Please ensure your GPU meets SM80+, CUDA >= 12.3, and flash-attn '
                    f'is installed, or disable speculative decoding.')

        self.enable_graph = self.check_enable_graph()

        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: dict[Any, CUDASingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False

        # strategy factory
        build_ctx = model.ctx_mgr.build_ctx
        strategy_factory: StrategyFactoryBase = build_ctx.strategy_factory
        self.cudagraph_strategy = strategy_factory.build_cudagraph_strategy()
        self._rca_forward_records = deque(maxlen=_rca_record_limit())
        self._rca_forward_seq = 0
        self._rca_last_deepep_mode = 'none'
        self._rca_forward_total = 0
        self._rca_local_prefill_forward_count = 0
        self._rca_local_decode_forward_count = 0
        self._rca_global_prefill_forward_count = 0
        self._rca_global_decode_forward_count = 0
        self._rca_deepep_normal_forward_count = 0
        self._rca_deepep_low_latency_forward_count = 0
        self._rca_deepep_none_forward_count = 0
        _CUDAGRAPH_RCA_RUNNERS.add(self)

    def _append_rca_record(self, record: dict[str, Any]):
        if not _rca_enabled():
            return
        role, model_class = _model_role(self.model)
        self._rca_forward_seq += 1
        base = dict(
            seq=self._rca_forward_seq,
            timestamp=time.time(),
            pid=os.getpid(),
            rank=_rca_rank(),
            local_rank=_rca_local_rank(),
            runner_id=f'{id(self):x}',
            role=role,
            model_class=model_class,
        )
        base.update(record)
        self._rca_forward_records.append(base)

    def _update_rca_forward_counters(self, local_is_decoding: bool | None, global_is_decoding: bool | None,
                                     deepep_mode: str):
        self._rca_forward_total = getattr(self, '_rca_forward_total', 0) + 1
        self._rca_local_prefill_forward_count = getattr(self, '_rca_local_prefill_forward_count', 0)
        self._rca_local_decode_forward_count = getattr(self, '_rca_local_decode_forward_count', 0)
        self._rca_global_prefill_forward_count = getattr(self, '_rca_global_prefill_forward_count', 0)
        self._rca_global_decode_forward_count = getattr(self, '_rca_global_decode_forward_count', 0)
        self._rca_deepep_normal_forward_count = getattr(self, '_rca_deepep_normal_forward_count', 0)
        self._rca_deepep_low_latency_forward_count = getattr(self, '_rca_deepep_low_latency_forward_count', 0)
        self._rca_deepep_none_forward_count = getattr(self, '_rca_deepep_none_forward_count', 0)

        if local_is_decoding is True:
            self._rca_local_decode_forward_count += 1
        elif local_is_decoding is False:
            self._rca_local_prefill_forward_count += 1

        if global_is_decoding is True:
            self._rca_global_decode_forward_count += 1
        elif global_is_decoding is False:
            self._rca_global_prefill_forward_count += 1

        deepep_mode = str(deepep_mode).upper()
        if deepep_mode == 'NORMAL':
            self._rca_deepep_normal_forward_count += 1
        elif deepep_mode == 'LOW_LATENCY':
            self._rca_deepep_low_latency_forward_count += 1
        elif deepep_mode == 'NONE':
            self._rca_deepep_none_forward_count += 1

        return dict(
            total=self._rca_forward_total,
            local_prefill=self._rca_local_prefill_forward_count,
            local_decode=self._rca_local_decode_forward_count,
            global_prefill=self._rca_global_prefill_forward_count,
            global_decode=self._rca_global_decode_forward_count,
            deepep_normal=self._rca_deepep_normal_forward_count,
            deepep_low_latency=self._rca_deepep_low_latency_forward_count,
            deepep_none=self._rca_deepep_none_forward_count,
        )

    def _record_forward(
        self,
        context: StepContext,
        kwargs: dict[str, Any],
        *,
        graph_action: str,
        enable_graph: bool,
        graph_key: tuple | None = None,
        single_runner: CUDASingleGraphRunner | None = None,
    ):
        if not _rca_enabled():
            return
        input_ids = kwargs.get('input_ids')
        attn_metadata = kwargs.get('attn_metadata')
        q_seqlens = getattr(attn_metadata, 'q_seqlens', None)
        batch_size = q_seqlens.size(0) if q_seqlens is not None else None
        num_tokens = input_ids.numel() if isinstance(input_ids, torch.Tensor) else None
        query_len = None
        if batch_size and num_tokens is not None:
            query_len = num_tokens // batch_size

        try:
            global_is_decoding = context.global_is_decoding()
        except Exception as exc:  # pragma: no cover - debug helper only
            global_is_decoding = f'error:{type(exc).__name__}'
        local_is_decoding = getattr(context, 'is_decoding', None)
        deepep_mode = self._rca_last_deepep_mode
        forward_counters = self._update_rca_forward_counters(local_is_decoding, global_is_decoding, deepep_mode)

        self._append_rca_record(
            dict(
                kind='forward',
                local_is_decoding=local_is_decoding,
                global_is_decoding=global_is_decoding,
                forward_counters=forward_counters,
                deepep_mode=deepep_mode,
                num_tokens=num_tokens,
                batch_size=batch_size,
                query_len=query_len,
                max_kv_seqlen=getattr(context, 'max_kv_seqlen', None),
                sum_kv_seqlen=getattr(context, 'sum_kv_seqlen', None),
                is_dummy=getattr(context, 'is_dummy', None),
                is_chunk=getattr(context, 'is_chunk', None),
                is_first_chunk=getattr(context, 'is_first_chunk', None),
                is_last_chunk=getattr(context, 'is_last_chunk', None),
                is_chunk_multimodal=getattr(context, 'is_chunk_multimodal', None),
                dp_meta=_dp_meta_record(getattr(context, 'dp_meta', None)),
                enable_graph=enable_graph,
                graph_key=_safe_value(graph_key),
                graph_action=graph_action,
                single_runner_id=f'{id(single_runner):x}' if single_runner is not None else None,
                runner_map_size=len(self._runner_map),
                num_gpu_blocks=self.num_blocks,
            ))

    def _record_reset(self, kind: str, deepep_destroy_called: bool = False):
        if not _rca_enabled():
            return
        self._append_rca_record(
            dict(
                kind=kind,
                runner_map_size=len(self._runner_map),
                graph_keys=[_safe_value(key) for key in self._runner_map.keys()],
                num_gpu_blocks=self.num_blocks,
                deepep_destroy_called=deepep_destroy_called,
            ))

    def _dump_rca_records(self, reason: str):
        role, model_class = _model_role(self.model)
        runner_payload = dict(
            event='runner',
            reason=reason,
            pid=os.getpid(),
            rank=_rca_rank(),
            local_rank=_rca_local_rank(),
            runner_id=f'{id(self):x}',
            role=role,
            model_class=model_class,
            record_count=len(self._rca_forward_records),
        )
        json_file = _append_rca_jsonl(
            [runner_payload] + [dict(event='record', **record) for record in self._rca_forward_records])
        logger.error(
            '[CUDAGRAPH_RCA_RUNNER] %s',
            _json_line(dict(**runner_payload, json_file=json_file)),
        )
        for record in self._rca_forward_records:
            logger.error('[CUDAGRAPH_RCA_RECORD] %s', _json_line(record))

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def _try_compile_model_once(self):
        if self.has_try_compile_model:
            return

        # TODO: recovery it when torch.compile is stable (should be add a flag to enable it?)
        # if hasattr(self.model, 'compile_model'):
        #     method = getattr(self.model, 'compile_model')
        #     method()

        self.has_try_compile_model = True

    def _get_capture_tokens(self, batch_size: int):
        """Get capture tokens."""
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f'Unsupported batch_size={batch_size}'

    def get_graph_key(self, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values: list,
                      attn_metadata: TritonAttentionMetadata, inputs_embeds: torch.Tensor, **kwargs):
        """Get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.global_is_decoding()
        batch_size = attn_metadata.q_seqlens.size(0)
        meta = self.get_meta()
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch
        query_len = input_ids.size(1) // batch_size
        if meta.padding_batch_size is None:
            batch_size = self._get_capture_tokens(batch_size)
        else:
            batch_size = self._get_capture_tokens(meta.padding_batch_size)
        return (batch_size, is_decoding, enable_microbatch, query_len)

    def _prepare_inputs(self, **kwargs):
        """Prepare inputs."""
        assert 'attn_metadata' in kwargs, 'attn_metadata is required for cudagraph.'
        attn_metadata: TritonAttentionMetadata = kwargs['attn_metadata']
        if not attn_metadata.block_offsets.dtype == torch.int32:
            attn_metadata.block_offsets = attn_metadata.block_offsets.to(torch.int32)
        return kwargs

    def _get_max_tokens(self, graph_key: tuple, input_ids: torch.Tensor, q_seqlens: torch.Tensor):
        max_batches = graph_key[0]
        is_decoding = graph_key[1]
        assert is_decoding
        origin_batch_size = q_seqlens.size(0)
        num_tokens = input_ids.size(1)
        return self.cudagraph_strategy.get_max_tokens(max_batches, origin_batch_size, num_tokens)

    def __call__(self, **kwargs):
        """call."""
        try:
            if not self.backend_config.eager_mode and get_backend().get_name() == 'cuda':
                self._try_compile_model_once()

            kwargs = self._prepare_inputs(**kwargs)
            context = self.ctx_mgr.current_context()
            enable_graph = context.global_is_decoding() and self.enable_graph(**kwargs)

            if not enable_graph:
                self._record_forward(context, kwargs, graph_action='eager', enable_graph=enable_graph)
                with record_function('forward_eager'):
                    output = self.model(**kwargs)
                    return self.model.make_output_buffers(output)

            graph_key = self.get_graph_key(**kwargs)
            max_batches = graph_key[0]
            is_decoding = graph_key[1]
            decode_query_len = graph_key[3]
            if graph_key not in self._runner_map:
                max_tokens = self._get_max_tokens(graph_key, kwargs['input_ids'], kwargs['attn_metadata'].q_seqlens)
                runner = CUDASingleGraphRunner(
                    self.model,
                    max_batches=max_batches,
                    max_tokens=max_tokens,
                    num_blocks=self.num_blocks,
                    is_decoding=is_decoding,
                    decode_query_len=decode_query_len,
                    pool=self.graph_pool_handle,
                    model_config=self.model_config,
                    device=self.device,
                )
                self._record_forward(
                    context,
                    kwargs,
                    graph_action='capture',
                    enable_graph=enable_graph,
                    graph_key=graph_key,
                    single_runner=runner,
                )
                output = runner.capture(**kwargs)
                self._runner_map[graph_key] = runner
                # SSM would update the state in capture(warmup), replay the graph will leads unexpected state update.
                return output
            else:
                runner = self._runner_map[graph_key]
                self._record_forward(
                    context,
                    kwargs,
                    graph_action='replay',
                    enable_graph=enable_graph,
                    graph_key=graph_key,
                    single_runner=runner,
                )
                output = runner.forward(**kwargs)
                return output
        except Exception as exc:
            if _fatal_for_rca(exc):
                _dump_all_cudagraph_rca_records(repr(exc))
            raise

    @record_function('prepare_inputs_for_generation')
    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""

        if get_deepep_state().enabled():
            from dlblas.layers.moe.token_dispatcher import DeepEPBuffer, DeepEPMode
            deepep_mode = DeepEPMode.LOW_LATENCY if context.global_is_decoding() else DeepEPMode.NORMAL
            self._rca_last_deepep_mode = getattr(deepep_mode, 'name', str(deepep_mode))
            DeepEPBuffer.set_deepep_mode(deepep_mode)
        else:
            self._rca_last_deepep_mode = 'none'

        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self):
        """Remove all graphs to prevent hanging on exit."""
        super().reset()
        deepep_destroy_called = False
        self._record_reset('reset_start')
        self._runner_map.clear()
        if get_deepep_state().enabled():
            from dlblas.layers.moe.token_dispatcher import DeepEPBuffer

            if hasattr(DeepEPBuffer, 'destroy'):
                from torch import distributed as dist

                DeepEPBuffer.destroy()
                deepep_destroy_called = True
                dist.barrier()
        self._record_reset('reset_done', deepep_destroy_called=deepep_destroy_called)
        if _rca_dump_on_reset_enabled():
            self._dump_rca_records('reset')

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
            self.cache_config.cudagraph_capture_batch_sizes = normalize_cudagraph_capture_batch_sizes(
                self.cache_config.cudagraph_capture_batch_sizes, self.cache_config.max_batches)
            return self.cache_config.cudagraph_capture_batch_sizes
        return _get_capture_batch_size_impl(self.cache_config.max_batches)
