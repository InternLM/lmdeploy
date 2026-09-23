# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from multiprocessing.reduction import ForkingPickler
import os
from os import getenv
from typing import Any

import numpy as np
import pybase64
import torch
import torch.distributed as dist
from torch.profiler import record_function

# V4_STEP_TIME diagnostic: running averages of the decode eager-build (host
# dispatch, NPU-idle gap source) vs the sync'd forward/replay wall.
_V4_STEP_CNT = {}
# V4_OPCOUNT diagnostic: one-shot, set True after the single profiled step.
_V4_OPCOUNT_DONE = [False]

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.config import (BackendConfig, CacheConfig, MiscConfig, ModelConfig, SpecDecodeConfig,
                                     TPMode)
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine
from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager
from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor, SamplingInputs
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta, step_ctx_manager
from lmdeploy.pytorch.models.patch import BuildModelContext, add_adapters, build_patched_model, update_custom_module_map
from lmdeploy.pytorch.spec_decode import build_spec_agent
from lmdeploy.pytorch.strategies import build_strategy_factory
from lmdeploy.pytorch.strategies.base.model_agent import ExtraInputs, ExtraOutputs, StoppingCriteria
from lmdeploy.pytorch.utils import get_gpu_memory, monkey_patch_hf_modules_cache, wait_for_async_tasks
from lmdeploy.pytorch.weight_loader.model_weight_loader import ModelWeightLoader, load_model_weights
from lmdeploy.serve.openai.protocol import (
    DestroyWeightsUpdateGroupRequest,
    InitWeightsUpdateGroupRequest,
    UpdateParamsRequest,
    UpdateWeightsFromDistributedRequest,
)
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import FlattenedTensorBucket, FlattenedTensorMetadata, get_logger, init_custom_process_group

from .dp_utils import DistGatherScalar, DPForwardMeta, GatheredDPForwardMeta
from .inputs_maker import build_inputs_maker
from .profiler import AgentProfiler
from .scoring import compute_input_ce_loss

logger = get_logger('lmdeploy')

_H2D_TRANSFER_KEY = '_h2d_transfer'


@dataclass
class _H2DTransfer:
    event: torch.cuda.Event
    refs: dict[str, Any]


@dataclass
class SleepWakeupState:
    to_sleep: asyncio.Event = field(default_factory=asyncio.Event)
    to_wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    is_sleeping: bool = False


@dataclass
class BatchedLogProbs:
    vals: torch.Tensor
    indices: torch.Tensor

    def to_cpu(self):
        """To cpu."""
        return BatchedLogProbs(vals=self.vals.cpu().detach(), indices=self.indices.cpu().detach())

    def to_numpy(self):
        """To numpy."""
        if self.vals.dtype == torch.bfloat16:
            np_vals = self.vals
        else:
            np_vals = self.vals.detach().numpy()
        return BatchedLogProbs(vals=np_vals, indices=self.indices.detach().numpy())

    def to_tensor(self):
        """To tensor."""
        if isinstance(self.vals, torch.Tensor):
            vals = self.vals
        else:
            vals = torch.from_numpy(self.vals)
        return BatchedLogProbs(vals=vals, indices=torch.from_numpy(self.indices))


@dataclass
class BatchedOutputs:
    next_token_ids: torch.Tensor
    stopped: torch.Tensor
    stop_pos: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    model_metas: list[dict[str, Any]] = None
    logprobs: BatchedLogProbs | None = None
    new_token_timestamp: int = 0
    extra_outputs: ExtraOutputs | None = None
    all_routed_experts: torch.Tensor | None = None
    ce_loss: torch.Tensor | None = None

    def to_cpu(self):
        """To cpu."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach()
            elif hasattr(v, 'to_cpu'):
                v = v.to_cpu()
            out[k] = v
        return BatchedOutputs(**out)

    def to_numpy(self):
        """To numpy."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor) and v.dtype != torch.bfloat16:
                v = v.detach().numpy()
            elif hasattr(v, 'to_numpy'):
                v = v.to_numpy()
            out[k] = v
        return BatchedOutputs(**out)

    def to_tensor(self):
        """To tensor."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif hasattr(v, 'to_tensor'):
                v = v.to_tensor()
            out[k] = v
        return BatchedOutputs(**out)


def msg_with_rank(rank: int, msg: str):
    """Return message with rank."""
    return f'rank[{rank}] - {msg}'


def cache_swapping(cache_engine: CacheEngine, swap_in_map: dict, swap_out_map: dict):
    """Perform cache swapping."""
    issued_cache_op = False
    swap_in_map = swap_in_map or dict()
    swap_out_map = swap_out_map or dict()
    if len(swap_in_map) > 0:
        cache_engine.swap_in(swap_in_map)
        issued_cache_op = True
    if len(swap_out_map) > 0:
        cache_engine.swap_out(swap_out_map)
        issued_cache_op = True

    if issued_cache_op:
        cache_engine.events.wait()


@torch.inference_mode()
def model_forward(
    model: torch.nn.Module,
    inputs: ModelInputs,
    cache_engine: CacheEngine,
    state_cache_engine: StateCacheEngine,
    stream: torch.cuda.Stream = None,
):
    """Perform model forward."""
    stream = stream or torch.cuda.current_stream()
    import os as _os, time as _time
    _step_time = _os.environ.get('V4_STEP_TIME', '0') == '1'
    _is_dec = getattr(inputs, 'is_decoding', None)
    if _step_time:
        import torch.distributed as _d
        _rk = (_d.get_rank() if (_d.is_available()
                                 and _d.is_initialized()) else 0)
    with torch.cuda.stream(stream), step_ctx_manager(model.ctx_mgr):
        # forward
        ctx_mgr = model.ctx_mgr
        if _step_time and _is_dec and (_rk == 0):
            _B = _time.perf_counter()
        context = ctx_mgr.build_context(
            inputs=inputs,
            model_config=cache_engine.model_config,
            cache_config=cache_engine.cache_config,
            kv_caches=cache_engine.gpu_cache,
            state_caches=state_cache_engine.state_caches,
            kv_quant_policy=cache_engine.cache_config.quant_policy,
        )

        with ctx_mgr.context(context):
            if (not inputs.is_dummy and inputs.state_offsets is not None
                    and inputs.state_prefix_cache_offsets is not None):
                # Restore frozen SSM prefix state into this request's runtime
                # slot on the forward stream.  The input maker already
                # compacted valid src/dst pairs on CPU, so no CUDA boolean
                # indexing/nonzero synchronization is needed here.
                state_cache_engine.copy_caches(inputs.state_prefix_cache_offsets,
                                               inputs.state_prefix_cache_dst_offsets)

            model_metas = model.update_model_metas(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            input_dict = model.prepare_inputs_for_generation(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            if _step_time and _is_dec and (_rk == 0):
                _C = _time.perf_counter()
            # V4_OPCOUNT: one-shot op counter over a single PREFILL forward
            # (eager -- NOT a decode graph replay, which would show ~1 op). The
            # captured decode graph replays the same op set, so eager prefill
            # op counts attribute the ~912 cast / ~303 mul kernels baked into
            # the decode burst. Uses a pure-Python TorchFunctionMode counter
            # (torch_npu patches BOTH torch.profiler and torch.autograd.profiler
            # into msprof, which lacks key_averages + hangs on export; this
            # avoids the C++ profiler entirely). Fires on the first prefill
            # (rank0); prints top-40 ops + cast/mul subset to worker .out.
            _opc = _os.environ.get('V4_OPCOUNT', '0') == '1'
            _opc_mode = None
            if (_opc and _is_dec is False and (_rk == 0)
                    and not _V4_OPCOUNT_DONE[0]):
                from collections import Counter as _Counter
                import torch.overrides as _ov

                class _OpCounter(_ov.TorchFunctionMode):
                    def __init__(self):
                        super().__init__()
                        self.counts = _Counter()

                    def __torch_function__(self, func, types,
                                           args=(), kwargs=None):
                        self.counts[func.__name__] += 1
                        return func(*args, **(kwargs or {}))

                _opc_mode = _OpCounter()
                _opc_mode.__enter__()
            # V4_PRE_REPLAY_SYNC: stream-specific sync right before the decode
            # graph replay. Mirrors vllm-ascend acl_graph.py:243-249: the per-step
            # graph param update (_graph.update(actual_seq_lengths_kv), done after
            # each replay in dlinfer's torch_npu_update path) is a CPU-side write
            # whose record event for step i can overtake step i-1's still-running
            # replay -- corrupting the KV-seqlen the next replay reads -> 507018.
            # vllm-ascend inserts `torch.npu.current_stream().synchronize()` before
            # replay() to enforce update(i-1) completes before replay(i). We use
            # the STREAM sync (NOT device-wide torch.npu.synchronize(), which
            # cross-syncs HCCL's stream and perturbs collective timing across
            # ranks -- that caused the non-monotonic 47->0 regression observed
            # with the device-wide variant). Prefill has no replay -> skip.
            if _os.environ.get('V4_PRE_REPLAY_SYNC', '0') == '1' and _is_dec:
                torch.npu.current_stream().synchronize()
            output = model(**input_dict)
            if _opc_mode is not None:
                _opc_mode.__exit__(None, None, None)
                _V4_OPCOUNT_DONE[0] = True
                try:
                    _kas = _opc_mode.counts
                    _tot = sum(_kas.values())
                    _top = sorted(_kas.items(), key=lambda kv: kv[1],
                                  reverse=True)
                    print(f'[V4-OPC] prefill total_torch_ops={_tot}',
                          flush=True)
                    for _n, _c in _top[:40]:
                        print(f'[V4-OPC] {_c:5d}  {_n}', flush=True)
                    _sub = [(n, c) for n, c in _top
                            if any(s in n for s in
                                   ('to', 'mul', 'copy_', 'clone', 'type_',
                                    'view', 'reshape', 'cat', 'expand',
                                    'contiguous'))]
                    print('[V4-OPC] --- cast/view/copy subset ---', flush=True)
                    for _n, _c in _sub:
                        print(f'[V4-OPC] {_c:5d}  {_n}', flush=True)
                except Exception as _e:
                    print(f'[V4-OPC] counter table failed: {_e!r}',
                          flush=True)
            if _step_time and _is_dec and (_rk == 0):
                torch.npu.synchronize()
                _D = _time.perf_counter()
                # _V4_STEP_CNT: ROLLING 20-step window (not cumulative average --
                # the cumulative avg masked regime changes, e.g. new-request
                # warmup pulling the all-time avg up). build=host dispatch of
                # build_context+prepare (NPU-idle gap source); fwd=NPU replay
                # wall (launch+execute, sync'd).
                _g = _V4_STEP_CNT
                _g['n'] = _g.get('n', 0) + 1
                if _g['n'] > 3:   # skip first 3 warmup steps
                    _win = _g.setdefault('win', [])
                    _win.append((_C - _B, _D - _C))
                    if len(_win) > 20:
                        _win.pop(0)
                    if _g['n'] % 20 == 0:
                        _nb = sum(x[0] for x in _win) / len(_win)
                        _nf = sum(x[1] for x in _win) / len(_win)
                        _xc = ''
                        try:
                            from lmdeploy.pytorch.backends.dlinfer\
                                .ascend.v4_dsa import sas_xcache_stats, \
                                qli_xcache_stats
                            _h, _m = sas_xcache_stats()
                            _qh, _qm = qli_xcache_stats()
                            _xc = (f' sas_xc hit={_h} miss={_m}'
                                   f' qli_xc hit={_qh} miss={_qm}')
                        except Exception:
                            pass
                        print(f'[V4-STEP] step={_g["n"]} '
                              f'build_host={1000*_nb:.1f}ms '
                              f'fwd_npu={1000*_nf:.1f}ms '
                              f'sum={1000*(_nb+_nf):.1f}ms{_xc}',
                              flush=True)
            if not isinstance(output, dict):
                output = dict(hidden_states=output)
            # InternVL-3.5-Flash will change the seqlen, model_metas during forward
            if getattr(context, 'is_model_meta_updated', False):
                model_metas = context.model_metas
            if (not inputs.is_dummy and inputs.state_offsets is not None
                    and inputs.state_prefix_cache_save_offsets is not None):
                # Save the post-forward runtime state into reserved checkpoint
                # slots.  The scheduler publishes these slots only after the
                # executor output boundary confirms the copy was enqueued.
                state_cache_engine.copy_caches(inputs.state_prefix_cache_save_src_offsets,
                                               inputs.state_prefix_cache_save_offsets)
            output['model_metas'] = model_metas
            output['seq_length'] = context.q_seqlens[:len(inputs.seq_length)]
            # for draft model reuse
            output['position_ids'] = context.position_ids
            return output


# ---------------------------------------------------------------------------
# Ascend profiling (torch_npu.profiler == msprof for PyTorch).
# Activated only on rank 0, only when V4_PROFILE=1, and only between two
# sentinel files so profiling can be scoped to exactly the measured workload
# (after graph warmup/capture, around the real requests) without disturbing
# capture. Produces a Chrome-format trace.json via tensorboard_trace_handler.
# ---------------------------------------------------------------------------
_V4_PROF = None
_V4_PROF_N = 0
_V4_PROF_MAX = int(os.environ.get('V4_PROFILE_STEPS', '400'))
_V4_PROF_START = os.environ.get('V4_PROFILE_START', '/tmp/v4_prof_start')
_V4_PROF_STOP = os.environ.get('V4_PROFILE_STOP', '/tmp/v4_prof_stop')


def _v4_prof_maybe_start():
    """Start the NPU profiler on the first forward after the start sentinel
    appears. No-op unless V4_PROFILE=1."""
    global _V4_PROF, _V4_PROF_N
    if _V4_PROF is not None:
        return
    if os.environ.get('V4_PROFILE') != '1':
        return
    if not os.path.exists(_V4_PROF_START):
        return
    import torch_npu.profiler as _P
    out_dir = os.environ.get('V4_PROFILE_DIR', '/deeplink/swang/claude_kfj/profile_out')
    os.makedirs(out_dir, exist_ok=True)
    _V4_PROF = _P.profile(
        activities=[_P.ProfilerActivity.CPU, _P.ProfilerActivity.NPU],
        on_trace_ready=_P.tensorboard_trace_handler(out_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    _V4_PROF.start()
    _V4_PROF_N = 0
    logger.info(f'[V4-PROF] rank0 profiler started -> {out_dir}')


def _v4_prof_maybe_step():
    """Advance the profiler; stop + export when the stop sentinel appears or
    the step cap is reached."""
    global _V4_PROF, _V4_PROF_N
    if _V4_PROF is None:
        return
    _V4_PROF_N += 1
    done = os.path.exists(_V4_PROF_STOP) or _V4_PROF_N >= _V4_PROF_MAX
    if done:
        _V4_PROF.stop()
        logger.info(f'[V4-PROF] profiler stopped after {_V4_PROF_N} forwards; '
                    f'trace exported to {os.environ.get("V4_PROFILE_DIR", "/deeplink/swang/claude_kfj/profile_out")}')
        _V4_PROF = None
        for _p in (_V4_PROF_START, _V4_PROF_STOP):
            try:
                os.remove(_p)
            except OSError:
                pass


def _try_to_cuda(val, non_blocking: bool = False):
    if val is None:
        return val
    elif isinstance(val, torch.Tensor):
        return val.cuda(non_blocking=non_blocking)
    elif hasattr(val, 'to_device'):
        return val.to_device('cuda', non_blocking=non_blocking)
    else:
        raise RuntimeError(f'Can not cast {type(val)} to cuda.')


SwapMap = dict[int, int]


class BaseModelAgent:
    """Base model agent.

    load model on local gpu

    Args:
        model_path (str): The hugging face model path.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        trust_remote_code (bool): Trust remote code
    """

    def __init__(
        self,
        model_path: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        misc_config: MiscConfig,
        dist_ctx: DistContext,
        device_ctx: DeviceContext,
        adapters: dict[str, str] = None,
        specdecode_config: SpecDecodeConfig = None,
        trust_remote_code: bool = False
    ):

        self.model_config = model_config
        self.cache_config = cache_config
        # use raw tokenizer
        if dist_ctx.dist_config.world_size > 1:
            monkey_patch_hf_modules_cache()
        self.tokenizer = Tokenizer(model_path, trust_remote_code=trust_remote_code).model.model

        # asyncio
        self._pre_in_que = None
        self._in_que = None
        self._out_que = None
        self._background_task = None
        self._preprocess_task = None
        self._pending_h2d_transfers = deque()
        self.tasks = set()

        # cuda stream
        self.stream = torch.cuda.Stream()
        self.out_stream = torch.cuda.Stream()
        self.cache_stream = torch.cuda.Stream()

        self.dist_ctx = dist_ctx
        self.device_ctx = device_ctx

        device = 'cuda'
        self.backend_config = backend_config
        self.misc_config = misc_config
        self.dist_config = dist_ctx.dist_config
        rank = dist_ctx.rank

        self.model_path = model_path
        self.adapters = adapters
        self.device = device
        self.rank = rank

        tp = self.dist_config.tp
        world_size = self.dist_config.world_size
        self.tp = tp
        self.world_size = world_size
        self.need_output = rank % self.dist_config.attn_tp == 0

        self.patched_model = None
        self.cache_engine = None
        self.state_cache_engine = None
        self.profiler: AgentProfiler = None
        try:
            self.guided_decoding_manager = GuidedDecodingManager(self.tokenizer, model_config.vocab_size)
        except ValueError as e:
            logger.warning(f'Failed to create GuidedManager for tokenizer {type(self.tokenizer)}: {e}')
            self.guided_decoding_manager = None

        # update_params_ipc_buffer
        self._update_params_ipc_tensor: torch.Tensor | None = None
        self._update_params_ipc_event: torch.cuda.Event | None = None

        # disaggregated weight-update process groups, keyed by group_name
        self._model_update_group: dict[str, dist.ProcessGroup] = {}

        # microbatch
        self.enable_microbatch = self.dist_config.enable_microbatch
        self.enable_microbatch_prefill_batchsize_threshold = \
            int(getenv('ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD', 2))
        self.enable_microbatch_prefill_token_threshold = \
            int(getenv('ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD', 2))
        self.enable_microbatch_decode_batchsize_threshold = \
            int(getenv('ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD', 2))

        # strategy
        self.strategy_factory = build_strategy_factory(model_config, misc_config, specdecode_config=specdecode_config)
        self.inputs_strategy = self.strategy_factory.build_model_inputs_strategy()
        self.agent_strategy = self.strategy_factory.build_model_agent_strategy()
        self.sampling_strategy = self.strategy_factory.build_sampling_strategy()

        # spec decoding
        self.spec_agent = build_spec_agent(specdecode_config,
                                           backend_config,
                                           dist_ctx,
                                           self.inputs_strategy,
                                           self.agent_strategy,
                                           misc_config=misc_config,
                                           device=device)
        # sleep wakeup state
        self.state: SleepWakeupState = SleepWakeupState()

        # decoding inputs
        self.step_inputs = self.strategy_factory.build_step_inputs()

        # long context
        self._prev_chunk_output: dict = None
        # chunked-prefill ppl: last logit row of the previous chunk, used to score the cross-chunk boundary token
        self._prev_chunk_last_logit: torch.Tensor | None = None

        # make dummy meta
        self.make_dummy_meta = self.inputs_strategy.create_make_dummy_meta(model_config)

    @contextmanager
    def all_context(self):
        device_mgr = get_device_manager()
        dist_mgr = get_dist_manager()
        with device_mgr.context(self.device_ctx), dist_mgr.context(self.dist_ctx), torch.inference_mode():
            yield

    def set_cache_config(self, cache_config: CacheConfig, spec_cache_config: CacheConfig = None):
        """Set all cache config."""
        self.cache_config = cache_config
        self.spec_agent.set_cache_config(spec_cache_config)

    def set_model_config(self, model_config: ModelConfig, spec_model_config: ModelConfig | None = None):
        """Set model config."""
        self.model_config = model_config
        self.spec_agent.set_model_config(spec_model_config)
        self.make_dummy_meta = self.inputs_strategy.create_make_dummy_meta(model_config)

    def get_free_mem(self):
        """Gather available memory."""
        with self.all_context():
            torch.cuda.empty_cache()
            gpu_mem_physical_free, _ = get_gpu_memory()
            return gpu_mem_physical_free

    def warmup(self):
        """warmup."""
        from lmdeploy.pytorch.envs import skip_warmup
        if skip_warmup:
            if self.rank == 0:
                logger.warning('Engine warmup is skipped. Set LMDEPLOY_SKIP_WARMUP=0 to enable warmup.')
            return
        warmup_start = time.perf_counter()
        if self.rank == 0:
            logger.info('Starting engine warmup. This may take a while...')

        with self.all_context(), torch.cuda.stream(self.stream):
            max_batches = self.cache_config.max_batches
            world_size = self.dist_config.world_size

            num_tokens = max_batches
            dp = self.dist_config.dp
            # With an asymmetric topology (attn_tp < mlp_tp, e.g. V4's grouped
            # o-proj forces attn_tp<=o_groups while EP needs the full width),
            # mlp/moe linears run in DP_TP mode and need dp_meta even when
            # dp==1 — the gather group (pairs of ranks sharing an attn slot)
            # still does an all_gather/reduce_scatter that requires tp_sizes.
            # The start barrier stays dp>1-only (it syncs *across* DP groups,
            # of which there is exactly one when dp==1).
            need_dp_meta = (dp > 1
                            or self.dist_config.mlp_tp_mode == TPMode.DP_TP
                            or self.dist_config.moe_tp_mode == TPMode.DP_TP)

            if dp > 1:
                # make sure warmup started together
                group = self.dist_ctx.cpu_group
                dist.barrier(group=group)

            # warmup prefill. The dummy must mirror the RUNTIME per-rank batch,
            # which internal DP splits across DP groups: at dp=2 a max_batches=30
            # engine batch gives each rank max_batches//dp=15 seqs. Feeding the
            # full max_batches to every rank (the old uniform
            # make_dummy(max_batches) + build_dp_meta([n]*world_size)) makes the
            # warmup forward see max_batches reqs per rank, but the V4 recurrent
            # state pool is sized for the per-rank ceiling (max_batches//dp), so
            # the warmup's state_bt (max_batches rows, values up to
            # max_batches*m_cr) over-indexes the pool by 2x -> a silent OOB that
            # corrupts adjacent GPU memory and later surfaces as a runtime
            # aicore MTE fault in the Compressor kernel. Halve the warmup dummy
            # to the per-rank ceiling so the warmup path matches runtime.
            _warmup_prefill_batch = (max_batches // dp
                                     if dp > 1 else max_batches)
            inputs = self.inputs_strategy.make_dummy(_warmup_prefill_batch,
                                                     is_decoding=False,
                                                     device='cuda',
                                                     vocab_size=self.model_config.vocab_size,
                                                     meta=self.make_dummy_meta)
            if need_dp_meta:
                num_tokens = inputs.input_ids.numel()
                inputs.build_dp_meta([num_tokens] * world_size)
            logger.debug('Warmup prefill start.')
            self._forward_impl(inputs)
            torch.cuda.synchronize()
            logger.debug('Warmup prefill done.')

            # warmup decoding(with cuda graph)
            capture_batch_sizes = self.patched_model.get_capture_batch_sizes()
            capture_batch_sizes = sorted(capture_batch_sizes, reverse=True)
            if self.cache_config.role == EngineRole.Prefill:
                # do not warmup decoding for prefill engine
                capture_batch_sizes = []
            # FULL-graph C1 step2b: mark the decode-graph-capture warmup steps
            # so build_v4_dsa_inputs inflates the SAS op's seqused_kv to the
            # session_len ceiling (vllm-ascend captures with
            # SEQ_LEN_WITH_MAX_PA_WORKSPACE=6144 to max-size the aclnn op's
            # internal temps; dlinfer captures at kv_len=1 -> OOB at 1st
            # replay). build_context runs BEFORE graph_runner.capture flips
            # AscendGraphRunner.capturing, so that flag can't gate the pre-step
            # -- this env is the only signal available at pre-step time. Set
            # only around the decode-capture loop (graph capture is
            # decode-only); cleared before the draft-model warmup below.
            os.environ['V4_GRAPH_CAPTURING'] = '1'
            for num_tokens in capture_batch_sizes:
                inputs = self.inputs_strategy.make_dummy(num_tokens,
                                                         is_decoding=True,
                                                         device='cuda',
                                                         vocab_size=self.model_config.vocab_size,
                                                         meta=self.make_dummy_meta)
                if need_dp_meta:
                    num_tokens = inputs.input_ids.numel()
                    inputs.build_dp_meta([num_tokens] * world_size)
                    # The warmup decode is a globally-uniform decode step (all
                    # DP ranks decode the same dummy is_decoding=True), so the
                    # captured graph's support_cuda_graph gate -- which reads
                    # dp_meta.dp_is_decoding via context.global_is_decoding() --
                    # must see True here, else dp>1 graph capture is skipped.
                    # _prepare_dp_v1 sets this for real steps; warmup bypasses
                    # that path, so set it explicitly.
                    inputs.dp_meta.dp_is_decoding = True
                logger.debug(f'Warmup decoding num_tokens={num_tokens} start.')
                self._forward_impl(inputs)
                torch.cuda.synchronize()
                logger.debug(f'Warmup decoding num_tokens={num_tokens} done.')
            os.environ['V4_GRAPH_CAPTURING'] = '0'

            # warmup draft model
            self.spec_agent.warmup(max_batches, self.model_config)
        elapsed_seconds = time.perf_counter() - warmup_start
        if self.rank == 0:
            logger.info(f'Engine warmup completed in {elapsed_seconds:.2f} seconds.')

    def _slice_outs(self, inputs: torch.Tensor, seq_length: torch.LongTensor):
        """Slice outputs."""
        return self.agent_strategy.slice_outputs(inputs, seq_length)

    def _postprocess_forward_output(self, output: dict, inputs: ModelInputs):
        """Post process forward output."""
        hidden_states = output['hidden_states']
        seq_length = output.get('seq_length', inputs.seq_length)
        hidden_states = self._slice_outs(hidden_states[0], seq_length)[None]
        output['hidden_states'] = hidden_states
        return output

    async def _async_model_forward(
        self,
        inputs: ModelInputs,
        return_logits: bool,
    ):
        """Model forward."""
        ret = await self.async_forward(inputs)

        if not return_logits:
            ret = self._postprocess_forward_output(ret, inputs)

        hidden_states, ret = self.spec_agent.update_main_model_outputs(ret, inputs)

        logits = self.get_logits(hidden_states)
        ret['logits'] = logits
        return ret

    async def async_sampling_logits(self, logits: torch.Tensor, inputs: ModelInputs,
                                    extra_inputs: ExtraInputs, sampling_inputs: SamplingInputs):
        """Sampling logits."""
        if self.spec_agent.is_enabled():
            extra_inputs = await self.spec_agent.async_sampling_logits(inputs, extra_inputs, sampling_inputs)
            return extra_inputs.next_token_ids, extra_inputs.logprobs, extra_inputs.output_token_ids, extra_inputs
        # record function does not support async function
        # so we can not decorate it on async_sampling_logits
        with record_function('sampling_logits'):
            logits_processor = FusedLogitsProcessor(
                sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
                guided_decoding_manager=self.guided_decoding_manager,
            )
            origin_logits = logits
            logits, raw_logprobs = await logits_processor(origin_logits)
            next_token_ids = logits_processor.sampling(logits)
            await logits_processor.accept_guided_tokens(next_token_ids)
            logprobs = logits_processor.compute_logprobs(raw_logprobs, next_token_ids)
            if logprobs is not None:
                logprobs = BatchedLogProbs(
                    vals=logprobs[0],
                    indices=logprobs[1],
                )
        # post sampling
        next_token_ids, extra_inputs = self.agent_strategy.post_sampling(inputs, logits, next_token_ids,
                                                                             extra_inputs)
        return next_token_ids, logprobs, next_token_ids, extra_inputs

    def _push_output(self, output: BatchedOutputs):
        """Push output."""
        event = torch.cuda.Event()
        event.record()
        self._out_que.put_nowait((output, event))

    @contextmanager
    def _broadcast_next_token(self, next_token_ids: torch.Tensor, extra_inputs: ExtraInputs, enable: bool = True):
        if not enable:
            yield
            return

        dist_ctx = self.dist_ctx
        with self.agent_strategy.broadcast_next_token(next_token_ids, extra_inputs, dist_ctx) as handle:
            yield handle

    @record_function('prepare_dp')
    async def _prepare_dp_v1(self, inputs: ModelInputs):
        """Prepare dp.

        If all inputs are dummy inputs, skip forward. If any of the inputs is prefill, then do prefill. Set padding
        batch size for decoding.
        """
        world_size = self.dist_config.world_size
        is_decoding = inputs.is_decoding
        num_tokens = inputs.input_ids.numel()
        is_dummy = inputs.is_dummy
        is_spec_enabled = self.spec_agent.is_enabled()
        is_microbatch_enabled = self.enable_microbatch

        # gather dp forward metadata
        batch_size = inputs.seq_length.numel()
        is_sleeping = self.state.is_sleeping
        draft_num_tokens = None
        if is_spec_enabled:
            draft_num_tokens = num_tokens
            if inputs.is_chunk:
                if inputs.is_first_chunk:
                    draft_num_tokens -= batch_size
                elif inputs.is_last_chunk:
                    draft_num_tokens += batch_size

        dp_forward_meta = DPForwardMeta(is_decoding=is_decoding,
                                        is_dummy=is_dummy,
                                        num_tokens=num_tokens,
                                        is_sleeping=is_sleeping,
                                        batch_size=batch_size,
                                        draft_num_tokens=draft_num_tokens)
        # check enable_microbatch
        if is_microbatch_enabled:
            tokens_num = inputs.input_ids.numel()
            if is_decoding:
                enable_microbatch = batch_size >= \
                    self.enable_microbatch_decode_batchsize_threshold
            else:
                enable_microbatch = batch_size >= \
                    self.enable_microbatch_prefill_batchsize_threshold and \
                    tokens_num >= self.enable_microbatch_prefill_token_threshold
            dp_forward_meta.enable_microbatch = enable_microbatch
        group = self.dist_ctx.cpu_group
        device = 'cpu'
        gathered_meta = DistGatherScalar(
            dp_forward_meta.values(
                is_spec_enabled=is_spec_enabled,
                is_microbatch_enabled=is_microbatch_enabled,
            ),
            world_size,
            device=device,
            group=group,
        )
        gathered_meta = GatheredDPForwardMeta.from_values(
            (await gathered_meta.async_wait()).cpu(),
            is_spec_enabled=is_spec_enabled,
            is_microbatch_enabled=is_microbatch_enabled,
        )

        # check is_decoding
        # if any one of the rank is prefill, then all ranks are prefill
        global_is_decoding = gathered_meta.global_is_decoding

        # check if all inputs are dummy inputs
        is_all_dummy = gathered_meta.is_all_dummy
        is_all_sleeping = gathered_meta.is_all_sleeping
        all_batch_sizes = gathered_meta.all_batch_sizes
        if is_all_dummy:
            return None, is_all_sleeping

        # pad batch size for decoding
        all_num_tokens = gathered_meta.all_num_tokens
        if os.environ.get('V4_DP_DEBUG'):
            try:
                _dp_rk = get_dist_manager().current_context().dist_config.dp_rank
            except Exception:
                _dp_rk = '?'
            logger.info(f'[V4DPDBG] dp_rank={_dp_rk} is_decoding={is_decoding} '
                        f'local_num_tokens={num_tokens} local_bs={batch_size} '
                        f'is_dummy={is_dummy} all_num_tokens={all_num_tokens} '
                        f'all_batch_sizes={all_batch_sizes}')
        if global_is_decoding:
            padding_batch_size = max(all_num_tokens)
            padding_batch_size = self.spec_agent.get_padding_batch_size(padding_batch_size)
            meta = self.patched_model.get_meta()
            meta.padding_batch_size = padding_batch_size
            logger.debug(f'padding_batch_size={padding_batch_size}')

        # update if enable_microbatch
        if is_microbatch_enabled:
            inputs.enable_microbatch = gathered_meta.global_enable_microbatch

        # update dp meta
        inputs.build_dp_meta(all_num_tokens)
        inputs.dp_meta.dp_batches = all_batch_sizes
        inputs.dp_meta.dp_is_decoding = global_is_decoding
        if is_spec_enabled:
            inputs.dp_meta.dp_draft_num_tokens = gathered_meta.all_draft_num_tokens
        inputs = self.patched_model.update_inputs(inputs)
        return inputs, is_all_sleeping

    def _get_inputs_from_delta(
        self,
        delta: ModelInputsDelta,
        sampling_inputs: SamplingInputs,
    ):
        """Get inputs from delta."""
        self.step_inputs.reindex(delta)
        inputs = self.step_inputs.model_inputs
        extra_inputs = self.step_inputs.extra_inputs
        stopping_criteria = self.step_inputs.stopping_criteria
        sampling_inputs.update_delta(self.step_inputs.sampling_delta)
        return inputs, extra_inputs, stopping_criteria, sampling_inputs

    def _prepare_inputs_prefill(
        self,
        inputs: ModelInputs,
        delta: ModelInputsDelta,
    ):
        """Prepare prefill inputs."""

        if delta is not None:
            # update decoding inputs with delta
            # for second round chat
            self.step_inputs.reindex(delta)

        if inputs.is_first_chunk:
            self._prev_chunk_output = None

        # check long context
        if inputs.is_chunk and self._prev_chunk_output is not None:
            # update model metas
            model_metas = self._prev_chunk_output.get('model_metas')
            inputs.model_metas = model_metas

            if inputs.is_last_chunk:
                # remove _prev_chunk_output
                self._prev_chunk_output = None

        return inputs

    async def _step_postprocess_with_output(self,
                                            last_logits: torch.Tensor,
                                            logits: torch.Tensor,
                                            inputs: ModelInputs,
                                            sampling_inputs: SamplingInputs,
                                            stopping_criteria: StoppingCriteria,
                                            model_metas: Any,
                                            need_broadcast_next: bool,
                                            return_logits: bool = False,
                                            return_ce_loss: bool = False,
                                            seq_length: torch.Tensor = None,
                                            all_routed_experts: Any = None,
                                            extra_inputs: ExtraInputs = None):
        """Step postprocess with output."""
        rank = self.rank
        logger.debug(f'<ForwardTask> rank[{rank}]: Sampling.')
        # Compute prompt CE before sampling, which may update last_logits in place.
        ce_loss = None
        if return_ce_loss and logits is not None and not inputs.is_dummy and not inputs.is_decoding:
            prev_last_logit = self._prev_chunk_last_logit if (inputs.is_chunk and not inputs.is_first_chunk) else None
            ce_loss = compute_input_ce_loss(logits, inputs.input_ids, seq_length, prev_last_logit=prev_last_logit)
            if inputs.is_chunk:
                self._prev_chunk_last_logit = None if inputs.is_last_chunk else logits[-1:].clone()

        (next_token_ids, logprobs, output_token_ids, extra_inputs) = await self.async_sampling_logits(
            last_logits, inputs, extra_inputs, sampling_inputs)
        with self._broadcast_next_token(next_token_ids, extra_inputs, enable=need_broadcast_next):
            logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids')

        extra_inputs = await self.spec_agent.async_model_forward(inputs, extra_inputs, sampling_inputs)

        if inputs.is_dummy:
            return inputs, extra_inputs, stopping_criteria, None, next_token_ids

        # post broadcast for spec agent
        with self.spec_agent.post_broadcast(extra_inputs, self.dist_ctx, need_broadcast_next):
            logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids')

        if self.spec_agent.is_enabled():
            logits = None

        # stopping criteria
        stopped, stop_pos, stopping_criteria = stopping_criteria.step(
            next_token_ids,
            sampling_inputs.stop_words,
            inputs=inputs,
            extra_inputs=extra_inputs,
        )

        # send output
        logger.debug(f'<ForwardTask> rank[{rank}]: Output')
        extra_outputs = self.agent_strategy.make_extra_outputs(extra_inputs)

        self._push_output(
            BatchedOutputs(next_token_ids=output_token_ids,
                           logits=logits if return_logits else None,
                           stopped=stopped,
                           stop_pos=stop_pos,
                           model_metas=model_metas,
                           logprobs=logprobs,
                           all_routed_experts=all_routed_experts,
                           extra_outputs=extra_outputs,
                           ce_loss=ce_loss))

        return inputs, extra_inputs, stopping_criteria, extra_outputs, next_token_ids

    async def _step_postprocess_without_output(
        self,
        inputs: ModelInputs,
        last_logits: torch.Tensor,
        extra_inputs: ExtraInputs,
        sampling_inputs: SamplingInputs,
        need_broadcast_next: bool,
    ):
        rank = self.rank
        # Avoid adding the ADInplaceOrView dispatch key to `next_token_ids`,
        # as it can trigger recompilation on different ranks when using torch.compile.
        next_token_ids, extra_inputs = self.agent_strategy.make_dummy_next_token(inputs, last_logits, extra_inputs)

        # broadcast next token for TP > 1
        with self._broadcast_next_token(next_token_ids, extra_inputs, enable=need_broadcast_next):
            logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids')

        extra_inputs = await self.spec_agent.async_model_forward(inputs, extra_inputs, sampling_inputs)

        if inputs.is_dummy:
            return inputs, next_token_ids, extra_inputs,  None

        # post broadcast for spec agent
        with self.spec_agent.post_broadcast(extra_inputs, self.dist_ctx, need_broadcast_next):
            logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids')

        if self.spec_agent.is_enabled():
            next_token_ids = extra_inputs.next_token_ids

        extra_outputs = self.agent_strategy.make_extra_outputs(extra_inputs)

        return inputs, next_token_ids, extra_inputs, extra_outputs

    async def _async_step(
        self,
        inputs: ModelInputs,
        delta: ModelInputsDelta = None,
        swap_in_map: dict = None,
        swap_out_map: dict = None,
        sampling_inputs: SamplingInputs = None,
        stopping_criteria: StoppingCriteria = None,
        return_logits: bool = False,
        return_routed_experts: bool = False,
        return_ce_loss: bool = False,
        extra_inputs: ExtraInputs = None,
    ):
        """Asyc forward task."""

        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        rank = self.rank
        tp = dist_config.attn_tp
        need_broadcast_next = (tp > 1)
        dp = dist_config.dp
        need_update_inputs = False

        if inputs is None:
            # decoding step, update prev_inputs with delta
            need_update_inputs = True
            assert delta is not None
            (
                inputs,
                extra_inputs,
                stopping_criteria,
                sampling_inputs,
            ) = self._get_inputs_from_delta(
                delta,
                sampling_inputs,
            )
        elif not inputs.is_dummy:
            # prefill step
            inputs = self._prepare_inputs_prefill(
                inputs,
                delta,
            )

        if dp > 1:
            # update inputs for dp
            inputs, is_all_sleeping = await self._prepare_dp_v1(inputs)
            # skip dummy forward.
            if inputs is None:
                if is_all_sleeping:
                    self.state.to_sleep.set()
                    await self.state.to_wakeup.wait()
                    self.state.to_wakeup.clear()
                    # sync after wakeup
                    dist.barrier()
                logger.debug(f'<ForwardTask> rank[{rank}]: all inputs are dummy, skip forward.')
                await asyncio.sleep(0.01)
                return

        # swap caches
        cache_swapping(self.cache_engine, swap_in_map=swap_in_map, swap_out_map=swap_out_map)

        # inference
        logger.debug(f'<ForwardTask> rank[{rank}]: model forward. '
                     f'batch_size={inputs.seq_length.size(0)} '
                     f'num_tokens={inputs.input_ids.size(-1)} '
                     f'is_dummy={inputs.is_dummy} '
                     f'is_chunk={inputs.is_chunk} '
                     f'is_first_chunk={inputs.is_first_chunk} '
                     f'is_last_chunk={inputs.is_last_chunk} '
                     f'dp_meta={inputs.dp_meta} '
                     f'is_decoding={inputs.is_decoding}')
        output = await self._async_model_forward(
            inputs,
            return_logits=return_logits or return_ce_loss,
            )

        if inputs.is_dummy and not self.spec_agent.is_enabled():
            # skip dummy forward output
            return

        logits = output['logits'][0]  # [bs, seq, prob] -> [seq, prob]
        seq_length = output.get('seq_length', inputs.seq_length)
        last_logits = self._slice_outs(logits, seq_length)  # [bs, 1, prob] -> [bs, prob]
        extra_inputs = self.agent_strategy.slice_extra_inputs(extra_inputs, inputs, output)
        model_metas = output.get('model_metas')

        if self.need_output:
            logger.debug(f'<ForwardTask> rank[{rank}]: Sampling.')
            # for router replay
            if return_routed_experts:
                all_routed_experts = output.get('all_routed_experts', None)
            else:
                all_routed_experts = None

            (
                inputs,
                extra_inputs,
                stopping_criteria,
                extra_outputs,
                next_token_ids,
            ) = await asyncio.shield(
                self._step_postprocess_with_output(
                    last_logits,
                    logits,
                    inputs,
                    sampling_inputs,
                    stopping_criteria,
                    model_metas,
                    need_broadcast_next,
                    return_logits=return_logits,
                    return_ce_loss=return_ce_loss,
                    seq_length=seq_length,
                    all_routed_experts=all_routed_experts,
                    extra_inputs=extra_inputs,
                ))
        else:
            (
                inputs,
                next_token_ids,
                extra_inputs,
                extra_outputs,
            ) = await asyncio.shield(
                self._step_postprocess_without_output(
                    inputs,
                    last_logits,
                    extra_inputs,
                    sampling_inputs,
                    need_broadcast_next,
                ))

        if inputs.is_dummy:
            # skip dummy forward output
            return

        sampling_delta = sampling_inputs.get_delta()
        if need_update_inputs:
            self.step_inputs.step_decode(
                inputs,
                extra_inputs,
                stopping_criteria,
                sampling_delta,
                next_token_ids,
                model_metas,
                extra_outputs,
            )
        elif inputs.is_chunk and not inputs.is_last_chunk:
            # _prev_chunk_output is used to update model metas
            self._prev_chunk_output = output
        elif self.cache_config.role != EngineRole.Prefill:
            self.step_inputs.merge_prefill(
                inputs,
                extra_inputs,
                stopping_criteria,
                sampling_delta,
                next_token_ids,
                model_metas,
                extra_outputs,
            )

    async def _async_loop_background(self, forward_event: asyncio.Event = None):
        """Async loop background."""
        with self.all_context(), torch.cuda.stream(self.stream), torch.inference_mode():

            # for dp
            input_maker = build_inputs_maker(self)

            while True:
                forward_inputs = await input_maker.get()
                h2d_transfer = forward_inputs.pop(_H2D_TRANSFER_KEY, None)
                if h2d_transfer is not None:
                    self._keep_h2d_transfer(h2d_transfer)
                    self.stream.wait_event(h2d_transfer.event)

                await self._async_step(**forward_inputs, )
                if forward_event is not None:
                    forward_event.set()

                input_maker.step()
                self._release_completed_h2d_transfers()

    def _keep_h2d_transfer(self, transfer: _H2DTransfer | None):
        """Keep H2D source refs alive until their async copies finish."""
        self._release_completed_h2d_transfers()
        if transfer is None or transfer.event.query():
            return
        self._pending_h2d_transfers.append(transfer)

    def _release_completed_h2d_transfers(self):
        """Release CPU-side H2D source refs after their async copies finish."""
        while len(self._pending_h2d_transfers) > 0:
            transfer = self._pending_h2d_transfers[0]
            if not transfer.event.query():
                break
            self._pending_h2d_transfers.popleft()

    async def _async_loop_inputs_preprocess(self, forward_event: asyncio.Event = None):
        """Async loop inputs preprocess."""
        non_blocking = True
        keys = ['inputs', 'delta', 'sampling_inputs', 'stopping_criteria', 'extra_inputs']
        while True:
            forward_inputs = await self._pre_in_que.get()
            forward_inputs_cuda = {}
            forward_inputs_cuda.update(forward_inputs)
            h2d_refs = {k: forward_inputs_cuda[k] for k in keys if forward_inputs_cuda.get(k) is not None}
            logger.debug('preprocessing forward inputs.')
            with torch.cuda.stream(self.out_stream), torch.inference_mode(), record_function('inputs_H2D'):
                for k in keys:
                    if k not in forward_inputs_cuda:
                        continue
                    forward_inputs_cuda[k] = _try_to_cuda(forward_inputs_cuda[k], non_blocking=non_blocking)
                h2d_event = torch.cuda.Event()
                h2d_event.record()
                forward_inputs_cuda[_H2D_TRANSFER_KEY] = _H2DTransfer(h2d_event, h2d_refs)
            logger.debug('preprocessing forward inputs done.')
            self._in_que.put_nowait(forward_inputs_cuda)
            if forward_event is not None:
                forward_event.clear()

    def start(self, forward_event: asyncio.Event = None):
        """Start event loop."""
        event_loop = asyncio.get_event_loop()
        self._pre_in_que = asyncio.Queue()
        self._in_que = asyncio.Queue()
        self._out_que = asyncio.Queue()

        # forward task
        logger.debug('Create task ModelAgentLoop.')
        self._background_task = event_loop.create_task(self._async_loop_background(forward_event),
                                                       name='ModelAgentLoop')
        self.tasks.add(self._background_task)
        self._background_task.add_done_callback(self.tasks.discard)

        # preprocess inputs task
        logger.debug('Create task ModelAgentPreprocess.')
        self._preprocess_task = event_loop.create_task(self._async_loop_inputs_preprocess(forward_event),
                                                       name='ModelAgentPreprocess')
        self.tasks.add(self._preprocess_task)
        self._preprocess_task.add_done_callback(self.tasks.discard)

        # profiler
        self.profiler = AgentProfiler(self.dist_ctx, self.stream)
        self.profiler.create_task()

    async def wait_tasks(self):
        """Wait tasks."""
        if len(self.tasks) == 0:
            return
        try:
            await wait_for_async_tasks(self.tasks)
        except asyncio.CancelledError:
            logger.debug(f'ModelAgent rank[{self.rank}] wait_tasks cancelled.')
            raise
        except BaseException as e:
            raise e from None
        finally:
            logger.debug(f'ModelAgent rank[{self.rank}] wait_tasks cleanup.')

    def stop(self):
        """Stop task."""
        if self.dist_config.dp > 1:
            return

        if self.profiler is not None:
            self.profiler.dump()

        for task in self.tasks:
            if not task.done():
                task.cancel()

        if self.guided_decoding_manager:
            self.guided_decoding_manager.clear()

    async def stop_async(self):
        """Stop task."""
        if self.dist_config.dp > 1:
            return

        if self.profiler is not None:
            # dirty hack for profiler
            while not self.stream.query():
                logger.debug('Profiler waiting for stream finish.')
                await asyncio.sleep(1)
            self.profiler.dump()

        for task in self.tasks:
            if not task.done():
                task.cancel()

        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.debug(f'ModelAgent {task.get_name()} task cancelled.')
        self._release_completed_h2d_transfers()

        if self.guided_decoding_manager:
            self.guided_decoding_manager.clear()

    def set_forward_inputs(self, inputs):
        """Set forward inputs."""
        assert self._pre_in_que is not None, ('Please start backendground task before forward.')
        self._pre_in_que.put_nowait(inputs)

    def _drain_queues(self):
        """Drain all internal queues to discard stale forward data."""
        for q in (self._pre_in_que, self._in_que, self._out_que):
            if q is None:
                continue
            while not q.empty():
                try:
                    item = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, dict):
                    self._keep_h2d_transfer(item.pop(_H2D_TRANSFER_KEY, None))

    async def get_output_async(self):
        """Async get output."""
        assert self._out_que is not None, ('Please start backendground task before forward.')
        out = await self._out_que.get()
        if out is None:
            return dict()

        out, event = out
        while not event.query():
            await asyncio.sleep(0.001)
        with torch.cuda.stream(self.out_stream), torch.inference_mode(), record_function('outputs_D2H'):
            event.wait()
            out = out.to_cpu()
            out.new_token_timestamp = time.time()
        self._release_completed_h2d_transfers()
        return out

    def _build_model(self):
        """Build patched model."""
        model_path = self.model_path
        adapters = self.adapters
        device = self.device
        rank = self.rank
        custom_module_map = self.model_config.custom_module_map
        if custom_module_map is not None:
            update_custom_module_map(custom_module_map)
        logger.debug(msg_with_rank(rank, 'build model.'))
        # for router replay
        enable_return_routed_experts = self.misc_config.enable_return_routed_experts and self.need_output

        build_model_ctx = BuildModelContext(language_model_only=self.misc_config.language_model_only,
                                            dllm_config=self.misc_config.dllm_config,
                                            strategy_factory=self.strategy_factory,
                                            enable_return_routed_experts=enable_return_routed_experts,
                                            quant_config=self.model_config.quant_config,
                                            fp32_lm_head=self.model_config.fp32_lm_head,
                                            tie_word_embeddings=self.model_config.tie_word_embeddings,
                                            num_spec_tokens=self.spec_agent.num_spec_tokens,
                                            max_batch_size=self.cache_config.max_batches)
        patched_model = build_patched_model(self.model_config, device=device, build_model_ctx=build_model_ctx)
        logger.debug(msg_with_rank(rank, 'loading weights.'))
        if not self.misc_config.empty_init:
            load_model_weights(patched_model, model_path, device=device)
        if adapters is not None:
            logger.debug(msg_with_rank(rank, 'loading adapters.'))
            add_adapters(patched_model, adapters, dtype=self.model_config.dtype, device=device)
        self.patched_model = patched_model
        self.build_model_ctx = build_model_ctx

    def build_model(self):
        """Build model api."""
        with self.all_context():
            self._build_model()
            self.spec_agent.build_model(self.misc_config.empty_init,
                                        self.patched_model,
                                        build_model_ctx=self.build_model_ctx)

    def build_graph_runner(self):
        """Build graph runner."""
        with self.all_context():
            backend = get_backend()
            self.patched_model = backend.build_graph_runner(self.patched_model,
                                                            model_config=self.model_config,
                                                            cache_config=self.cache_config,
                                                            backend_config=self.backend_config,
                                                            device=self.device)
            self.spec_agent.build_graph_runner()

    def build_cache_engine(self):
        """Build cache engine."""
        with self.all_context():
            dist_ctx = get_dist_manager().current_context()
            dist_cfg = self.dist_config
            tp = dist_cfg.attn_tp

            self.cache_engine = CacheEngine(self.cache_config,
                                            self.model_config,
                                            rank=self.rank,
                                            tp_rank=dist_ctx.attn_tp_group.rank,
                                            world_size=tp,
                                            cache_stream=self.cache_stream)
            self.state_cache_engine = StateCacheEngine(self.cache_config)

            self.spec_agent.build_cache_engine(self.cache_stream)

    def _forward_impl(self, inputs: ModelInputs):
        if self.rank == 0:
            _v4_prof_maybe_start()
        output = model_forward(
            self.patched_model,
            inputs,
            self.cache_engine,
            state_cache_engine=self.state_cache_engine,
            stream=self.stream,
        )
        # DEBUG (V4_SAS_SYNC=1): force a sync after every forward so an async
        # aicore OOB surfaces at the END of the SAME step (not deferred to the
        # next step's get_cpu_seqlens .cpu()). On crash, stop+export the msprof
        # profiler so the trace.json captures the failing step's op sequence ->
        # the last recorded op before the OOB is the culprit. Re-raise after.
        if os.environ.get('V4_SAS_SYNC', '0') == '1':
            try:
                torch.npu.synchronize()
            except Exception as _e:
                if _V4_PROF is not None:
                    try:
                        _V4_PROF.stop()
                        logger.error(f'[V4-SAS-SYNC] profiler stopped on crash; '
                                     f'trace -> {os.environ.get("V4_PROFILE_DIR")}')
                        _V4_PROF = None
                    except Exception:
                        pass
                logger.error(f'[V4-SAS-SYNC] forward crashed synchronously: {_e}')
                raise
        if self.rank == 0:
            _v4_prof_maybe_step()
        return output

    async def async_forward(self, inputs: ModelInputs):
        """Model forward.

        Args:
            inputs (dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        output = self._forward_impl(inputs)
        await asyncio.sleep(0)
        return output

    @record_function('get_logits')
    def get_logits(self, hidden_states: torch.Tensor):
        """Get logits of model output."""
        return self.patched_model.get_logits(hidden_states)

    def get_input_processor(self):
        """Get input processor."""
        return self.patched_model.get_input_processor()

    def reset_graph_runner(self):
        """Reset graph runner to prevent tp hanging."""
        with self.all_context():
            if hasattr(self.patched_model, 'reset'):
                self.patched_model.reset()

            self.spec_agent.reset_graph_runner()

    @torch.inference_mode()
    def update_params(self, request: UpdateParamsRequest):
        """Update params."""

        # modified from https://github.com/vllm-project/vllm/blob/v0.8.5/examples/offline_inference/rlhf_utils.py#L82
        def _construct(item, require_clone: bool = True):
            func, args = item
            args = list(args)
            args[6] = torch.cuda.current_device()  # device id.
            ipc_tensor = func(*args)
            return ipc_tensor.clone() if require_clone else ipc_tensor

        def _deserialize_weights(serialized_data):
            weights = ForkingPickler.loads(pybase64.b64decode(serialized_data))
            if request.load_format == 'flattened_bucket':
                metadata: list[FlattenedTensorMetadata] = weights['metadata']
                if not metadata:
                    return []
                if 'flattened_tensor' in weights:
                    # Determine if clone is required
                    require_clone = weights.get('require_clone', True)
                    if 'event_ipc_handle' in weights and not hasattr(torch.cuda.Event, 'from_ipc_handle'):
                        # Force clone when IPC event is provided but cannot be used
                        require_clone = True
                    self._update_params_ipc_tensor = _construct(weights['flattened_tensor'],
                                                                require_clone=require_clone)
                elif self._update_params_ipc_tensor is None:
                    raise ValueError(
                        'flattened_tensor is not provided in weights and no cached ipc tensor is available. '
                        'Please provide flattened_tensor on the first update_params call.')
                if 'event_ipc_handle' in weights and hasattr(torch.cuda.Event, 'from_ipc_handle'):
                    self._update_params_ipc_event = torch.cuda.Event.from_ipc_handle(
                        device=torch.cuda.current_device(),
                        handle=weights['event_ipc_handle'],
                    )
                flattened_tensor: torch.Tensor = self._update_params_ipc_tensor
                if self._update_params_ipc_event is not None:
                    self._update_params_ipc_event.wait()
                bucket = FlattenedTensorBucket(flattened_tensor=flattened_tensor, metadata=metadata)
                return list(bucket.reconstruct_tensors())
            return [(k, _construct(v)) for k, v in weights]

        def _split_main_and_draft(weights):
            # TODO, zhouxinyu, support split and update weights for other mtp methods
            if not self.spec_agent.is_enabled() or self.spec_agent.method != 'qwen3_5_mtp':
                return weights, []
            main = [(name, weight) for name, weight in weights if not name.startswith('mtp.')]
            draft = [(name, weight) for name, weight in weights if name.startswith('mtp.')]
            return main, draft

        with self.all_context():
            # After deserialization, weights is a dict with following keys:
            # - metadata: List[FlattenedTensorMetadata]
            # - flattened_tensor: the flattened tensor for weights, optional
            # - event_ipc_handle: the ipc handle of the event
            #   that used to sync stream across processes, optional
            serialized_data = request.serialized_named_tensors
            if isinstance(serialized_data, list):
                serialized_data = serialized_data[self.dist_ctx.tp_group.rank]

            model = self.patched_model.get_model()
            spec_model = self.spec_agent.get_model()

            weights = _deserialize_weights(serialized_data)
            main_weights, draft_weights = _split_main_and_draft(weights)

            for m, w, tag in [(model, main_weights, 'main'), (spec_model, draft_weights, 'draft')]:
                if m is None or not w:
                    continue

                w = list(ModelWeightLoader._rename_weights_iterator(w, m))
                logger.debug(f'Update_params: {tag}_num_tensors={len(w)}')
                m.load_weights(iter(w))

                if self._update_params_ipc_event is not None:
                    self._update_params_ipc_event.record()

            if request.finished:
                for m in filter(None, [model, spec_model]):
                    for _, mod in m.named_modules():
                        if hasattr(mod, 'update_weights'):
                            mod.update_weights()

                    torch.cuda.synchronize()
                    self._update_params_ipc_event = None
                    self._update_params_ipc_tensor = None

            torch.cuda.empty_cache()

    def init_weights_update_group(self, request: InitWeightsUpdateGroupRequest):
        """Create a NCCL process group with an external trainer for the
        disaggregated weight-update path.

        rank 0 is the trainer; this engine's local TP ranks fill `rank_offset .. rank_offset + tp - 1`.
        """
        with self.all_context():
            group_name = request.group_name
            if not group_name:
                return False, 'group_name cannot be empty'
            if group_name in self._model_update_group:
                return False, f'group {group_name!r} already initialized'

            local_rank = self.dist_ctx.tp_group.rank
            rank = request.rank_offset + local_rank
            init_method = f'tcp://{request.master_address}:{request.master_port}'
            logger.info(f'init weights update group: master={request.master_address}:{request.master_port}, '
                        f'rank_offset={request.rank_offset}, rank={rank}, world_size={request.world_size}, '
                        f'group_name={group_name}, backend={request.backend}')
            try:
                pg = init_custom_process_group(
                    backend=request.backend,
                    init_method=init_method,
                    world_size=request.world_size,
                    rank=rank,
                    group_name=group_name,
                )
                self._model_update_group[group_name] = pg
                return True, 'Succeeded to initialize weights update group.'
            except Exception as e:
                msg = f'Failed to initialize weights update group: {e}'
                logger.exception(msg)
                return False, msg

    @torch.inference_mode()
    def update_weights_from_distributed(self, request: UpdateWeightsFromDistributedRequest):
        """Receive a bucket of weights through the previously initialized NCCL
        group and load them into the running model."""
        with self.all_context():
            group_name = request.group_name
            pg = self._model_update_group.get(group_name)
            if pg is None:
                return False, (f'group {group_name!r} not initialized. '
                               'Call init_weights_update_group first.')

            device = torch.cuda.current_device()
            try:
                if request.names:
                    named_tensors = []
                    for name, dtype_str, shape in zip(request.names, request.dtypes, request.shapes):
                        target_dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str
                        named_tensors.append((name, torch.empty(shape, dtype=target_dtype, device=device)))

                    if request.load_format == 'flattened_bucket':
                        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
                        flattened_tensor = bucket.get_flattened_tensor()
                        dist.broadcast(flattened_tensor, src=0, group=pg)
                        weights = list(bucket.reconstruct_tensors())
                    else:
                        handles = []
                        for _, tensor in named_tensors:
                            handles.append(dist.broadcast(tensor, src=0, group=pg, async_op=True))
                        for handle in handles:
                            handle.wait()
                        weights = named_tensors
                else:
                    weights = []

                model = self.patched_model.get_model() if self.patched_model is not None else None
                spec_model = self.spec_agent.get_model()
                # Same draft-split rule as update_params (currently only qwen3_5_mtp).
                if self.spec_agent.is_enabled() and self.spec_agent.method == 'qwen3_5_mtp':
                    main_weights = [(n, w) for n, w in weights if not n.startswith('mtp.')]
                    draft_weights = [(n, w) for n, w in weights if n.startswith('mtp.')]
                else:
                    main_weights, draft_weights = weights, []

                for m, w, tag in [(model, main_weights, 'main'), (spec_model, draft_weights, 'draft')]:
                    if m is None or not w:
                        continue
                    renamed = list(ModelWeightLoader._rename_weights_iterator(w, m))
                    logger.info(f'update_weights_from_distributed: {tag}_num_tensors={len(renamed)}')
                    m.load_weights(iter(renamed))

                if request.finished:
                    for m in filter(None, [model, spec_model]):
                        for _, mod in m.named_modules():
                            if hasattr(mod, 'update_weights'):
                                mod.update_weights()
                        torch.cuda.synchronize()
                    # FusedMoE.update_weights() above replaces the gate_up / down
                    # Parameter objects (LinearWeights.update_weight registers a new
                    # nn.Parameter), so any CUDA graph captured before the update
                    # still references the freed old pointers. Drop the captured
                    # graphs so the next forward re-captures with the new params.
                    self.reset_graph_runner()

                torch.cuda.empty_cache()
                return True, 'Succeeded to update parameter online.'
            except Exception as e:
                msg = (f'Failed to update parameter online: {e}. The model weights are partially updated; '
                       'please discard them and reload.')
                logger.exception(msg)
                return False, msg

    def destroy_weights_update_group(self, request: DestroyWeightsUpdateGroupRequest):
        """Destroy a previously initialized weights-update process group."""
        group_name = request.group_name
        pg = self._model_update_group.get(group_name)
        if pg is None:
            return False, f'group {group_name!r} not initialized'
        try:
            dist.destroy_process_group(pg)
            self._model_update_group.pop(group_name)
            return True, f'Succeeded to destroy group {group_name!r}.'
        except Exception as e:
            msg = f'Failed to destroy weights update group {group_name!r}: {e}'
            logger.exception(msg)
            return False, msg

    @torch.inference_mode()
    async def sleep(self, level: int = 1):
        """Sleep."""
        self.state.is_sleeping = True
        if self.dist_config.dp > 1:
            await self.state.to_sleep.wait()
        device = 'cpu' if level == 1 else 'meta'
        self.cache_engine = None
        self.state_cache_engine = None
        self.reset_graph_runner()
        self.patched_model.get_model().to(device=device, non_blocking=True)

        spec_model = self.spec_agent.get_model()
        if spec_model is not None:
            self.spec_agent.cache_engine = None
            spec_model.to(device=device, non_blocking=True)

        self._drain_queues()
        torch.cuda.synchronize()
        self._release_completed_h2d_transfers()
        # force clean _update_params_ipc tensor and event after all gpu jobs done
        self._update_params_ipc_tensor = None
        self._update_params_ipc_event = None
        torch.cuda.empty_cache()
        self.state.to_sleep.clear()

    @torch.inference_mode()
    def wakeup(self, tags: list[str] | None = None):
        """Wakeup."""
        if tags is None:
            tags = ['weights', 'kv_cache']

        if 'weights' in tags:
            device = next(self.patched_model.get_model().parameters()).device
            assert device.type in ['cpu', 'meta']
            spec_model =  self.spec_agent.get_model()

            if device.type == 'cpu':
                self.patched_model.get_model().to(torch.cuda.current_device())
                if spec_model is not None:
                    spec_model.to(torch.cuda.current_device())
            else:
                # user should update weights after wakeup
                old_empty_init = self.misc_config.empty_init
                self.misc_config.empty_init = True
                self.build_model()
                self.build_graph_runner()
                self.misc_config.empty_init = old_empty_init

        if 'kv_cache' in tags:
            self.build_cache_engine()
            self.warmup()
            self.state.is_sleeping = False
            # wake up signal
            if self.dist_config.dp > 1:
                self.state.to_wakeup.set()

    def release(self):
        """release."""
        self.reset_graph_runner()
        self.patched_model = None
        self.cache_engine = None
        self.state_cache_engine = None
        torch.cuda.empty_cache()
