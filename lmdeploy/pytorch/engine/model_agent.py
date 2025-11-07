# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import base64
import functools
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, fields
from multiprocessing.reduction import ForkingPickler
from os import getenv
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger

from ..backends import get_backend
from ..config import BackendConfig, CacheConfig, MiscConfig, ModelConfig
from ..devices import DeviceContext, get_device_manager
from ..distributed import DistContext, get_dist_manager
from ..model_inputs import ModelInputs, step_ctx_manager
from ..models.patch import BuildModelContext, add_adapters, build_patched_model, update_custom_module_map
from ..strategies import build_strategy_factory
from ..strategies.base.model_agent import ExtraInputs, ExtraOutputs, StoppingCriteria
from ..utils import get_gpu_memory
from ..weight_loader.model_weight_loader import ModelWeightLoader, load_model_weights
from .cache_engine import CacheEngine, StateCacheEngine
from .guided_process import GuidedDecodingManager
from .logits_process import FusedLogitsProcessor, SamplingInputs

logger = get_logger('lmdeploy')


@dataclass
class BatchedLogProbs:
    vals: torch.Tensor
    indices: torch.Tensor

    def to_cpu(self):
        """To cpu."""
        return BatchedLogProbs(vals=self.vals.cpu(), indices=self.indices.cpu())

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
            vals = torch.from_numpy(vals)
        return BatchedLogProbs(vals=vals, indices=torch.from_numpy(self.indices))


@dataclass
class BatchedOutputs:
    next_token_ids: torch.Tensor
    stopped: torch.Tensor
    stop_pos: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    model_metas: List[Dict[str, Any]] = None
    logprobs: Optional[BatchedLogProbs] = None
    new_token_timestamp: int = 0
    extra_outputs: Optional[ExtraOutputs] = None

    def to_cpu(self):
        """To cpu."""
        out = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.cpu()
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


class AgentProfiler:

    def __init__(self, dist_ctx: DistContext, stream: torch.Stream):
        from lmdeploy.pytorch import envs
        self.rank = dist_ctx.rank
        self.dp_rank = dist_ctx.dp_rank
        self.dp = dist_ctx.dp
        self.stream = stream
        self.profiler = None
        if self.dp == 1:
            self.name = f'rank[{self.rank}]'
        else:
            self.name = f'dp_rank[{self.dp_rank}]'

        self.delay = envs.torch_profile_delay
        self.duration = envs.torch_profile_duration

        self.profiler = self._build_profiler()
        self.prefix = envs.torch_profile_output_prefix
        self._task = None
        self._started = False
        if self.dp > 1 and self.duration < 0 and self.profiler is not None:
            logger.warning('Do not support duration<=0 for dp > 1.')
            self.profiler = None

    def _build_profiler(self):
        from lmdeploy.pytorch import envs
        activities = []
        if envs.torch_profile_cpu:
            activities.append(ProfilerActivity.CPU)
        if envs.torch_profile_cuda:
            activities.append(ProfilerActivity.CUDA)
        if len(activities) > 0:
            logger.warning(f'Profiler start on {self.name}. '
                           'Please Note that profiling might harm performance.')
            profiler = profile(activities=activities)
            return profiler
        else:
            return None

    def dump(self):
        """Dump profile result."""
        if self.profiler is None:
            return

        if not self._started:
            logger.warning(f'Profiler {self.name} not started, skip dump.')
            return

        try:
            self.profiler.stop()
            rank = self.rank if self.dp == 1 else self.dp_rank
            dump_path = f'{self.prefix}{rank}.json'
            self.profiler.export_chrome_trace(dump_path)
            logger.warning(f'Profiler {self.name} dump to {dump_path}.')
        except Exception as e:
            logger.error(f'Failed to dump profile {self.name} result: {e}')
        finally:
            self.profiler = None

    async def profile_task(self):
        """Profile task."""
        if self.profiler is None:
            return

        # start profiler with delay
        await asyncio.sleep(self.delay)
        self.profiler.start()
        self._started = True

        if self.duration <= 0:
            return

        # dump profiler
        await asyncio.sleep(self.duration)
        self.dump()

    def create_task(self):
        """Create task."""
        event_loop = asyncio.get_event_loop()
        self._task = event_loop.create_task(self.profile_task())


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
    with torch.cuda.stream(stream), step_ctx_manager(model.ctx_mgr):
        # forward
        ctx_mgr = model.ctx_mgr
        context = ctx_mgr.build_context(
            inputs=inputs,
            model_config=cache_engine.model_config,
            kv_caches=cache_engine.gpu_cache,
            state_caches=state_cache_engine.state_caches,
            kv_quant_policy=cache_engine.cache_config.quant_policy,
        )
        with ctx_mgr.context(context):
            model_metas = None
            model_metas = model.update_model_metas(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            input_dict = model.prepare_inputs_for_generation(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            output = model(**input_dict)

            # InternVL-3.5-Flash will change the seqlen, model_metas during forward
            if context.model_metas is not None and context.model_metas[0] is not None:
                model_metas = context.model_metas
            seq_length = context.q_seqlens[:len(inputs.seq_length)]

    return dict(hidden_states=output, model_metas=model_metas, seq_length=seq_length)


def _try_to_cuda(val, non_blocking: bool = False):
    if val is None:
        return val
    elif isinstance(val, torch.Tensor):
        return val.cuda(non_blocking=non_blocking)
    elif hasattr(val, 'to_device'):
        return val.to_device('cuda', non_blocking=non_blocking)
    else:
        raise RuntimeError(f'Can not cast {type(val)} to cuda.')


class DistGatherScalar:
    """Distribute value gather."""

    def __init__(self, val, size: int, device: str = 'cpu', group: dist.ProcessGroup = None):
        self.val = val
        self.device = device
        self.group = group

        self.all_vals = torch.tensor([val] * size, device=device)
        self.worker = dist.all_gather_into_tensor(self.all_vals,
                                                  self.all_vals.new_tensor([val]),
                                                  group=group,
                                                  async_op=True)

    async def async_wait(self, timeout: float = 0.001):
        while not self.worker.is_completed():
            await asyncio.sleep(timeout)
        self.worker.wait()
        return self.all_vals


SwapMap = Dict[int, int]


class BaseModelAgent:
    """Base model agent.

    load model on local gpu

    Args:
        model_path (str): The hugging face model path.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        trust_remote_code (bool): Trust remote code
    """

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 misc_config: MiscConfig,
                 dist_ctx: DistContext,
                 device_ctx: DeviceContext,
                 adapters: Dict[str, str] = None):

        self.model_config = model_config
        self.cache_config = cache_config
        # use raw tokenizer
        self.tokenizer = Tokenizer(model_path).model.model

        self._pre_in_que = None
        self._in_que = None
        self._out_que = None
        self._background_task = None
        self._preprocess_task = None

        self.stream = torch.cuda.Stream()
        self.out_stream = torch.cuda.Stream()
        self.cache_stream = torch.cuda.Stream()

        self.dist_ctx = dist_ctx
        self.device_ctx = device_ctx

        device = 'cuda'
        self.backend_config = backend_config
        self.misc_config = misc_config
        rank = dist_ctx.rank

        self.model_path = model_path
        self.adapters = adapters
        self.device = device
        self.rank = rank

        tp_rank = dist_ctx.tp_rank
        tp = dist_ctx.tp
        world_size = dist_ctx.world_size
        self.tp = tp
        self.world_size = world_size
        self.tp_rank = tp_rank

        self.patched_model = None
        self.cache_engine = None
        self.state_cache_engine = None
        self.profiler: AgentProfiler = None
        try:
            self.guided_decoding_manager = GuidedDecodingManager(self.tokenizer, model_config.vocab_size)
        except ValueError as e:
            logger.warning(f'Failed to create GuidedManager for tokenizer {type(self.tokenizer)}: {e}')
            self.guided_decoding_manager = None

        # microbatch
        self.enable_microbatch = self.dist_ctx.dist_config.enable_microbatch
        self.enable_microbatch_prefill_batchsize_threshold = \
            int(getenv('ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD', 2))
        self.enable_microbatch_prefill_token_threshold = \
            int(getenv('ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD', 2))
        self.enable_microbatch_decode_batchsize_threshold = \
            int(getenv('ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD', 2))

        # strategy
        self.strategy_factory = build_strategy_factory(model_config, misc_config)
        self.inputs_strategy = self.strategy_factory.build_model_inputs_strategy()
        self.agent_strategy = self.strategy_factory.build_model_agent_strategy()

    @contextmanager
    def all_context(self):
        device_mgr = get_device_manager()
        dist_mgr = get_dist_manager()
        with device_mgr.context(self.device_ctx), dist_mgr.context(self.dist_ctx):
            yield

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        self.cache_config = cache_config

    def set_model_config(self, model_config: ModelConfig):
        """Set model config."""
        self.model_config = model_config

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
            return

        with self.all_context():
            max_batches = self.cache_config.max_batches
            num_tokens = max_batches
            dist_ctx = get_dist_manager().current_context()
            dp = dist_ctx.dp
            # warmup prefill
            inputs = self.inputs_strategy.make_dummy(max_batches,
                                                     is_decoding=False,
                                                     device='cuda',
                                                     vocab_size=self.model_config.vocab_size)
            if dp > 1:
                inputs.build_dp_meta()
            self._forward_impl(inputs)

            # warmup decoding(with cuda graph)
            capture_batch_sizes = self.patched_model.get_capture_batch_sizes()
            capture_batch_sizes = sorted(capture_batch_sizes, reverse=True)
            for num_tokens in capture_batch_sizes:
                inputs = self.inputs_strategy.make_dummy(num_tokens,
                                                         is_decoding=True,
                                                         device='cuda',
                                                         vocab_size=self.model_config.vocab_size)
                if dp > 1:
                    inputs.build_dp_meta()
                self._forward_impl(inputs)

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
        sync_long_context: bool,
    ):
        """Model forward."""
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        strategy = self.agent_strategy

        class _OutputGather:
            """Output gather."""

            def __init__(self, max_seq_len):
                self._max_seq_len = max_seq_len
                self._start = 0
                self._output: torch.Tensor = None
                self._device: torch.device = None

            def gather(self, output):
                """gather."""
                tmp_output = output['hidden_states']

                if not return_logits:
                    self._output = tmp_output
                    return

                out_logits = self._output
                start = self._start
                seq_len = tmp_output.size(-2)
                if out_logits is None:
                    out_logits = tmp_output.new_empty(1, self._max_seq_len, tmp_output.size(-1), device='cpu')
                    self._device = tmp_output.device
                out_logits[:, start:start + seq_len].copy_(tmp_output, non_blocking=True)
                self._start = start + seq_len
                self._output = out_logits

            def get_output(self):
                """Get tmp_output."""
                if not return_logits:
                    seqlen = torch.full((1, ),
                                        self._output.numel() // self._output.size(-1),
                                        device=self._output.device,
                                        dtype=self._output.dtype)
                    return strategy.slice_outputs(self._output, seqlen)
                torch.cuda.synchronize()
                return self._output.to(self._device)

        __forward = self.async_forward

        async def __long_context_single_forward(new_inputs, max_seqlen: int):
            """One large sequence."""
            dist_ctx = get_dist_manager().current_context()
            dp = dist_ctx.dp
            model_metas = new_inputs[0].model_metas
            output_gather = _OutputGather(max_seqlen)
            for inp in new_inputs:
                if dp > 1:
                    inp.build_dp_meta()
                inp.model_metas = model_metas
                tmp_out = await __forward(inp)
                model_metas = tmp_out.get('model_metas')
                output_gather.gather(tmp_out)
                tmp_out.pop('hidden_states', None)
            tmp_out['hidden_states'] = output_gather.get_output()
            return tmp_out

        origin_inputs = inputs

        # make long context inputs
        is_long_context = inputs.input_ids.numel() > max_prefill_token_num and not inputs.is_decoding
        max_seqlen = 0
        if is_long_context:
            seq_len = inputs.seq_length
            batch_size = seq_len.size(0)
            assert batch_size == 1, 'Do not support batched long context.'
            max_seqlen = inputs.seq_length[0]
            inputs = inputs.split(max_prefill_token_num)

        # get num dummy loop
        dummy_loop = 0
        if sync_long_context:
            forward_loop = 1
            if is_long_context:
                forward_loop = len(inputs)
            max_loop = torch.tensor(forward_loop, device='cuda')
            dist.all_reduce(max_loop, op=dist.ReduceOp.MAX)
            dummy_loop = max_loop - forward_loop

        if not is_long_context:
            ret = await __forward(inputs)
        else:
            ret = await __long_context_single_forward(inputs, max_seqlen)

        if not return_logits:
            ret = self._postprocess_forward_output(ret, origin_inputs)

        # compute dummy loop
        if dummy_loop > 0:
            dummy_inputs = self.inputs_strategy.make_dummy(1, False, 'cuda', vocab_size=self.model_config.vocab_size)
        for _ in range(dummy_loop):
            await __forward(dummy_inputs)

        hidden_states = ret.pop('hidden_states')
        logits = self.get_logits(hidden_states)
        ret['logits'] = logits
        return ret

    async def async_sampling_logits(self, logits: torch.Tensor, sampling_inputs: SamplingInputs, inputs: ModelInputs):
        """Sampling logits."""

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
            logprobs = logits_processor.compute_logprobs(raw_logprobs, next_token_ids)
            if logprobs is not None:
                logprobs = BatchedLogProbs(
                    vals=logprobs[0],
                    indices=logprobs[1],
                )

        return next_token_ids, logprobs

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

    async def _async_step_background(
        self,
        inputs: ModelInputs,
        loop_count: int,
        swap_in_map: Dict = None,
        swap_out_map: Dict = None,
        sampling_inputs: SamplingInputs = None,
        stopping_criteria: StoppingCriteria = None,
        return_logits: bool = False,
        is_dummy: bool = False,
        sync_long_context: bool = False,
        extra_inputs: ExtraInputs = None,
    ):
        """Asyc forward task."""
        dist_ctx = get_dist_manager().current_context()

        @record_function('update_inputs_for_next_step')
        def __update_inputs(next_token_ids, model_metas, extra_inputs):
            """Update inputs."""
            return self.agent_strategy.update_inputs_for_next_step(
                inputs,
                sampling_inputs,
                next_token_ids=next_token_ids,
                model_metas=model_metas,
                extra_inputs=extra_inputs,
            )

        @asynccontextmanager
        async def __prepare_dp():
            """Prepare dp."""
            if dp == 1:
                yield
                return

            nonlocal inputs, sync_long_context, is_all_dummy

            # gather dp forward metadata
            batch_size = inputs.seq_length.numel()
            dp_forward_meta = [int(is_decoding), int(is_dummy), batch_size, int(sync_long_context)]
            # check enable_microbatch
            if self.enable_microbatch:
                tokens_num = inputs.input_ids.numel()
                if is_decoding:
                    enable_microbatch = batch_size >= \
                        self.enable_microbatch_decode_batchsize_threshold
                else:
                    enable_microbatch = batch_size >= \
                        self.enable_microbatch_prefill_batchsize_threshold and \
                        tokens_num >= self.enable_microbatch_prefill_token_threshold
                dp_forward_meta.append(int(enable_microbatch))
            gathered_meta = DistGatherScalar(dp_forward_meta, dp, device='cuda')

            yield

            gathered_meta = (await gathered_meta.async_wait()).cpu()

            # check is_decoding
            all_is_decoding = gathered_meta[:, 0]
            assert all_is_decoding.sum().item() in [0, dp]

            # check if all inputs are dummy inputs
            is_all_dummy = gathered_meta[:, 1].all()
            if is_all_dummy:
                return

            if is_decoding:
                all_batch_sizes = gathered_meta[:, 2]
                padding_batch_size = all_batch_sizes.max().item()
                meta = self.patched_model.get_meta()
                meta.padding_batch_size = padding_batch_size
                logger.debug(f'padding_batch_size={padding_batch_size}')
            else:
                all_sync_flags = gathered_meta[:, 3].bool()
                sync_long_context = all_sync_flags.any()
                logger.debug(f'sync_long_context={sync_long_context}')

            # update if enable_microbatch
            if self.enable_microbatch and gathered_meta[:, 4].all():
                inputs.enable_microbatch = True

            # update dp meta
            inputs.build_dp_meta()
            inputs = self.patched_model.update_inputs(inputs)

        # dist tools
        dist_ctx = get_dist_manager().current_context()
        rank = dist_ctx.rank
        tp = dist_ctx.tp
        dp = dist_ctx.dp
        sync_long_context = False if dp == 1 else sync_long_context
        is_decoding = inputs.is_decoding

        logger.debug(f'<ForwardTask> rank[{rank}]: '
                     f'batch_size={inputs.seq_length.size(0)} '
                     f'num_tokens={inputs.input_ids.size(-1)} '
                     f'is_decoding={inputs.is_decoding}')

        # is_all_dummy would be updated in __prepare_dp
        is_all_dummy = False
        async with __prepare_dp():
            pass

        need_output = dp > 1 or rank % tp == 0

        # skip dummy forward.
        if is_all_dummy:
            logger.debug(f'<ForwardTask> rank[{rank}]: all inputs are dummy, skip forward.')
            return

        if not is_decoding:
            # init state cache for first time prefill
            # I don't know if this is necessary...
            self.state_cache_engine.init_caches(inputs.state_offsets, inputs.history_lengths == 0)
        cache_swapping(self.cache_engine, swap_in_map=swap_in_map, swap_out_map=swap_out_map)
        for idx in range(loop_count):
            # inference
            logger.debug(f'<ForwardTask> rank[{rank}]: model forward [{idx}].')
            output = await self._async_model_forward(
                inputs,
                return_logits=return_logits,
                sync_long_context=sync_long_context,
            )
            logits = output['logits']
            logits = logits[0]  # [bs, seq, prob] -> [seq, prob]
            seq_length = inputs.seq_length
            seq_length = output.get('seq_length', inputs.seq_length)
            last_logits = self._slice_outs(logits, seq_length)  # [bs, 1, prob] -> [bs, prob]
            extra_inputs = self.agent_strategy.slice_extra_inputs(extra_inputs, seq_length)
            model_metas = output.get('model_metas')

            # output empty for dummy inputs
            if is_dummy:
                continue

            need_broadcast_next = (dp == 1 and tp > 1 and idx < loop_count - 1)

            # sampling and stopping
            if need_output:
                logger.debug(f'<ForwardTask> rank[{rank}]: Sampling [{idx}].')
                # sampling
                next_token_ids, logprobs = await self.async_sampling_logits(last_logits, sampling_inputs, inputs)

                # post sampling
                next_token_ids, extra_inputs = self.agent_strategy.post_sampling(inputs, last_logits, next_token_ids,
                                                                                 extra_inputs)

                with self._broadcast_next_token(next_token_ids, extra_inputs, enable=need_broadcast_next):
                    logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids [{idx}]')

                    # stopping criteria
                    stopped, stop_pos, stopping_criteria = stopping_criteria.step(next_token_ids,
                                                                                  sampling_inputs.stop_words,
                                                                                  inputs=inputs,
                                                                                  extra_inputs=extra_inputs)

                    # send output
                    logger.debug(f'<ForwardTask> rank[{rank}]: Output [{idx}]')
                    extra_outputs = self.agent_strategy.make_extra_outputs(extra_inputs)
                    self._push_output(
                        BatchedOutputs(next_token_ids=next_token_ids,
                                       logits=logits if return_logits else None,
                                       stopped=stopped,
                                       stop_pos=stop_pos,
                                       model_metas=model_metas,
                                       logprobs=logprobs,
                                       extra_outputs=extra_outputs))
            else:
                # Avoid adding the ADInplaceOrView dispatch key to `next_token_ids`,
                # as it can trigger recompilation on different ranks when using torch.compile.
                next_token_ids, extra_inputs = self.agent_strategy.make_dummy_next_token(
                    inputs, last_logits, extra_inputs)

                # broadcast next token for TP > 1
                with self._broadcast_next_token(next_token_ids, extra_inputs, enable=need_broadcast_next):
                    logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids [{idx}]')

            # update for next loop
            if is_decoding and idx < loop_count - 1:
                inputs, extra_inputs = __update_inputs(next_token_ids, model_metas, extra_inputs)

    async def _async_loop_background(self, forward_event: asyncio.Event = None):
        """Async loop background."""
        with self.all_context(), torch.cuda.stream(self.stream), torch.inference_mode():
            dist_ctx = get_dist_manager().current_context()
            dp = dist_ctx.dp

            # for dp
            if dp > 1:
                input_maker = DPForwardInputsMaker(self)
            else:
                input_maker = DefaultForwardInputsMaker(self)

            while True:
                forward_inputs = await input_maker.get()

                await self._async_step_background(**forward_inputs, )
                if forward_event is not None:
                    forward_event.set()

                input_maker.step()

    async def _async_loop_inputs_preprocess(self, forward_event: asyncio.Event = None):
        """Async loop inputs preprocess."""
        non_blocking = True
        keys = ['inputs', 'sampling_inputs', 'stopping_criteria', 'extra_inputs']
        while True:
            forward_inputs = await self._pre_in_que.get()

            logger.debug('preprocessing forward inputs.')
            with torch.cuda.stream(self.out_stream), torch.inference_mode(), record_function('inputs_H2D'):
                for k in keys:
                    if k not in forward_inputs:
                        continue
                    forward_inputs[k] = _try_to_cuda(forward_inputs[k], non_blocking=non_blocking)
                self.out_stream.synchronize()
            logger.debug('preprocessing forward inputs done.')
            self._in_que.put_nowait(forward_inputs)
            if forward_event is not None:
                forward_event.clear()

    @staticmethod
    def _on_finish_callback(task: asyncio.Task, ptasks: asyncio.Task) -> None:
        """Raise exception on finish."""
        task_name = task.get_name()
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug(f'Task <{task_name}> cancelled.')
            return
        except BaseException:
            logger.exception(f'Task <{task_name}> failed')
        finally:
            for ptask in ptasks:
                if not ptask.done():
                    ptask.cancel()

    def start(self, forward_event: asyncio.Event = None):
        """Start event loop."""
        event_loop = asyncio.get_event_loop()
        self._pre_in_que = asyncio.Queue()
        self._in_que = asyncio.Queue()
        self._out_que = asyncio.Queue()

        tasks_to_cancel = [asyncio.current_task()]

        # forward task
        logger.debug('Create task ModelAgentLoop.')
        self._background_task = event_loop.create_task(self._async_loop_background(forward_event),
                                                       name='ModelAgentLoop')
        tasks_to_cancel.append(self._background_task)

        # preprocess inputs task
        logger.debug('Create task ModelAgentPreprocess.')
        self._preprocess_task = event_loop.create_task(self._async_loop_inputs_preprocess(forward_event),
                                                       name='ModelAgentPreprocess')
        tasks_to_cancel.append(self._preprocess_task)

        # profiler
        self.profiler = AgentProfiler(self.dist_ctx, self.stream)
        self.profiler.create_task()

        # binding done task
        logger.debug('binding done callback.')
        backgroup_done_callback = functools.partial(self._on_finish_callback, ptasks=tasks_to_cancel)
        self._background_task.add_done_callback(backgroup_done_callback)
        preprocess_done_callback = functools.partial(self._on_finish_callback, ptasks=tasks_to_cancel)
        self._preprocess_task.add_done_callback(preprocess_done_callback)

    def stop(self):
        """Stop task."""
        if self.dist_ctx.dp > 1:
            return

        if self.profiler is not None:
            self.profiler.dump()

        if self._background_task is not None:
            if not self._background_task.done():
                self._background_task.cancel()

        if self._preprocess_task is not None:
            if not self._preprocess_task.done():
                self._preprocess_task.cancel()

        if self.guided_decoding_manager:
            self.guided_decoding_manager.clear()

    async def stop_async(self):
        """Stop task."""
        if self.dist_ctx.dp > 1:
            return

        if self.profiler is not None:
            # dirty hack for profiler
            while not self.stream.query():
                logger.debug('Profiler waiting for stream finish.')
                await asyncio.sleep(1)
            self.profiler.dump()

        if self._background_task is not None:
            if not self._background_task.done():
                self._background_task.cancel()
                try:
                    await self._background_task
                except asyncio.CancelledError:
                    logger.debug('ModelAgent background task cancelled.')

        if self._preprocess_task is not None:
            if not self._preprocess_task.done():
                self._preprocess_task.cancel()
                try:
                    await self._preprocess_task
                except asyncio.CancelledError:
                    logger.debug('ModelAgent preprocess task cancelled.')

        if self.guided_decoding_manager:
            self.guided_decoding_manager.clear()

    def set_forward_inputs(self, inputs):
        """Set forward inputs."""
        assert self._pre_in_que is not None, ('Please start backendground task before forward.')
        self._pre_in_que.put_nowait(inputs)

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
            out = out.to_cpu()
            out.new_token_timestamp = time.time()
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
        build_model_ctx = BuildModelContext(disable_vision_encoder=self.misc_config.disable_vision_encoder,
                                            dllm_config=self.misc_config.dllm_config,
                                            strategy_factory=self.strategy_factory)
        patched_model = build_patched_model(self.model_config,
                                            device=device,
                                            model_format=self.misc_config.model_format,
                                            build_model_ctx=build_model_ctx)
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

    def build_graph_runner(self):
        """Build graph runner."""
        with self.all_context():
            backend = get_backend()
            self.patched_model = backend.build_graph_runner(self.patched_model,
                                                            model_config=self.model_config,
                                                            cache_config=self.cache_config,
                                                            backend_config=self.backend_config,
                                                            device=self.device)

    def build_cache_engine(self):
        """Build cache engine."""
        with self.all_context():
            dist_ctx = self.dist_ctx
            attn_dist_cfg = dist_ctx.dist_config.attn_config
            tp = attn_dist_cfg.tp

            self.cache_engine = CacheEngine(self.cache_config,
                                            self.model_config,
                                            rank=self.rank,
                                            tp_rank=self.tp_rank,
                                            world_size=tp,
                                            cache_stream=self.cache_stream)
            self.state_cache_engine = StateCacheEngine(self.cache_config)

    def _forward_impl(self, inputs: ModelInputs):
        output = model_forward(
            self.patched_model,
            inputs,
            self.cache_engine,
            state_cache_engine=self.state_cache_engine,
            stream=self.stream,
        )
        return output

    async def async_forward(self, inputs: ModelInputs):
        """Model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
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
        """Get input processor.."""
        return self.patched_model.get_input_processor()

    def reset_graph_runner(self):
        """Reset graph runner to prevent tp hanging."""
        if hasattr(self.patched_model, 'reset'):
            self.patched_model.reset()

    @torch.inference_mode()
    def update_params(self, request: UpdateParamsRequest):
        """Update params."""

        # modified from https://github.com/vllm-project/vllm/blob/v0.8.5/examples/offline_inference/rlhf_utils.py#L82
        def _construct(item):
            func, args = item
            args = list(args)
            args[6] = torch.cuda.current_device()  # device id.
            # clone() seems necessary otherwise the producer can not release the memory
            return func(*args).clone()

        with self.all_context():
            serialized_data = request.serialized_named_tensors
            if isinstance(serialized_data, list):
                serialized_data = serialized_data[self.dist_ctx.tp_rank]
            model = self.patched_model.get_model()
            weights = ForkingPickler.loads(base64.b64decode(serialized_data))
            weights = [(k, _construct(v)) for k, v in weights]
            weights = ModelWeightLoader._rename_weights_iterator(weights, model)
            model.load_weights(weights)

            if request.finished:
                for _, mod in model.named_modules():
                    if not hasattr(mod, 'update_weights'):
                        continue
                    mod.update_weights()

            torch.cuda.empty_cache()

    @torch.inference_mode()
    def sleep(self, level: int = 1):
        """Sleep."""
        self.cache_engine = None
        self.reset_graph_runner()
        device = 'cpu' if level == 1 else 'meta'
        self.patched_model.get_model().to(device=device, non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def wakeup(self, tags: Optional[List[str]] = None):
        """Wakeup."""
        if tags is None:
            tags = ['weights', 'kv_cache']
        if 'weights' in tags:
            device = next(self.patched_model.get_model().parameters()).device
            assert device.type in ['cpu', 'meta']
            if device.type == 'cpu':
                self.patched_model.get_model().to(torch.cuda.current_device())
            else:
                # user should update weights after wakeup
                old_empty_init = self.misc_config.empty_init
                self.misc_config.empty_init = True
                self.build_model()
                self.build_graph_runner()
                self.misc_config.empty_init = old_empty_init
        if 'kv_cache' in tags:
            self.build_cache_engine()

    def release(self):
        """release."""
        self.reset_graph_runner()
        self.patched_model = None
        self.cache_engine = None
        torch.cuda.empty_cache()


class DefaultForwardInputsMaker:
    """Default forward inputs maker."""

    def __init__(self, model_agent: BaseModelAgent):
        self._in_que = model_agent._in_que

    async def get(self):
        """get."""
        return await self._in_que.get()

    def step(self):
        """step."""
        # No-op for default maker
        pass


class DPForwardInputsMaker:
    """Dp forward inputs maker."""

    def __init__(self, model_agent: BaseModelAgent):
        self.model_agent = model_agent
        self.dist_ctx = model_agent.dist_ctx
        self.model_config = model_agent.model_config
        self.cache_config = model_agent.cache_config
        self.misc_config = model_agent.misc_config
        self.inputs_strategy = model_agent.inputs_strategy
        self.device = model_agent.device
        self._in_que = model_agent._in_que

        # maker metas
        self._next_inputs = None
        self._is_decoding = False
        self._ready_event = torch.cuda.Event()

    def _make_dummy_forward_inputs(self):
        """Make dummy forward inputs."""
        is_decoding = self._is_decoding
        loop_count = self.misc_config.prefill_interval if is_decoding else 1
        dist_config = self.dist_ctx.dist_config
        batch_size = 2 if dist_config.enable_microbatch else 1
        batch_size = min(self.cache_config.max_batches, batch_size)
        model_inputs = self.inputs_strategy.make_dummy(batch_size,
                                                       is_decoding,
                                                       device=self.device,
                                                       vocab_size=self.model_config.vocab_size)
        forward_inputs = dict(
            inputs=model_inputs,
            loop_count=loop_count,
            is_dummy=True,
            sync_long_context=False,
        )
        return forward_inputs

    def _update_is_decoding(self, forward_inputs):
        """Update is decoding."""
        model_inputs = forward_inputs['inputs']
        assert model_inputs.is_decoding == self._is_decoding
        if self.cache_config.role != EngineRole.Prefill:
            self._is_decoding = not self._is_decoding

    async def get(self):
        """get."""
        if self._next_inputs is not None:
            forward_inputs = self._next_inputs
            self._next_inputs = None
            self._update_is_decoding(forward_inputs)
            return forward_inputs

        # wait until has inputs or prev forward finish
        while self._in_que.qsize() == 0 and not self._ready_event.query():
            await asyncio.sleep(0.001)

        # try get inputs
        need_dummy = True
        try:
            forward_inputs = await asyncio.wait_for(self._in_que.get(), timeout=0.02)
            model_inputs = forward_inputs['inputs']
            if model_inputs.is_decoding != self._is_decoding:
                self._next_inputs = forward_inputs
            else:
                need_dummy = False
        except asyncio.TimeoutError:
            pass

        # make dummy inputs
        if need_dummy:
            forward_inputs = self._make_dummy_forward_inputs()

        self._update_is_decoding(forward_inputs)

        return forward_inputs

    def step(self):
        """step."""
        self._ready_event = torch.cuda.Event()
        self._ready_event.record()


def build_model_agent(model_path: str,
                      model_config: ModelConfig,
                      cache_config: CacheConfig,
                      backend_config: BackendConfig,
                      misc_config: MiscConfig,
                      dist_ctx: DistContext = None,
                      device_ctx: DeviceContext = None,
                      adapters: Dict[str, str] = None):
    """Create model agent.

    Args:
        model_path (str): the path of the input model
        cache_config (CacheConfig): config of kv cache
        backend_config (BackendConfig): config of backend devices
        trust_remote_code (bool): To use the remote modeling code or not
        adapters (Dict): lora adapters
        tp (int): the number of devices to be used in tensor parallelism
        dtype (str): the data type of model weights and activations
        custom_module_map (str): customized nn module map
    """
    if device_ctx is None:
        device_mgr = get_device_manager()
        device_ctx = device_mgr.current_context()
    if dist_ctx is None:
        dist_mgr = get_dist_manager()
        dist_ctx = dist_mgr.current_context()

    model_agent = BaseModelAgent(
        model_path,
        model_config=model_config,
        cache_config=cache_config,
        backend_config=backend_config,
        misc_config=misc_config,
        adapters=adapters,
        dist_ctx=dist_ctx,
        device_ctx=device_ctx,
    )
    return model_agent
