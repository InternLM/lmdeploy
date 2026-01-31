# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from multiprocessing.reduction import ForkingPickler
from os import getenv
from typing import Any, Dict, List, Optional

import numpy as np
import pybase64
import torch
import torch.distributed as dist
from torch.profiler import record_function

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine
from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager
from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor, SamplingInputs, SamplingInputsDelta
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta, step_ctx_manager
from lmdeploy.pytorch.models.patch import BuildModelContext, add_adapters, build_patched_model, update_custom_module_map
from lmdeploy.pytorch.spec_decode import build_spec_agent
from lmdeploy.pytorch.strategies import build_strategy_factory
from lmdeploy.pytorch.strategies.base.model_agent import ExtraInputs, ExtraOutputs, StoppingCriteria
from lmdeploy.pytorch.utils import get_gpu_memory, monkey_patch_hf_modules_cache, wait_for_async_tasks
from lmdeploy.pytorch.weight_loader.model_weight_loader import ModelWeightLoader, load_model_weights
from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import FlattenedTensorBucket, FlattenedTensorMetadata, get_logger

from .inputs_maker import build_inputs_maker
from .profiler import AgentProfiler

logger = get_logger('lmdeploy')


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
            vals = torch.from_numpy(self.vals)
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
    all_routed_experts: Optional[torch.Tensor] = None

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
            cache_config=cache_engine.cache_config,
            kv_caches=cache_engine.gpu_cache,
            state_caches=state_cache_engine.state_caches,
            kv_quant_policy=cache_engine.cache_config.quant_policy,
        )

        with ctx_mgr.context(context):
            model_metas = model.update_model_metas(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            input_dict = model.prepare_inputs_for_generation(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            output = model(**input_dict)
            if not isinstance(output, Dict):
                output = dict(hidden_states=output)
            # InternVL-3.5-Flash will change the seqlen, model_metas during forward
            if context.model_metas is not None and context.model_metas[0] is not None:
                model_metas = context.model_metas
            output['model_metas'] = model_metas
            output['seq_length'] = context.q_seqlens[:len(inputs.seq_length)]
            # for draft model reuse
            output['position_ids'] = context.position_ids
            return output


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


@dataclass
class StepInputs:
    """Step inputs."""
    model_inputs: ModelInputs = None
    extra_inputs: ExtraInputs = None
    stopping_criteria: StoppingCriteria = None
    sampling_delta: SamplingInputsDelta = None

    @record_function('StepInputs.merge')
    def merge(
        self,
        inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas,
        extra_outputs: ExtraOutputs,
        model_agent: 'BaseModelAgent',
    ):
        """Merge prefill inputs."""
        inputs, extra_inputs = model_agent.agent_strategy.update_prefill_for_next_step(
            inputs,
            extra_inputs,
            next_token_ids,
            model_metas,
            extra_outputs,
        )
        stopping_criteria = stopping_criteria.clone()
        sampling_delta = model_agent.sampling_strategy.step_sampling_delta(sampling_delta,
                                                                           next_token_ids,
                                                                           extra_inputs=extra_inputs)
        if self.model_inputs is None:
            self.model_inputs = inputs
            self.extra_inputs = extra_inputs
            self.stopping_criteria = stopping_criteria
            self.sampling_delta = sampling_delta
        else:
            self.model_inputs = model_agent.inputs_strategy.merge(self.model_inputs, inputs)
            self.extra_inputs = self.extra_inputs.merge(extra_inputs)
            self.stopping_criteria = self.stopping_criteria.merge(stopping_criteria)
            self.sampling_delta = model_agent.sampling_strategy.merge_sampling_delta(
                self.sampling_delta, sampling_delta)

    def update_delta(
        self,
        delta: ModelInputsDelta,
        model_agent: 'BaseModelAgent',
    ):
        """Get inputs from delta."""
        self.model_inputs = model_agent.inputs_strategy.update_inputs(self.model_inputs, delta)
        self.extra_inputs = model_agent.agent_strategy.update_extra_inputs(self.extra_inputs, delta)
        self.stopping_criteria = self.stopping_criteria.update(delta)
        self.sampling_delta = model_agent.sampling_strategy.update_sampling_delta(self.sampling_delta, delta)

    @record_function('StepInputs.step')
    def step(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        stopping_criteria: StoppingCriteria,
        sampling_delta: SamplingInputsDelta,
        next_token_ids: torch.Tensor,
        model_metas,
        extra_outputs: ExtraOutputs,
        model_agent: 'BaseModelAgent',
    ):
        """Update inputs."""
        # dp might change is_decoding of decoding inputs
        model_inputs.is_decoding = True
        (
            self.model_inputs,
            self.extra_inputs,
        ) = model_agent.agent_strategy.update_decoding_for_next_step(
            model_inputs,
            next_token_ids=next_token_ids,
            model_metas=model_metas,
            extra_inputs=extra_inputs,
            extra_outputs=extra_outputs,
        )
        self.stopping_criteria = stopping_criteria.clone()
        self.sampling_delta = model_agent.sampling_strategy.step_sampling_delta(sampling_delta,
                                                                                next_token_ids,
                                                                                extra_inputs=extra_inputs)


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
        adapters: Dict[str, str] = None,
        specdecode_config: SpecDecodeConfig = None,
    ):

        self.model_config = model_config
        self.cache_config = cache_config
        # use raw tokenizer
        if dist_ctx.dist_config.world_size > 1:
            monkey_patch_hf_modules_cache()
        self.tokenizer = Tokenizer(model_path).model.model

        # asyncio
        self._pre_in_que = None
        self._in_que = None
        self._out_que = None
        self._background_task = None
        self._preprocess_task = None
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
                                           device=device)
        # sleep wakeup state
        self.state: SleepWakeupState = SleepWakeupState()

        # decoding inputs
        self.step_inputs = StepInputs()

        # long context
        self._prev_chunk_output: Dict = None

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

    def set_model_config(self, model_config: ModelConfig, spec_model_config: ModelConfig = None):
        """Set model config."""
        self.model_config = model_config
        self.spec_agent.set_model_config(spec_model_config)

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

        with self.all_context(), torch.cuda.stream(self.stream):
            max_batches = self.cache_config.max_batches
            world_size = self.dist_config.world_size

            num_tokens = max_batches
            dp = self.dist_config.dp

            if dp > 1:
                # make sure warmup started together
                group = self.dist_ctx.cpu_group
                dist.barrier(group=group)

            # warmup prefill
            inputs = self.inputs_strategy.make_dummy(max_batches,
                                                     is_decoding=False,
                                                     device='cuda',
                                                     vocab_size=self.model_config.vocab_size)
            if dp > 1:
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
            for num_tokens in capture_batch_sizes:
                inputs = self.inputs_strategy.make_dummy(num_tokens,
                                                         is_decoding=True,
                                                         device='cuda',
                                                         vocab_size=self.model_config.vocab_size)
                if dp > 1:
                    num_tokens = inputs.input_ids.numel()
                    inputs.build_dp_meta([num_tokens] * world_size)
                logger.debug(f'Warmup decoding num_tokens={num_tokens} start.')
                self._forward_impl(inputs)
                torch.cuda.synchronize()
                logger.debug(f'Warmup decoding num_tokens={num_tokens} done.')

            # warmup draft model
            self.spec_agent.warmup(max_batches, self.model_config)

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
        origin_inputs = inputs
        ret = await self.async_forward(inputs)

        if not return_logits:
            ret = self._postprocess_forward_output(ret, origin_inputs)

        hidden_states, ret = self.spec_agent.update_main_model_outputs(ret, origin_inputs)

        logits = self.get_logits(hidden_states)
        ret['logits'] = logits
        return ret

    async def async_sampling_logits(self, logits: torch.Tensor, sampling_inputs: SamplingInputs):
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

        # gather dp forward metadata
        batch_size = inputs.seq_length.numel()
        is_sleeping = self.state.is_sleeping
        dp_forward_meta = [int(is_decoding), int(is_dummy), num_tokens, int(is_sleeping)]
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
        group = self.dist_ctx.cpu_group
        device = 'cpu'
        gathered_meta = DistGatherScalar(dp_forward_meta, world_size, device=device, group=group)
        gathered_meta = (await gathered_meta.async_wait()).cpu()

        # check is_decoding
        # if any one of the rank is prefill, then all ranks are prefill
        is_decoding = gathered_meta[:, 0].all().item()
        inputs.is_decoding = is_decoding

        # check if all inputs are dummy inputs
        is_all_dummy = gathered_meta[:, 1].all().item()
        is_all_sleeping = gathered_meta[:, 3].all().item()
        if is_all_dummy:
            return None, is_all_sleeping

        # pad batch size for decoding
        all_num_tokens = gathered_meta[:, 2].tolist()
        if is_decoding:
            max_num_tokens = max(all_num_tokens)
            meta = self.patched_model.get_meta()
            meta.padding_batch_size = max_num_tokens
            logger.debug(f'max_num_tokens={max_num_tokens}')

        # update if enable_microbatch
        if self.enable_microbatch:
            inputs.enable_microbatch = gathered_meta[:, 3].all().item()

        # update dp meta
        inputs.build_dp_meta(all_num_tokens)
        inputs = self.patched_model.update_inputs(inputs)
        return inputs, is_all_sleeping

    def _get_inputs_from_delta(
        self,
        delta: ModelInputsDelta,
        sampling_inputs: SamplingInputs,
    ):
        """Get inputs from delta."""
        self.step_inputs.update_delta(delta, self)
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
            self.step_inputs.update_delta(delta, self)

        if inputs.is_first_chunk:
            self._prev_chunk_output = None

        # check long context
        if self._prev_chunk_output is not None:
            # update model metas
            model_metas = self._prev_chunk_output.get('model_metas')
            inputs.model_metas = model_metas

            if not inputs.is_chunk:
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
                                            all_routed_experts: Any = None,
                                            extra_inputs: ExtraInputs = None):
        """Step postprocess with output."""
        rank = self.rank
        logger.debug(f'<ForwardTask> rank[{rank}]: Sampling.')
        # sampling
        next_token_ids, logprobs = await self.async_sampling_logits(last_logits, sampling_inputs)

        # post sampling
        next_token_ids, extra_inputs = self.agent_strategy.post_sampling(inputs, last_logits, next_token_ids,
                                                                         extra_inputs)

        # spec decoding
        output_token_ids = next_token_ids
        if self.spec_agent.is_enabled():
            extra_inputs = await self.spec_agent.async_model_forward(next_token_ids, inputs, extra_inputs,
                                                                     sampling_inputs)
            next_token_ids = extra_inputs.next_token_ids
            output_token_ids = extra_inputs.output_token_ids
            logits = None

        with self._broadcast_next_token(next_token_ids, extra_inputs, enable=need_broadcast_next):
            logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids')

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
                           extra_outputs=extra_outputs))

        return inputs, extra_inputs, stopping_criteria, extra_outputs, next_token_ids

    async def _step_postprocess_without_output(
        self,
        inputs: ModelInputs,
        last_logits: torch.Tensor,
        extra_inputs: ExtraInputs,
        need_broadcast_next: bool,
    ):
        rank = self.rank
        # Avoid adding the ADInplaceOrView dispatch key to `next_token_ids`,
        # as it can trigger recompilation on different ranks when using torch.compile.
        next_token_ids, extra_inputs = self.agent_strategy.make_dummy_next_token(inputs, last_logits, extra_inputs)

        # broadcast next token for TP > 1
        with self._broadcast_next_token(next_token_ids, extra_inputs, enable=need_broadcast_next):
            logger.debug(f'<ForwardTask> rank[{rank}]: synchronize token ids')

        extra_outputs = self.agent_strategy.make_extra_outputs(extra_inputs)

        return inputs, next_token_ids, extra_inputs, extra_outputs

    async def _async_step(
        self,
        inputs: ModelInputs,
        delta: ModelInputsDelta = None,
        swap_in_map: Dict = None,
        swap_out_map: Dict = None,
        sampling_inputs: SamplingInputs = None,
        stopping_criteria: StoppingCriteria = None,
        return_logits: bool = False,
        return_routed_experts: bool = False,
        extra_inputs: ExtraInputs = None,
    ):
        """Asyc forward task."""

        @record_function('update_decoding_for_next_step')
        def __update_inputs(
            inputs,
            next_token_ids,
            model_metas,
            extra_inputs,
            extra_outputs,
            stopping_criteria,
            sampling_delta: SamplingInputsDelta = None,
        ):
            """Update inputs."""
            # dp might change is_decoding of decoding inputs
            self.step_inputs.step(
                inputs,
                extra_inputs,
                stopping_criteria,
                sampling_delta,
                next_token_ids,
                model_metas,
                extra_outputs,
                model_agent=self,
            )

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

        # dp might change is_decoding in inputs
        is_decoding = inputs.is_decoding
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

        if not is_decoding:
            # init state cache for first time prefill
            # I don't know if this is necessary...
            self.state_cache_engine.init_caches(inputs.state_offsets, inputs.history_lengths == 0)

        # swap caches
        cache_swapping(self.cache_engine, swap_in_map=swap_in_map, swap_out_map=swap_out_map)

        # inference
        logger.debug(f'<ForwardTask> rank[{rank}]: model forward. '
                     f'batch_size={inputs.seq_length.size(0)} '
                     f'num_tokens={inputs.input_ids.size(-1)} '
                     f'is_decoding={inputs.is_decoding}')
        output = await self._async_model_forward(
            inputs,
            return_logits=return_logits,
        )
        # recovery is_decoding
        inputs.is_decoding = is_decoding

        if inputs.is_dummy:
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
            ) = await self._step_postprocess_with_output(
                last_logits,
                logits,
                inputs,
                sampling_inputs,
                stopping_criteria,
                model_metas,
                need_broadcast_next,
                return_logits=return_logits,
                all_routed_experts=all_routed_experts,
                extra_inputs=extra_inputs,
            )
        else:
            (
                inputs,
                next_token_ids,
                extra_inputs,
                extra_outputs,
            ) = await self._step_postprocess_without_output(
                inputs,
                last_logits,
                extra_inputs,
                need_broadcast_next,
            )

        sampling_delta = sampling_inputs.get_delta()
        if need_update_inputs:
            __update_inputs(inputs,
                            next_token_ids,
                            model_metas,
                            extra_inputs,
                            extra_outputs,
                            stopping_criteria,
                            sampling_delta=sampling_delta)
        elif inputs.is_chunk:
            # _prev_chunk_output is used to update model metas
            self._prev_chunk_output = output
        elif self.cache_config.role != EngineRole.Prefill:
            self.step_inputs.merge(
                inputs,
                extra_inputs,
                stopping_criteria,
                sampling_delta,
                next_token_ids,
                model_metas,
                extra_outputs,
                model_agent=self,
            )

    async def _async_loop_background(self, forward_event: asyncio.Event = None):
        """Async loop background."""
        with self.all_context(), torch.cuda.stream(self.stream), torch.inference_mode():

            # for dp
            input_maker = build_inputs_maker(self)

            while True:
                forward_inputs = await input_maker.get()

                await self._async_step(**forward_inputs, )
                if forward_event is not None:
                    forward_event.set()

                input_maker.step()

    async def _async_loop_inputs_preprocess(self, forward_event: asyncio.Event = None):
        """Async loop inputs preprocess."""
        non_blocking = True
        keys = ['inputs', 'delta', 'sampling_inputs', 'stopping_criteria', 'extra_inputs']
        while True:
            forward_inputs = await self._pre_in_que.get()
            forward_inputs_cuda = {}
            forward_inputs_cuda.update(forward_inputs)
            logger.debug('preprocessing forward inputs.')
            with torch.cuda.stream(self.out_stream), torch.inference_mode(), record_function('inputs_H2D'):
                for k in keys:
                    if k not in forward_inputs_cuda:
                        continue
                    forward_inputs_cuda[k] = _try_to_cuda(forward_inputs_cuda[k], non_blocking=non_blocking)
                self.out_stream.synchronize()
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
            event.wait()
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
        # for router replay
        enable_return_routed_experts = self.misc_config.enable_return_routed_experts and self.need_output

        build_model_ctx = BuildModelContext(
            disable_vision_encoder=self.misc_config.disable_vision_encoder,
            dllm_config=self.misc_config.dllm_config,
            strategy_factory=self.strategy_factory,
            enable_return_routed_experts=enable_return_routed_experts,
        )
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
            self.spec_agent.build_model(self.misc_config.empty_init,
                                        self.patched_model,
                                        model_format=self.misc_config.model_format,
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
        """Get input processor."""
        return self.patched_model.get_input_processor()

    def reset_graph_runner(self):
        """Reset graph runner to prevent tp hanging."""
        if hasattr(self.patched_model, 'reset'):
            self.patched_model.reset()

        self.spec_agent.reset_graph_runner()

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
                serialized_data = serialized_data[self.dist_ctx.tp_group.rank]
            model = self.patched_model.get_model()
            weights = ForkingPickler.loads(pybase64.b64decode(serialized_data))
            if request.load_format == 'flattened_bucket':
                metadata: List[FlattenedTensorMetadata] = weights['metadata']
                if metadata:
                    flattened_tensor: torch.Tensor = _construct(weights['flattened_tensor'])
                    bucket = FlattenedTensorBucket(flattened_tensor=flattened_tensor, metadata=metadata)
                    weights = bucket.reconstruct_tensors()
                else:
                    # empty data
                    weights = []
            else:
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
    async def sleep(self, level: int = 1):
        """Sleep."""
        self.state.is_sleeping = True
        await self.state.to_sleep.wait()
        self.cache_engine = None
        self.reset_graph_runner()
        device = 'cpu' if level == 1 else 'meta'
        self.patched_model.get_model().to(device=device, non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.state.to_sleep.clear()

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
            # wake up signal
            self.state.is_sleeping = False
            self.state.to_wakeup.set()

    def release(self):
        """release."""
        self.reset_graph_runner()
        self.patched_model = None
        self.cache_engine = None
        torch.cuda.empty_cache()
