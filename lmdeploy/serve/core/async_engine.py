# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import concurrent.futures
import dataclasses
import random
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, Dict, List, Literal

import torch

from lmdeploy.archs import get_model_arch
from lmdeploy.logger import RequestLogger
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig, Response, ResponseType, SpeculativeConfig,
                               TurbomindEngineConfig)
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.metrics.stats import IterationStats, RequestStats, SpeculativeDecodingStats
from lmdeploy.model import ChatTemplateConfig, get_chat_template
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
from lmdeploy.serve.managers import Session, SessionManager
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import _get_and_verify_max_len, _stop_words, get_hf_gen_cfg, get_logger

from .exceptions import SafeRunException

logger = get_logger('lmdeploy')


@dataclasses.dataclass
class GenOut:
    """Pack all response information together."""
    response: str
    history_token_len: int
    input_token_len: int
    generate_token_len: int
    finish_reason: Literal['stop', 'length', 'error'] | None = None
    token_ids: List[int] | None = None
    logprobs: List[Dict[int, float]] | None = None
    logits: Any = None
    last_hidden_state: Any = None
    cache_block_ids: List[int] | None = None  # for disaggregation
    routed_experts: Any = None  # for RL router replay

    def to_response(self, index: int = 0) -> Response:
        """Convert GenOut to Response object.

        Args:
            index: The index position in the batch. Default to 0.
        """
        return Response(text=self.response,
                        generate_token_len=self.generate_token_len,
                        input_token_len=self.input_token_len,
                        finish_reason=self.finish_reason,
                        token_ids=self.token_ids or [],
                        logprobs=self.logprobs,
                        last_hidden_state=self.last_hidden_state,
                        logits=self.logits,
                        routed_experts=self.routed_experts,
                        index=index)


# class AsyncEngine(LogitsMixin):
class AsyncEngine:
    """Async inference engine. Maintaining a bunch of tm_model instances.

    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm/internlm-chat-7b",
            "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat" and so on.
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        max_log_len (int): Max number of prompt characters or prompt tokens
            being printed in log. Default: Unlimited
    """

    def __init__(self,
                 model_path: str,
                 model_name: str | None = None,
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                 backend_config: TurbomindEngineConfig | PytorchEngineConfig | None = None,
                 chat_template_config: ChatTemplateConfig | None = None,
                 max_log_len: int | None = None,
                 speculative_config: SpeculativeConfig | None = None,
                 **kwargs) -> None:
        logger.info(f'input backend={backend}, backend_config={backend_config}')
        logger.info(f'speculative_config={speculative_config}')
        backend_config = backend_config or (TurbomindEngineConfig()
                                            if backend == 'turbomind' else PytorchEngineConfig())
        self.model_name = model_name if model_name else model_path
        self.chat_template = get_chat_template(model_path, chat_template_config)
        self.tokenizer = Tokenizer(model_path)
        self.prompt_processor = MultimodalProcessor(self.tokenizer, self.chat_template)
        self.hf_gen_cfg = get_hf_gen_cfg(model_path)
        self.arch, self.hf_cfg = get_model_arch(model_path)
        self.session_len = (_get_and_verify_max_len(self.hf_cfg, None)
                            if backend_config.session_len is None else backend_config.session_len)
        backend_config.session_len = self.session_len
        if speculative_config is not None and backend == 'turbomind':
            logger.warning('speculative decoding is not supported by turbomind ')
        # build backend engine
        if backend == 'turbomind':
            self.engine = self._build_turbomind(model_path=model_path, backend_config=backend_config, **kwargs)
        elif backend == 'pytorch':
            self.engine = self._build_pytorch(model_path=model_path,
                                              backend_config=backend_config,
                                              speculative_config=speculative_config,
                                              **kwargs)
        else:
            raise ValueError(f'unsupported backend {backend}')
        self.backend_config = self.engine.engine_config
        self.is_sleeping = backend_config.empty_init
        self.sleeping_tags: set[str] = set() if not backend_config.empty_init else {'weights', 'kv_cache'}
        logger.info(f'updated backend_config={self.backend_config}')

        # parameters for member functions
        self.stop_words = _stop_words(self.chat_template.stop_words, self.tokenizer)
        if self.stop_words is not None:
            self.stop_words = self.stop_words[0][0].tolist()
        self.backend = backend
        self.request_logger = RequestLogger(max_log_len)

        self.num_spec_token = 0 if backend == 'turbomind' or speculative_config is None \
            else speculative_config.num_speculative_tokens

        self.session_mgr = SessionManager()
        self.session_mgr.build_request_handle_pool(self.engine, self.backend_config.max_batch_size)

        # build stat loggers
        self._build_stat_loggers()
        self.epoch = 0

    def close(self):
        self.session_mgr.clear()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _build_turbomind(self, model_path: str, backend_config: TurbomindEngineConfig | None = None, **kwargs):
        """Inner build method for turbomind backend."""
        from lmdeploy import turbomind as tm
        return tm.TurboMind.from_pretrained(model_path, engine_config=backend_config, **kwargs)

    def _build_pytorch(self,
                       model_path: str,
                       backend_config: PytorchEngineConfig | None = None,
                       speculative_config: SpeculativeConfig | None = None,
                       **kwargs):
        """Inner build method for pytorch backend."""
        from lmdeploy.pytorch.engine import Engine
        return Engine.from_pretrained(model_path, engine_config=backend_config, speculative_config=speculative_config)

    def _build_stat_loggers(self):
        self.stat_loggers = []

        if getattr(self.backend_config, 'enable_metrics', False):
            from lmdeploy.metrics.loggers import LoggingStatLogger, PrometheusStatLogger

            # currently, metrics in TM engine doesn't support dp
            dp_rank = self.backend_config.dp_rank if self.backend == 'pytorch' else 0

            logger.info(f'enable metrics, with dp: {self.backend_config.dp} dp_rank: {dp_rank}')
            self.stat_loggers = [
                LoggingStatLogger(dp_rank=dp_rank),
                PrometheusStatLogger(model_name=self.model_name, max_model_len=self.session_len, dp_rank=dp_rank)
            ]

            # set stats loggers of metrics processor
            metrics_processor.stat_loggers = self.stat_loggers

    def get_schedule_metrics(self):
        return self.engine.get_schedule_metrics()

    async def do_log_stats(self):
        """Loop through CLI logger and Prometheus logger and output the
        metrics."""
        for stat_logger in self.stat_loggers:
            stat_logger.log()

    async def stop_all_session(self):
        """Stop all running sessions."""
        logger.info('stop all sessions')
        self.epoch += 1
        await self.session_mgr.async_abort_all()

    def sleep(self, level: int = 1):
        """Sleep the model.

        Args:
            level (int): The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache. Level 2 sleep will
                discard both the model weights and the kv cache.
        """
        self.engine.sleep(level)
        self.sleeping_tags = {'weights', 'kv_cache'}
        self.is_sleeping = True

    def wakeup(self, tags: List[str] | None = None):
        """Wake up the model.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                engine is used again.
        """
        tags = tags or list(self.sleeping_tags)
        if any(tag not in self.sleeping_tags for tag in tags):
            logger.warning(f'some tag in {tags} not in sleeping tags {self.sleeping_tags}')
            return
        self.engine.wakeup(tags)
        # for TM backend, sleep/wakeup will reset gateway, therefore we need to rebuild instances
        if self.backend == 'turbomind' and 'kv_cache' in tags:
            self.session_mgr.build_request_handle_pool(self.engine, self.backend_config.max_batch_size)
        self.sleeping_tags = self.sleeping_tags - set(tags)
        self.is_sleeping = bool(self.sleeping_tags)

    def _determine_gen_config(self, session, input_ids, gen_config: GenerationConfig | None = None) -> GenerationConfig:
        """Determine the generation configuration."""
        gen_config = deepcopy(gen_config) or GenerationConfig()
        gen_config.convert_stop_bad_words_to_ids(self.tokenizer)
        gen_config.stop_token_ids = gen_config.stop_token_ids or self.stop_words
        gen_config.update_from_hf_gen_cfg(self.hf_gen_cfg, self.tokenizer.eos_token_id)
        if not gen_config.do_sample:
            # greedy decode
            gen_config.top_k = 1
            # avoid unnecessary process
            gen_config.temperature = 1.0
            gen_config.repetition_penalty = 1.0
        # set random if it is not set and sequence_start is True
        elif gen_config.random_seed is None and session.step == 0:
            gen_config.random_seed = random.getrandbits(64)
        if gen_config.n > 1:
            logger.warning(f'n({gen_config.n}) > 1 hasn\'t been supported yet. Fallback to 1')
            gen_config.n = 1
        if gen_config.max_new_tokens is None:
            gen_config.max_new_tokens = max(0, self.session_len - session.step - len(input_ids))
        return gen_config

    @asynccontextmanager
    async def safe_run(self, handle, session, **kwargs):
        generator = handle.async_stream_infer(session.session_id, **kwargs)
        try:
            yield generator
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:  # noqa
            logger.error(f'[safe_run] session {session.session_id} exception caught: {type(e).__name__} {e}')
            await session.async_abort()
            if self.backend == 'pytorch':
                await handle.async_end(session.session_id)
            raise SafeRunException(f'Safe run exception for session {session.session_id}') from e
        finally:
            await generator.aclose()

    async def generate(
            self,
            messages,
            session_id: int | Session,
            gen_config: GenerationConfig | None = None,
            tools: List[object] | None = None,
            reasoning_effort: Literal['low', 'medium', 'high'] | None = None,
            stream_response: bool = True,
            sequence_start: bool = True,
            sequence_end: bool = True,  # no interactive mode by default
            step: int = 0,
            do_preprocess: bool = True,
            adapter_name: str | None = None,
            rewind_stop_tokens: bool = False,
            input_ids: List | None = None,
            enable_thinking: bool | None = None,
            chat_template_kwargs: Dict | None = None,
            mm_processor_kwargs: Dict[str, Any] | None = None,
            **kwargs):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            session_id (int | Session): the session id or instance of Session
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            stream_response (bool): whether return responses streamingly
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
        """
        epoch = self.epoch
        if (messages is not None) ^ (input_ids is None):
            raise ValueError('You must specify exactly one of messages or input_ids')
        if isinstance(session_id, Session):
            session = session_id
        elif isinstance(session_id, int):
            session = self.session_mgr.get(session_id, step=step)
        else:
            raise ValueError(f'Invalid session_id: {session_id}. It should be an instance of Session or an integer.')
        session_id = session.session_id
        chat_template_kwargs = chat_template_kwargs or {}
        if enable_thinking is not None:
            logger.warning('enable_thinking is deprecated, use chat_template_kwargs["enable_thinking"] instead')
            if chat_template_kwargs.get('enable_thinking') is None:
                chat_template_kwargs['enable_thinking'] = enable_thinking
            else:
                logger.warning('chat_template_kwargs["enable_thinking"] is already set, '
                               'the value will not be overwritten by enable_thinking')
        if messages:
            prompt = messages
            self.request_logger.log_prompt(session, prompt=prompt)
            prompt_input = await self.prompt_processor.get_prompt_input(prompt=prompt,
                                                                        do_preprocess=do_preprocess,
                                                                        sequence_start=sequence_start,
                                                                        adapter_name=adapter_name,
                                                                        tools=tools,
                                                                        reasoning_effort=reasoning_effort,
                                                                        chat_template_kwargs=chat_template_kwargs,
                                                                        mm_processor_kwargs=mm_processor_kwargs,
                                                                        **kwargs)
            prompt = prompt_input['prompt']
            input_ids = prompt_input['input_ids']
            self.request_logger.log_inputs(session,
                                           prompt=prompt,
                                           prompt_token_ids=input_ids,
                                           gen_config=gen_config,
                                           adapter_name=adapter_name)
        else:
            # TODO(lvhan) VLM doesn't support input_ids as an argument.
            # Figure out a graceful way to handle the invalid input
            prompt_input = dict(input_ids=input_ids)
        if gen_config.max_new_tokens is None:
            max_new_tokens = max(0, self.session_len - session.step - len(input_ids))
            if max_new_tokens == 0:
                logger.error(f'run out of tokens. session={session_id}.')
                yield GenOut(response='',
                             history_token_len=session.step,
                             input_token_len=len(input_ids),
                             generate_token_len=0,
                             finish_reason='length',
                             token_ids=[])
                if sequence_end is True and sequence_start is False:
                    await session.async_close()
                return
        if self.backend_config.enable_prefix_caching and (gen_config.output_last_hidden_state == 'all'
                                                          or gen_config.output_logits == 'all'):
            errmsg = ('lmdeploy does not support outputting all token\'s logits or last_hidden_state '
                      'when prefix caching is ON')
            yield GenOut(response=errmsg,
                         history_token_len=session.step,
                         input_token_len=len(input_ids),
                         generate_token_len=0,
                         finish_reason='error',
                         token_ids=[])
            return
        logger.info(f'session={session_id}, '
                    f'history_tokens={session.step}, '
                    f'input_tokens={len(input_ids)}, '
                    f'max_new_tokens={gen_config.max_new_tokens}, '
                    f'seq_start={sequence_start}, seq_end={sequence_end}, '
                    f'step={step}, prep={do_preprocess}')

        def is_error(status):
            return status not in [ResponseType.SUCCESS, ResponseType.FINISH, ResponseType.CANCEL]

        gen_config = self._determine_gen_config(session, input_ids, gen_config=gen_config)

        stop_ids = []
        if not gen_config.ignore_eos:
            stop_ids = gen_config.stop_token_ids or []

        metrics_processor.increment_total_requests()

        async with session.request_handle() as handle:
            if epoch != self.epoch:
                logger.debug(f'[generate] session {session_id} got aborted before starting inference')
                # TODO(lvhan): metrics_processor.increment_failed_requests('abort')
                metrics_processor.increment_finished_requests()
                yield GenOut(response='',
                             history_token_len=0,
                             input_token_len=len(input_ids),
                             generate_token_len=0,
                             finish_reason='abort',
                             token_ids=[])
                return
            token_ids = input_ids.copy()
            history_len = session.step
            input_len = len(input_ids)
            output_len, gen_len = 0, 0
            state = DetokenizeState(input_len)
            response = ''
            finish_reason = None
            async with self.safe_run(handle,
                                     session=session,
                                     **prompt_input,
                                     gen_config=gen_config,
                                     adapter_name=adapter_name,
                                     stream_output=stream_response,
                                     sequence_start=sequence_start,
                                     sequence_end=sequence_end,
                                     step=history_len) as gen:
                logger.debug(f'[generate] session {session_id} started')
                hit_stop_token = 0
                req_stats = RequestStats(prompt_tokens=input_len)  # per-request stats
                async for outputs in gen:
                    iteration_stats = IterationStats()  # per-iteration stats
                    specdecode_stats = SpeculativeDecodingStats(
                        self.num_spec_token) if self.num_spec_token > 0 else None
                    metrics_processor.queue_update((outputs, req_stats, iteration_stats, specdecode_stats))
                    # decode res
                    if is_error(outputs.status):
                        break

                    output_len = len(outputs.token_ids)
                    if hit_stop_token or output_len == 0:
                        continue

                    # This assumes the engine will stop when stop token is hit
                    if output_len and outputs.token_ids[-1] in stop_ids:
                        hit_stop_token = 1

                    token_ids += outputs.token_ids[:output_len - hit_stop_token]
                    gen_len = len(token_ids) - input_len

                    ids_offset = state.ids_offset
                    response, state = self.tokenizer.detokenize_incrementally(
                        token_ids,
                        state,
                        skip_special_tokens=gen_config.skip_special_tokens,
                        spaces_between_special_tokens=gen_config.spaces_between_special_tokens)
                    res = token_ids[ids_offset:]

                    out = GenOut(response,
                                 history_len,
                                 input_len,
                                 gen_len,
                                 finish_reason,
                                 token_ids=res,
                                 routed_experts=outputs.routed_experts,
                                 cache_block_ids=outputs.cache_block_ids)
                    if outputs.logprobs is not None:
                        out.logprobs = (outputs.logprobs[:-hit_stop_token] if hit_stop_token else outputs.logprobs)
                    if outputs.last_hidden_state is not None:
                        out.last_hidden_state = (outputs.last_hidden_state[:-hit_stop_token]
                                                 if hit_stop_token else outputs.last_hidden_state)
                    if outputs.logits is not None:
                        out.logits = (outputs.logits[:-hit_stop_token] if hit_stop_token else outputs.logits)
                    yield out
                # end of generator loop
                metrics_processor.increment_finished_requests()

                if not is_error(outputs.status):
                    if outputs.status == ResponseType.CANCEL:
                        finish_reason = 'abort'
                    else:
                        finish_reason = 'stop' if outputs.token_ids[-1] in stop_ids else 'length'

                    # utf-8 char at the end means it's a potential unfinished byte sequence
                    if not response.endswith('�'):
                        # avoid returning the last response twice
                        response = ''
                    token_ids, logits, last_hidden_state, logprobs = [], None, None, None
                    if gen_config.include_stop_str_in_output and finish_reason == 'stop':
                        # return the eos token id (MUST be in a list), eos string, eos token's logits and so on
                        token_ids = outputs.token_ids[-1:]
                        response = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                        logits = outputs.logits[-1:] if outputs.logits is not None else None
                        last_hidden_state = outputs.last_hidden_state[-1:] if outputs.last_hidden_state else None
                        logprobs = outputs.logprobs[-1:] if outputs.logprobs else None
                        gen_len += 1

                    # router replay
                    routed_experts = outputs.routed_experts
                    if routed_experts is not None and not isinstance(routed_experts, str) and (
                            not gen_config.include_stop_str_in_output) and finish_reason == 'stop':
                        routed_experts = routed_experts[:-1]

                    logger.info(f'session {session_id} finished, reason '
                                f'"{finish_reason}", input_tokens '
                                f'{len(input_ids)}, output_tokens {gen_len}')
                    yield GenOut(response,
                                 session.step,
                                 len(input_ids),
                                 gen_len,
                                 finish_reason,
                                 token_ids=token_ids,
                                 logprobs=logprobs,
                                 logits=logits,
                                 last_hidden_state=last_hidden_state,
                                 routed_experts=routed_experts,
                                 cache_block_ids=outputs.cache_block_ids)
                    # Note: We remove the session step update here. Let the caller(e.g., pipeline.chat) take care of it.
                else:
                    logger.error(f'session {session_id} finished, {outputs.status}, '
                                 'reason "error"')
                    yield GenOut(response=f'internal error happened, status code {outputs.status}',
                                 history_token_len=session.step,
                                 input_token_len=len(input_ids),
                                 generate_token_len=0,
                                 finish_reason='error',
                                 token_ids=[])
            # update step
            if sequence_end:
                if self.backend == 'pytorch':
                    # manually end pytorch session
                    # note: Using session.async_abort() here results in deadlock
                    # because it waits for session's _active event to be set, but the event won't be set
                    # until the session is finished, i.e., session.request_handle() context exits.
                    await handle.async_end(session.session_id)
                self.session_mgr.remove(session)
        # if sequence_end:
        #     if self.backend == 'pytorch':
        #         # manually end pytorch session. session cannot be ended until session.request_handle()
        #         # context exits
        #         await session.async_close()
        #     self.session_mgr.remove(session)

    def start_loop(self, loop, use_async_api=False):
        """Start engine loop.

        When using pytorch backend with dp > 1, all dp_rank should receive at least one request before it can start
        processing (warmup). Since pytorch engine will bound to event loop, the pipeline can only choose either the
        synchronous apis(__call__, stream_infer, etc.) or the asynchronous api (generate) during its lifetime.

        The purpose of this function is to allow users to choose whether to use the synchronous interface or the
        asynchronous interface for the pipeline.
        """
        self.session_mgr.attach_event_loop(loop)
        if hasattr(self.engine, 'start_loop'):
            if use_async_api:
                return self.engine.start_loop()
            else:
                fut = concurrent.futures.Future()

                def _start_loop(fut):
                    res = self.engine.start_loop()
                    fut.set_result(res)

                loop.call_soon_threadsafe(_start_loop, fut)
                return fut.result()
        else:
            return True

    """ DistServe Async Engine API Begin """

    def free_cache(self, session_id: int):
        if self.engine.end_session(session_id):
            logger.debug(f'successfully free session {session_id}')
        else:
            logger.warning(f'Invalid Free session {session_id}.')

    def p2p_initialize(self, init_request: DistServeInitRequest):
        return self.engine.p2p_initialize(init_request)

    def p2p_connect(self, conn_request: List[DistServeConnectionRequest]):
        return self.engine.p2p_connect(conn_request)

    def p2p_drop_connect(self, drop_conn_request: List[DistServeDropConnectionRequest]):
        return self.engine.p2p_drop_connect(drop_conn_request)

    """ DistServe Async Engine API End """

    async def async_get_reward_score(self, input_ids: List) -> List[float]:
        """Async version of get_reward_score."""
        supported_reward_models = ['InternLM2ForRewardModel', 'Qwen2ForRewardModel']
        if self.arch not in supported_reward_models:
            raise ValueError(f'{self.arch} is not in reward model list: {supported_reward_models}')
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(isinstance(x, List) for x in input_ids)
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids

        logits = await self.async_get_logits(input_ids=input_ids)

        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    async def async_get_logits(self,
                               input_ids,
                               steps: List[int] | None = None,
                               sequence_start: bool = True,
                               sequence_end: bool = True) -> List[torch.Tensor]:
        assert input_ids and all(isinstance(_, List) for _ in input_ids)
        assert steps is None or (len(steps) == len(input_ids))

        logits = [None] * len(input_ids)

        async def _proc(session, i):
            async with session.request_handle() as handle:
                input_len = len(input_ids[i])
                # TODO(lvhan): Fix the ugly code later on
                max_new_tokens = 1 if self.backend == 'turbomind' else 0
                # The reason to set `top_k=1` is that pt engine crashes at top_k sampling stage
                # when perform inference on a reward model.
                gen_config = GenerationConfig(max_new_tokens=max_new_tokens, output_logits='all', top_k=1)
                async with self.safe_run(handle,
                                         session=session,
                                         input_ids=input_ids[i],
                                         gen_config=gen_config,
                                         stream_output=False,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         step=steps[i] if steps else 0) as gen:
                    async for outputs in gen:
                        pass
                    logits[i] = outputs.logits[:input_len, :]

        sessions = [self.session_mgr.get() for _ in range(len(input_ids))]
        tasks = [_proc(session, i) for i, session in enumerate(sessions)]
        await asyncio.gather(*tasks)
        if sequence_end and self.backend == 'pytorch':
            for session in sessions:
                await session.async_close()
        return logits
