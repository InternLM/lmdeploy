# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import atexit
import concurrent.futures
import dataclasses
import random
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import partial
from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from lmdeploy.archs import get_model_arch
from lmdeploy.logger import RequestLogger
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig, Response, ResponseType, SpeculativeConfig,
                               TurbomindEngineConfig)
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.metrics.stats import IterationStats, RequestStats, SpeculativeDecodingStats
from lmdeploy.model import ChatTemplateConfig, get_chat_template
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
from lmdeploy.serve.managers import RequestHandleManager, SessionManager
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import _get_and_verify_max_len, _stop_words, get_hf_gen_cfg, get_logger

from .exceptions import SafeRunException
from .mixin import LogitsMixin

logger = get_logger('lmdeploy')


@dataclasses.dataclass
class GenOut:
    """Pack all response information together."""
    response: str
    history_token_len: int
    input_token_len: int
    generate_token_len: int
    finish_reason: Optional[Literal['stop', 'length', 'error']] = None
    token_ids: List[int] = None
    logprobs: List[Dict[int, float]] = None
    logits: Any = None
    last_hidden_state: Any = None
    cache_block_ids: List[int] = None  # for disaggregation
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


class _EventLoopThread:

    def __init__(self, daemon=False):
        fut = concurrent.futures.Future()
        self.thread = Thread(target=partial(self._thread_entry, fut), daemon=daemon)
        self.thread.start()
        self.loop: asyncio.AbstractEventLoop = fut.result()
        self.closed = False
        if daemon:
            atexit.register(self.close)

    def _thread_entry(self, fut):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut.set_result(loop)
        try:
            loop.run_forever()
        except BaseException as e:
            logger.error(f'[internal_thread] {type(e).__name__} {e}')
        finally:
            try:
                self._cancel_all_tasks()
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _cancel_all_tasks(self):
        """Modified from asyncio/runners.py."""
        to_cancel = asyncio.all_tasks(self.loop)
        if not to_cancel:
            return

        for task in to_cancel:
            task.cancel()

        async def _gather():
            await asyncio.gather(*to_cancel, return_exceptions=True)

        self.loop.run_until_complete(_gather())

        for task in to_cancel:
            if task.cancelled():
                continue
            if task.exception() is not None:
                self.loop.call_exception_handler({
                    'message': 'unhandled exception during worker thread shutdown',
                    'exception': task.exception(),
                    'task': task,
                })

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


class AsyncEngine(LogitsMixin):
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
                 model_name: Optional[str] = None,
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                 backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
                 chat_template_config: Optional[ChatTemplateConfig] = None,
                 max_log_len: int = None,
                 speculative_config: SpeculativeConfig = None,
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
        self.internal_thread = _EventLoopThread(daemon=True)
        self.limiter: asyncio.Semaphore = None
        self.num_spec_token = 0 if backend == 'turbomind' or speculative_config is None \
            else speculative_config.num_speculative_tokens

        # Initialize inference instance manager to handle instance lifecycle
        self.req_hnd_mgr = RequestHandleManager(self.engine, self.backend_config.max_batch_size)
        self.session_mgr = SessionManager()

        # build stat loggers
        self._build_stat_loggers()
        self.epoch = 0

    def close(self):
        self.internal_thread.close()
        self.req_hnd_mgr.clear()
        self.session_mgr.clear()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _build_turbomind(self, model_path: str, backend_config: TurbomindEngineConfig = None, **kwargs):
        """Inner build method for turbomind backend."""
        from lmdeploy import turbomind as tm
        return tm.TurboMind.from_pretrained(model_path, engine_config=backend_config, **kwargs)

    def _build_pytorch(self,
                       model_path: str,
                       backend_config: PytorchEngineConfig = None,
                       speculative_config: SpeculativeConfig = None,
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

    async def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        logger.info(f'stop session {session_id}')
        await self.session_mgr.async_abort(session_id)

    async def end_session(self, session_id: int):
        """For ending a session that is not running."""
        logger.info(f'end session {session_id}')
        await self.session_mgr.async_end(session_id)

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

    def wakeup(self, tags: Optional[List[str]] = None):
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
            self.req_hnd_mgr.rebuild(self.engine)
        self.sleeping_tags = self.sleeping_tags - set(tags)
        self.is_sleeping = bool(self.sleeping_tags)

    def _get_limiter(self):
        if not self.limiter:
            self.limiter = asyncio.Semaphore(self.backend_config.max_batch_size)
        return self.limiter

    def _infer(self, requests: Iterator[Dict], multiplex: bool, pbar=None, loop=None) -> Iterator[Iterator[Response]]:

        async def _sync_resp(g, que: Queue, idx: int, sem: asyncio.Semaphore):
            async for out in g:
                que.put(out.to_response(idx))
            sem.release()
            if not multiplex:
                que.put(None)  # sentinel of inner generator
            if pbar:
                pbar.update(1)

        que = Queue()

        async def _infer():
            sem = self._get_limiter()
            tasks = []
            for idx, req in enumerate(requests):
                await sem.acquire()
                gen = self.generate(**req)
                dst = que if multiplex else Queue()
                if not multiplex:
                    que.put(iter(dst.get, None))
                # create a task to send the responses
                task = asyncio.create_task(_sync_resp(gen, dst, idx, sem))
                tasks.append(task)
            if not multiplex:  # sentinel of outer generator
                que.put(None)
            await asyncio.gather(*tasks)
            if multiplex:
                que.put(None)  # sentinel of inner generator

        loop = loop or self.internal_thread.loop
        # submit the coroutine to async world
        asyncio.run_coroutine_threadsafe(_infer(),
                                         loop).add_done_callback(lambda f: None if f.cancelled() else f.result())

        return iter(que.get, None)

    def _determine_gen_config(self,
                              session,
                              input_ids,
                              gen_config: Optional[GenerationConfig] = None) -> GenerationConfig:
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
    async def safe_run(self, handle, session_id, **kwargs):
        generator = handle.async_stream_infer(session_id, **kwargs)
        try:
            yield generator
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:  # noqa
            logger.error(f'[safe_run] session {session_id} exception caught: {type(e).__name__} {e}')
            # TODO: remove session_id from async cancel
            await handle.async_cancel(session_id)
            raise SafeRunException(f'Safe run exception for session {session_id}') from e
        finally:
            await generator.aclose()

    async def generate(
            self,
            messages,
            session_id: int,
            gen_config: Optional[GenerationConfig] = None,
            tools: Optional[List[object]] = None,
            reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None,
            stream_response: bool = True,
            sequence_start: bool = True,
            sequence_end: bool = True,  # no interactive mode by default
            step: int = 0,
            do_preprocess: bool = True,
            adapter_name: Optional[str] = None,
            rewind_stop_tokens: bool = False,
            input_ids: Optional[List] = None,
            enable_thinking: Optional[bool] = None,
            chat_template_kwargs: Optional[Dict] = None,
            mm_processor_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            session_id (int): the session id
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
        session = self.session_mgr.get(session_id, step=step)
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
            self.request_logger.log_prompt(session_id=session_id, prompt=prompt)
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
            self.request_logger.log_inputs(session_id=session_id,
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
                    await self.end_session(session_id)
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

        async with session.acquire_request_handle(self.req_hnd_mgr) as handle:
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
                                     session_id=session_id,
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
                    # note: Using session_mgr.async_end(session) here results in deadlock
                    # because it waits for session's _active event to be set, but the event won't be set
                    # until the session is finished, i.e., session.acuqire_request_handle() context exits.
                    await handle.async_end(session_id)
                self.session_mgr.sessions.pop(session_id)

    def _run(self, fn=None, coro=None, loop=None):
        assert (fn or coro) and not (fn and coro)
        loop = loop or self.internal_thread.loop
        if fn:

            async def _coro():
                return fn()

            coro = _coro()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def start_loop(self, use_async_api=False):
        """Start engine loop.

        When using pytorch backend with dp > 1, all dp_rank should receive at least one request before it can start
        processing (warmup). Since pytorch engine will bound to event loop, the pipeline can only choose either the
        synchronous apis(__call__, stream_infer, etc.) or the asynchronous api (generate) during its lifetime.

        The purpose of this function is to allow users to choose whether to use the synchronous interface or the
        asynchronous interface for the pipeline.
        """
        if hasattr(self.engine, 'start_loop'):
            if use_async_api:
                return self.engine.start_loop()
            else:
                fut = concurrent.futures.Future()

                def _start_loop(fut):
                    res = self.engine.start_loop()
                    fut.set_result(res)

                self.internal_thread.loop.call_soon_threadsafe(_start_loop, fut)
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
