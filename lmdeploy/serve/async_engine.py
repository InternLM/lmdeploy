# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import atexit
import concurrent.futures
import dataclasses
import random
from contextlib import asynccontextmanager, closing
from copy import deepcopy
from functools import partial
from itertools import count
from queue import Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple, Union

import tqdm

from lmdeploy import Tokenizer
from lmdeploy.archs import get_model_arch
from lmdeploy.logger import RequestLogger
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, Response, ResponseType, TurbomindEngineConfig
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.metrics.stats import IterationStats, RequestState
from lmdeploy.model import MODELS, BaseChatTemplate, ChatTemplateConfig, best_match_model
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
from lmdeploy.serve.utils import LogitsMixin
from lmdeploy.tokenizer import DetokenizeState
from lmdeploy.utils import _get_and_verify_max_len, _stop_words, get_hf_gen_cfg, get_logger

logger = get_logger('lmdeploy')


def _merge_message_content(msg: Dict) -> Dict:
    """Merge multimodal content blocks and ensure content field exists.

    This function normalizes message content to match vLLM's behavior:
    1. Missing content field -> add content='' (empty string)
    2. None content -> convert to content='' (empty string)
    3. String content -> return as-is
    4. List content (multimodal) -> merge all text blocks with newline separator

    Args:
        msg: A message dict with 'role' and optionally 'content' field

    Returns:
        A message dict with 'content' field guaranteed to exist

    Note:
        This implementation is based on vLLM's content processing logic.
        vLLM uses "\n".join() to merge multiple text blocks from multimodal content.

    References:
        - vLLM content normalization:
          https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/chat_utils.py
          See _parse_chat_message_content() and _parse_chat_message_content_parts()
        - vLLM text merging logic:
          text_prompt = "\n".join(texts)
    """
    # If content is missing or None, convert to empty string (matches vLLM behavior)
    # This prevents Jinja2 template errors when rendering chat templates
    if 'content' not in msg or msg['content'] is None:
        result = dict(msg)
        result['content'] = ''
        return result

    # If content is already a string, return as-is
    if isinstance(msg['content'], str):
        return msg

    # If content is a list, merge all text blocks into a single string
    # This matches vLLM's behavior: text_prompt = "\n".join(texts)
    content_parts = []
    for block in msg['content']:
        if isinstance(block, dict) and block.get('type') == 'text':
            content_parts.append(block.get('text', ''))
    merged_content = '\n'.join(content_parts)

    # Preserve all other fields in the message (e.g., tool_calls)
    result = dict(msg)
    result['content'] = merged_content
    return result


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

    # for disaggregation
    cache_block_ids: List[int] = None


def _gen_out_to_response(out: GenOut, index) -> Response:
    return Response(text=out.response,
                    generate_token_len=out.generate_token_len,
                    input_token_len=out.input_token_len,
                    finish_reason=out.finish_reason,
                    token_ids=out.token_ids or [],
                    logprobs=out.logprobs,
                    last_hidden_state=out.last_hidden_state,
                    logits=out.logits,
                    index=index)


def _append_response(dst: Response, src: Response):
    """Dst += src."""
    if not dst:
        return src
    dst.text += src.text
    dst.generate_token_len = src.generate_token_len
    dst.input_token_len = src.input_token_len
    dst.finish_reason = src.finish_reason
    dst.index = src.index
    if src.token_ids:
        dst.token_ids += src.token_ids
    if src.logprobs:
        dst.logprobs = dst.logprobs or []
        dst.logprobs += src.logprobs
    return dst


class Session:
    """Session for AsyncEngine.chat.

    Args:
        _id (int): session_id for internal use.
        _step (int): the offset of the k/v cache for internal use.
        _prompt (Any): input prompt for internal use.
        _response (Reaponse): model output for prompt.
        _engine (Any): engine for internal use.
        history (List[Any, str]): chat history.
    """

    def __init__(self, session_id: int, engine: Any, gen_config: GenerationConfig = None):
        self._id: int = session_id
        self._engine = engine
        self._step: int = 0
        self._prompt: Any = None
        self._response: Response = None
        self._gen_config = gen_config
        self.history: List[Tuple[Any, str]] = []

    def _merge_response(self, resp: Response, step: Union[Response, GenOut]):
        """Merge response."""
        resp.text += step.text if isinstance(step, Response) else step.response
        resp.input_token_len = step.input_token_len
        resp.generate_token_len = step.generate_token_len
        resp.finish_reason = step.finish_reason
        return resp

    @property
    def response(self) -> Response:
        """Return response."""
        return self._response

    def close(self):
        """Release engine storage for this session."""
        if self._engine and self._prompt:
            self._engine._run(coro=self._engine.end_session(self._id)).result()
            self._engine = None

    def stop(self):
        if self._engine and self._prompt:
            self._engine._run(coro=self._engine.stop_session(self._id)).result()

    def __repr__(self) -> str:
        res = ''
        for user, assistant in self.history:
            if isinstance(user, list):
                user = str(user)
            res += f'USER: \n{user}\nASSISTANT: \n{assistant}\n'
        return res

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __call__(self,
                 prompt: str,
                 gen_config: Optional[GenerationConfig] = None,
                 stream_response: bool = True,
                 do_preprocess: bool = True,
                 adapter_name: str = None) -> Union[Response, Iterator[Response]]:
        self._engine.chat(prompt,
                          gen_config=gen_config or self._gen_config,
                          stream_response=stream_response,
                          do_preprocess=do_preprocess,
                          session=self,
                          adapter_name=adapter_name)
        if stream_response:
            return self.generator
        else:
            return self.response


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
                 **kwargs) -> None:
        logger.info(f'input backend={backend}, backend_config={backend_config}')
        logger.info(f'input chat_template_config={chat_template_config}')

        backend_config = backend_config or (TurbomindEngineConfig()
                                            if backend == 'turbomind' else PytorchEngineConfig())
        self.model_name = model_name if model_name else model_path
        chat_template_name = best_match_model(model_path)
        if chat_template_config is None:
            chat_template_config = ChatTemplateConfig(chat_template_name, model_path=model_path)
        elif chat_template_config.model_name is None:
            chat_template_config.model_name = chat_template_name
        self.chat_template = chat_template_config.chat_template

        logger.info(f'updated chat_template_onfig={chat_template_config}')

        self.tokenizer = Tokenizer(model_path)
        self.hf_gen_cfg = get_hf_gen_cfg(model_path)
        self.arch, cfg = get_model_arch(model_path)
        self.session_len = (_get_and_verify_max_len(cfg, None)
                            if backend_config.session_len is None else backend_config.session_len)
        backend_config.session_len = self.session_len
        # build backend engine
        if backend == 'turbomind':
            self.engine = self._build_turbomind(model_path=model_path, backend_config=backend_config, **kwargs)
            self.hf_tm_cfg = self.engine.config
        elif backend == 'pytorch':
            self.engine = self._build_pytorch(model_path=model_path, backend_config=backend_config, **kwargs)
            self.hf_tm_cfg = getattr(self.engine.model_config, 'hf_config', None)
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
        self.instance_num = self.backend_config.max_batch_size
        self.id2step = {}
        self.id2inst = {}
        self.free_insts: asyncio.Queue = None
        self.instances = [self.engine.create_instance() for _ in range(self.instance_num)]
        self._session_id = count(0)
        self.request_logger = RequestLogger(max_log_len)
        self.internal_thread = _EventLoopThread(daemon=True)
        self.limiter: asyncio.Semaphore = None

        # build stat loggers
        self._build_stat_loggers()

    def close(self):
        self.internal_thread.close()
        self.free_insts = None
        self.instances.clear()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _get_free_insts(self):
        if self.free_insts is None:
            # `asyncio.Queue` must be created in an async context
            self.free_insts = asyncio.Queue()
            for inst in self.instances:
                self.free_insts.put_nowait(inst)
        return self.free_insts

    def _build_turbomind(self,
                         model_path: str,
                         backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
                         **kwargs):
        """Innter build method for turbomind backend."""
        from lmdeploy import turbomind as tm
        return tm.TurboMind.from_pretrained(model_path, engine_config=backend_config, **kwargs)

    def _build_pytorch(self,
                       model_path: str,
                       backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
                       **kwargs):
        """Innter build method for pytorch backend."""
        from lmdeploy.pytorch.engine import Engine
        return Engine.from_pretrained(model_path, engine_config=backend_config)

    def _build_stat_loggers(self):
        self.stat_loggers = []

        if getattr(self.backend_config, 'enable_metrics', False):
            from lmdeploy.metrics.loggers import LoggingStatLogger, PrometheusStatLogger
            dp_rank = self.backend_config.dp_rank if self.backend_config.dp > 1 else 0

            logger.info(f'enable metrics, with dp: {self.backend_config.dp} dp_rank: {dp_rank}')
            self.stat_loggers = [
                LoggingStatLogger(dp_rank=dp_rank),
                PrometheusStatLogger(model_name=self.model_name, max_model_len=self.session_len, dp_rank=dp_rank)
            ]

            # set stats loggers of metrics processor
            metrics_processor.stat_loggers = self.stat_loggers

    def get_schedule_metrics(self):
        return self.engine.get_schedule_metrics()

    def __call__(self,
                 prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
                 gen_config: Optional[GenerationConfig] = None,
                 do_preprocess: bool = True,
                 adapter_name: Optional[str] = None,
                 use_tqdm: bool = False,
                 **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str] | str | List[Dict] | List[List[Dict]]]): a
            batch of prompts. It accepts: string prompt, a list of string
            prompts, a chat history in OpenAI format or a list of chat
            history.
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
            use_tqdm (bool): Whether use the progress bar. Default to False
        """
        if gen_config is None:
            gen_config = GenerationConfig()
        return self.batch_infer(prompts,
                                gen_config=gen_config,
                                do_preprocess=do_preprocess,
                                adapter_name=adapter_name,
                                use_tqdm=use_tqdm,
                                **kwargs)

    async def do_log_stats(self):
        """Loop through CLI logger and Prometheus logger and output the
        metrics."""
        for stat_logger in self.stat_loggers:
            stat_logger.log()

    async def stop_all_session(self):
        """Stop all running sessions."""
        logger.info('stop all sessions')
        tasks = []
        session_ids = []
        for session_id in list(self.id2inst.keys()):
            generator = self.id2inst.get(session_id)
            if generator:
                session_ids.append(session_id)
                tasks.append(generator.async_cancel(session_id))
        await asyncio.gather(*tasks)
        logger.info(f'all {len(session_ids)} sessions stopped')

    async def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        logger.info(f'stop session {session_id}')
        generator = self.id2inst.get(session_id)
        if generator:
            await generator.async_cancel(session_id)
            logger.info(f'session {session_id} stopped')
        # else it's not running at all

    async def end_session(self, session_id: int):
        """For ending a session that is not running."""
        logger.info(f'end session {session_id}')
        inst = self.id2inst.get(session_id)
        if inst:
            await inst._active.wait()
            assert session_id not in self.id2inst
        inst = await self._get_free_insts().get()
        try:
            await inst.async_end(session_id)
            self.id2step[session_id] = 0
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:  # noqa
            logger.error(f'[end_session] exception caught: {e}')
        finally:
            self._get_free_insts().put_nowait(inst)

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
        # for TM backend, sleep/wakeup will reset gateway, therefore we need to rebuild instance
        if self.backend == 'turbomind' and 'kv_cache' in tags:
            self.instances = [self.engine.create_instance() for _ in range(self.instance_num)]
            self.free_insts = None
        self.sleeping_tags = self.sleeping_tags - set(tags)
        self.is_sleeping = bool(self.sleeping_tags)

    def _get_limiter(self):
        if not self.limiter:
            self.limiter = asyncio.Semaphore(self.instance_num)
        return self.limiter

    async def _async_infer(self, requests: AsyncIterator[Dict], **kwargs) -> AsyncIterator[AsyncIterator[Response]]:
        async for req in requests:
            gen = self.generate(**req, **kwargs)
            yield gen

    def _infer(self, requests: Iterator[Dict], multiplex: bool, pbar=None, loop=None) -> Iterator[Iterator[Response]]:

        async def _sync_resp(g, que: Queue, idx: int, sem: asyncio.Semaphore):
            async for out in g:
                que.put(_gen_out_to_response(out, idx))
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

    @staticmethod
    def _is_single(prompts):
        return isinstance(prompts, str) or isinstance(prompts[0], Dict)

    def infer(self,
              prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
              gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
              do_preprocess: bool = True,
              adapter_name: Optional[str] = None,
              stream_response: bool = False,
              multiplex: bool = False,
              pbar: Optional[tqdm.tqdm] = None,
              **kwargs):

        prompts = [prompts] if AsyncEngine._is_single(prompts) else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        gen_config = gen_config or GenerationConfig()
        if not isinstance(gen_config, List):
            gen_config = [gen_config] * len(prompts)
        assert len(prompts) == len(gen_config), \
                'input gen_confg length differs from the length of prompts'  # noqa

        def requests():
            for prompt, gen_cfg in zip(prompts, gen_config):
                r = dict(messages=prompt,
                         gen_config=gen_cfg,
                         do_preprocess=do_preprocess,
                         adapter_name=adapter_name,
                         stream_response=stream_response,
                         **kwargs)
                r.setdefault('sequence_start', True)
                r.setdefault('sequence_end', True)
                if 'session_id' not in r:
                    r['session_id'] = next(self._session_id)
                yield r

        return self._infer(requests(), multiplex, pbar)

    def batch_infer(self,
                    prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
                    gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
                    do_preprocess: bool = True,
                    adapter_name: Optional[str] = None,
                    use_tqdm: bool = False,
                    **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str] | str | List[Dict] | List[List[Dict]]]): a
            batch of prompts. It accepts: string prompt, a list of string
            prompts, a chat history in OpenAI format or a list of chat
            history.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
            use_tqdm (bool): Whether use the progress bar. Default to False
        """
        is_single = AsyncEngine._is_single(prompts)
        outputs = []
        pbar = tqdm.tqdm(total=1 if is_single else len(prompts)) if use_tqdm else None
        try:
            for g in self.infer(prompts,
                                gen_config,
                                do_preprocess,
                                adapter_name,
                                stream_response=False,
                                pbar=pbar,
                                **kwargs):
                res = None
                for out in g:
                    res = _append_response(res, out)
                outputs.append(res)
        finally:
            if pbar: pbar.close()  # noqa
        if is_single:
            return outputs[0]
        return outputs

    def stream_infer(self,
                     prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
                     gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
                     do_preprocess: bool = True,
                     adapter_name: Optional[str] = None,
                     stream_response: bool = True,
                     **kwargs):
        """Inference a batch of prompts with stream mode.

        Args:
            prompts (List[str] | str | List[Dict] | List[List[Dict]]]):a
            batch of prompts. It accepts: string prompt, a list of string
            prompts, a chat history in OpenAI format or a list of chat
            history.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
        """
        return self.infer(prompts, gen_config, do_preprocess, adapter_name, stream_response, multiplex=True, **kwargs)

    async def _get_prompt_input(self,
                                prompt: str,
                                do_preprocess: bool,
                                sequence_start: bool,
                                adapter_name: str,
                                tools: Optional[List[object]] = None,
                                reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None,
                                enable_thinking: Optional[bool] = None,
                                **kwargs):
        # Change multimodal data to openai text messages, i.e.,
        # [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}] ->
        # [{'role': 'user', 'content': 'hi']
        # Also ensure all messages have 'content' field (set to None if missing, e.g., assistant with tool_calls)
        if isinstance(prompt, list):
            prompt = [_merge_message_content(msg) for msg in prompt]
        if do_preprocess:
            # use adapter's chat template if possible
            chat_template = self.chat_template
            if adapter_name in MODELS.module_dict:
                chat_template = MODELS.module_dict[adapter_name]()
        else:
            chat_template = BaseChatTemplate()
        prompt = chat_template.messages2prompt(prompt,
                                               sequence_start,
                                               tools=tools,
                                               enable_thinking=enable_thinking,
                                               reasoning_effort=reasoning_effort,
                                               **kwargs)
        if prompt is None:
            raise ValueError(
                f'You are using base template to handle chat task. Please specify a `--chat-template` name chosen from `lmdeploy list` if you want to use OpenAI messages input.'  # noqa
            )
        input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)
        return {'prompt': prompt, 'input_ids': input_ids}

    @asynccontextmanager
    async def model_inst(self, session_id: int):
        """A context manager to make sure server's safe running."""
        assert session_id not in self.id2inst
        free_insts = self._get_free_insts()
        inst = await free_insts.get()
        inst._active = asyncio.Event()
        self.id2inst[session_id] = inst
        try:
            yield inst
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:
            logger.error(f'[model_inst] exception caught: {e}')
            if self.backend == 'pytorch':
                # manually end pytorch session
                await inst.async_end(session_id)
        finally:
            self.id2inst.pop(session_id)
            inst._active.set()
            free_insts.put_nowait(inst)

    @asynccontextmanager
    async def safe_run(self, inst, session_id, **kwargs):
        generator = inst.async_stream_infer(session_id, **kwargs)
        try:
            yield generator
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:  # noqa
            logger.error(f'[safe_run] exception caught: {type(e).__name__} {e}')
            # TODO: remove session_id from async cancel
            await inst.async_cancel(session_id)
            raise e
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
        if (messages is not None) ^ (input_ids is None):
            raise ValueError('You must specify exactly one of messages or input_ids')
        if session_id not in self.id2step:
            self.id2step[session_id] = 0
        if step != 0:
            self.id2step[session_id] = step
        if gen_config is None:
            gen_config = GenerationConfig()
        else:
            gen_config = deepcopy(gen_config)
        gen_config.convert_stop_bad_words_to_ids(self.tokenizer)
        if gen_config.stop_token_ids is None:
            gen_config.stop_token_ids = self.stop_words
        gen_config.update_from_hf_gen_cfg(self.hf_gen_cfg, self.tokenizer.eos_token_id)
        if not gen_config.do_sample:
            # greedy decode
            gen_config.top_k = 1
            # avoid unnecessary process
            gen_config.temperature = 1.0
            gen_config.repetition_penalty = 1.0
        # set random if it is not set and sequence_start is True
        elif gen_config.random_seed is None and sequence_start:
            gen_config.random_seed = random.getrandbits(64)
        if gen_config.n > 1:
            logger.warning(f'n({gen_config.n}) > 1 hasn\'t been supported yet. Fallback to 1')
            gen_config.n = 1
        if messages:
            prompt = messages
            self.request_logger.log_prompt(session_id=session_id, prompt=prompt)
            prompt_input = await self._get_prompt_input(prompt,
                                                        do_preprocess,
                                                        sequence_start,
                                                        adapter_name,
                                                        tools=tools,
                                                        reasoning_effort=reasoning_effort,
                                                        enable_thinking=enable_thinking,
                                                        **kwargs)
            prompt = prompt_input['prompt']
            input_ids = prompt_input['input_ids']
            self.request_logger.log_inputs(session_id=session_id,
                                           prompt=prompt,
                                           prompt_token_ids=input_ids,
                                           gen_config=gen_config,
                                           adapter_name=adapter_name)
            logger.info(f'session={session_id}, '
                        f'history_tokens={self.id2step[session_id]}, '
                        f'input_tokens={len(input_ids)}, '
                        f'max_new_tokens={gen_config.max_new_tokens}, '
                        f'seq_start={sequence_start}, seq_end={sequence_end}, '
                        f'step={step}, prep={do_preprocess}')
        else:
            # TODO(lvhan) VLM doesn't support input_ids as an argument.
            # Figure out a graceful way to handle the invalid input
            prompt_input = dict(input_ids=input_ids)

        if gen_config.max_new_tokens is None:
            gen_config.max_new_tokens = max(0, self.session_len - self.id2step[session_id] - len(input_ids))
            if gen_config.max_new_tokens == 0:
                logger.error(f'run out of tokens. session={session_id}.')
                yield GenOut('', self.id2step[session_id], len(input_ids), 0, 'length')
                if sequence_end is True and sequence_start is False:
                    await self.end_session(session_id)
                return
        if self.backend_config.enable_prefix_caching and (gen_config.output_last_hidden_state == 'all'
                                                          or gen_config.output_logits == 'all'):
            errmsg = ('lmdeploy does not support outputting all token\'s logits or last_hidden_state '
                      'when prefix caching is ON')
            yield GenOut(response=errmsg,
                         history_token_len=self.id2step[session_id],
                         input_token_len=len(input_ids),
                         generate_token_len=0,
                         finish_reason='error',
                         token_ids=[])
            return

        def is_error(status):
            return status not in [ResponseType.SUCCESS, ResponseType.FINISH, ResponseType.CANCEL]

        stop_ids = []
        if not gen_config.ignore_eos:
            stop_ids = gen_config.stop_token_ids or []

        metrics_processor.increment_total_requests()
        async with self.model_inst(session_id) as inst:
            token_ids = input_ids.copy()
            history_len = self.id2step[session_id]
            input_len = len(input_ids)
            output_len, gen_len = 0, 0
            state = DetokenizeState(len(input_ids))
            response = ''
            finish_reason = None
            async with self.safe_run(inst,
                                     session_id=session_id,
                                     **prompt_input,
                                     gen_config=gen_config,
                                     adapter_name=adapter_name,
                                     stream_output=stream_response,
                                     sequence_start=sequence_start,
                                     sequence_end=sequence_end,
                                     step=history_len) as gen:
                hit_stop_token = 0
                req_state = RequestState(prompt_tokens=input_len)  # per-requst state
                async for outputs in gen:
                    iteration_stats = IterationStats()  # per-iteration stats
                    metrics_processor.queue_update((outputs, req_state, iteration_stats))
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
                        logits = outputs.logits[-1:] if outputs.logits else None
                        last_hidden_state = outputs.last_hidden_state[-1:] if outputs.last_hidden_state else None
                        logprobs = outputs.logprobs[-1:] if outputs.logprobs else None

                    logger.info(f'session {session_id} finished, reason '
                                f'"{finish_reason}", input_tokens '
                                f'{len(input_ids)}, output_tokens {gen_len}')
                    yield GenOut(response,
                                 self.id2step[session_id],
                                 len(input_ids),
                                 gen_len,
                                 finish_reason,
                                 token_ids=token_ids,
                                 logprobs=logprobs,
                                 logits=logits,
                                 last_hidden_state=last_hidden_state,
                                 cache_block_ids=outputs.cache_block_ids)
                    # Update a session's sequence only when it is in finished status
                    if outputs.status == ResponseType.FINISH:
                        if rewind_stop_tokens:
                            # rewind the step to the token before the stop token
                            output_len = gen_len
                        self.id2step[session_id] += input_len + output_len
                else:
                    logger.error(f'session {session_id} finished, {outputs.status}, '
                                 'reason "error"')
                    yield GenOut(response=f'internal error happened, status code {outputs.status}',
                                 history_token_len=self.id2step[session_id],
                                 input_token_len=len(input_ids),
                                 generate_token_len=0,
                                 finish_reason='error',
                                 token_ids=[])
            # update step
            if sequence_end:
                self.id2step[session_id] = 0
                if self.backend == 'pytorch':
                    # manually end pytorch session
                    await inst.async_end(session_id)

    def _run(self, fn=None, coro=None, loop=None):
        assert (fn or coro) and not (fn and coro)
        loop = loop or self.internal_thread.loop
        if fn:

            async def _coro():
                return fn()

            coro = _coro()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def session(self, gen_config: GenerationConfig = None):
        return Session(self._run(fn=lambda: next(self._session_id)).result(), engine=self, gen_config=gen_config)

    def chat(self,
             prompt: str,
             session=None,
             gen_config: Optional[GenerationConfig] = None,
             stream_response=False,
             adapter_name=None,
             **kwargs) -> Union[Session, Iterator]:
        """Chat.

        Args:
            prompt (str): prompt
            session (Session): the chat session
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            **kwargs (dict): ad hoc parametrization of `gen_config
        """
        if session is None:
            session = self.session()

        # sync & init
        session._prompt = prompt
        session._response = None

        sequence_start = session._step == 0

        generator = self.infer(prompt,
                               gen_config,
                               adapter_name=adapter_name,
                               sequence_start=sequence_start,
                               sequence_end=False,
                               session_id=session._id,
                               stream_response=stream_response,
                               multiplex=True,
                               step=session._step)

        def _gen():
            resp = None
            try:
                for out in generator:
                    resp = _append_response(resp, out)
                    yield out
            except:  # noqa
                self._run(coro=self.stop_session(session._id)).result()
                raise
            else:
                session._response = resp
                session._step += resp.generate_token_len + resp.input_token_len
                session.history.append((session._prompt, resp.text))

        if stream_response:
            session.generator = _gen()
        else:
            # run the generator until finish
            with closing(_gen()) as gen:
                for _ in gen:
                    pass
            session.generator = None

        return session

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
