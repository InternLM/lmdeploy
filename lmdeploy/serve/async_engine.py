# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import atexit
import concurrent.futures
import dataclasses
import os
import random
from contextlib import asynccontextmanager, closing
from copy import deepcopy
from functools import partial
from itertools import count
from queue import Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch
import tqdm

from lmdeploy import Tokenizer
from lmdeploy.archs import get_model_arch
from lmdeploy.logger import RequestLogger
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, Response, ResponseType, TurbomindEngineConfig
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.metrics.stats import IterationStats, RequestState
from lmdeploy.model import MODELS, BaseChatTemplate, ChatTemplateConfig, best_match_model
from lmdeploy.pytorch.disagg.conn.protocol import DistServeConnectionRequest, DistServeInitRequest
from lmdeploy.serve.utils import LogitsMixin
from lmdeploy.tokenizer import DetokenizeState
from lmdeploy.utils import _get_and_verify_max_len, _stop_words, get_hf_gen_cfg, get_logger

logger = get_logger('lmdeploy')


def get_names_from_model(model_path: str, model_name: str = None):
    """Get model name and chat template name from workspace model."""
    triton_model_path = os.path.join(model_path, 'triton_models', 'weights')
    if not os.path.exists(triton_model_path):
        chat_template_name = best_match_model(model_path)
    else:
        # `model_path` refers to a turbomind model, reading
        # chat_template_name from the config
        config_path = os.path.join(triton_model_path, 'config.yaml')
        with open(config_path, 'r') as f:
            import yaml

            config = yaml.safe_load(f)
        chat_template_name = config['model_config']['chat_template']
    model_name = model_name if model_name else model_path
    return model_name, chat_template_name


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
    return Response(
        text=out.response,
        generate_token_len=out.generate_token_len,
        input_token_len=out.input_token_len,
        finish_reason=out.finish_reason,
        token_ids=out.token_ids or [],
        logprobs=out.logprobs,
        last_hidden_state=out.last_hidden_state,
        logits=out.logits,
        index=index,
    )


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
        if self._engine:
            self._engine._run(coro=self._engine.end_session(self._id)).result()
            self._engine = None

    def __repr__(self) -> str:
        res = ''
        for user, assistant in self.history:
            if isinstance(user, list):
                user = str(user)
            res += f"USER:\n{user}\nASSISTANT:\n{assistant}\n"
        return res

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __call__(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        stream_response: bool = True,
        do_preprocess: bool = True,
    ) -> Union[Response, Iterator[Response]]:
        self._engine.chat(
            prompt=prompt,
            gen_config=gen_config or self._gen_config,
            stream_response=stream_response,
            do_preprocess=do_preprocess,
            session=self,
        )
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
            logger.error(f"[internal_thread] {type(e).__name__} {e}")
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

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        backend: Literal['turbomind', 'pytorch'] = 'turbomind',
        backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
        chat_template_config: Optional[ChatTemplateConfig] = None,
        max_log_len: int = None,
        **kwargs,
    ) -> None:
        logger.info(f"input backend={backend}, backend_config={backend_config}")
        logger.info(f"input chat_template_config={chat_template_config}")

        self.model_name, chat_template_name = get_names_from_model(model_path, model_name)
        if chat_template_config is None:
            chat_template_config = ChatTemplateConfig(chat_template_name)
        elif chat_template_config.model_name is None:
            chat_template_config.model_name = chat_template_name
        self.chat_template = chat_template_config.chat_template

        logger.info(f"updated chat_template_onfig={chat_template_config}")

        self.tokenizer = Tokenizer(model_path)
        self.hf_gen_cfg = get_hf_gen_cfg(model_path)
        self.arch, _ = get_model_arch(model_path)

        # build backend engine
        if backend == 'turbomind':
            self._build_turbomind(model_path=model_path, backend_config=backend_config, **kwargs)
        elif backend == 'pytorch':
            self._build_pytorch(model_path=model_path, backend_config=backend_config, **kwargs)
        else:
            raise ValueError(f"unsupported backend {backend}")

        logger.info(f"updated backend_config={self.backend_config}")

        # parameters for member functions
        self.session_len = _get_and_verify_max_len(self.hf_tm_cfg, self.backend_config.session_len)
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
        torch._C._cuda_clearCublasWorkspaces()

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

    def _build_turbomind(
        self,
        model_path: str,
        backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
        **kwargs,
    ):
        """Innter build method for turbomind backend."""
        from lmdeploy import turbomind as tm

        self.engine = tm.TurboMind.from_pretrained(model_path,
                                                   tokenizer=self.tokenizer,
                                                   engine_config=backend_config,
                                                   **kwargs)
        self.backend_config = self.engine.engine_config
        self.hf_tm_cfg = self.engine.config

    def _build_pytorch(
        self,
        model_path: str,
        backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
        **kwargs,
    ):
        """Innter build method for pytorch backend."""
        from lmdeploy.pytorch.engine import Engine
        self.engine = Engine.from_pretrained(model_path, tokenizer=self.tokenizer, engine_config=backend_config)
        self.backend_config = self.engine.engine_config
        self.hf_tm_cfg = getattr(self.engine.model_config, 'hf_config', None)

    def _build_stat_loggers(self):
        self.stat_loggers = []

        if getattr(self.backend_config, 'enable_metrics', False):
            from lmdeploy.metrics.loggers import LoggingStatLogger, PrometheusStatLogger
            dp_rank = self.backend_config.dp_rank if self.backend_config.dp else 0

            logger.info(f'enable metrics, with dp: {self.backend_config.dp} dp_rank: {dp_rank}')
            self.stat_loggers = [
                LoggingStatLogger(dp_rank=dp_rank),
                PrometheusStatLogger(model_name=self.model_name, max_model_len=self.session_len, dp_rank=dp_rank)
            ]

            # set stats loggers of metrics processor
            metrics_processor.stat_loggers = self.stat_loggers

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
        return self.batch_infer(
            prompts,
            gen_config=gen_config,
            do_preprocess=do_preprocess,
            adapter_name=adapter_name,
            use_tqdm=use_tqdm,
            **kwargs,
        )

    async def do_log_stats(self):
        # loop through CLI logger and Prometheus logger
        for stat_logger in self.stat_loggers:
            stat_logger.log()

    async def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        logger.info(f"stop session {session_id}")
        generator = self.id2inst.get(session_id)
        if generator:
            await generator.async_cancel(session_id)
        # else it's not running at all

    async def end_session(self, session_id: int):
        """For ending a session that is not running."""
        logger.info(f"end session {session_id}")
        inst = self.id2inst.get(session_id)
        if inst:
            await inst._active.wait()
            assert session_id not in self.id2inst
        inst = await self._get_free_insts().get()
        try:
            await inst.async_end(session_id)
            self.id2step[session_id] = 0
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:  # noqa
            logger.error(f"[end_session] exception caught: {e}")
        finally:
            self._get_free_insts().put_nowait(inst)

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
        asyncio.run_coroutine_threadsafe(_infer(), loop).add_done_callback(lambda x: x.result())

        return iter(que.get, None)

    @staticmethod
    def _is_single(prompts):
        return isinstance(prompts, str) or isinstance(prompts[0], Dict)

    def infer(
        self,
        prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
        gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
        do_preprocess: bool = True,
        adapter_name: Optional[str] = None,
        stream_response: bool = False,
        multiplex: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
        **kwargs,
    ):

        prompts = [prompts] if AsyncEngine._is_single(prompts) else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        gen_config = gen_config or GenerationConfig()
        if not isinstance(gen_config, List):
            gen_config = [gen_config] * len(prompts)
        assert len(prompts) == len(gen_config), 'input gen_confg length differs from the length of prompts'  # noqa

        def requests():
            for prompt, gen_cfg in zip(prompts, gen_config):
                r = dict(
                    messages=prompt,
                    gen_config=gen_cfg,
                    do_preprocess=do_preprocess,
                    adapter_name=adapter_name,
                    stream_response=stream_response,
                    **kwargs,
                )
                r.setdefault('sequence_start', True)
                r.setdefault('sequence_end', True)
                if 'session_id' not in r:
                    r['session_id'] = next(self._session_id)
                yield r

        return self._infer(requests(), multiplex, pbar)

    def batch_infer(
        self,
        prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
        gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
        do_preprocess: bool = True,
        adapter_name: Optional[str] = None,
        use_tqdm: bool = False,
        **kwargs,
    ):
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
            for g in self.infer(
                    prompts,
                    gen_config,
                    do_preprocess,
                    adapter_name,
                    stream_response=False,
                    pbar=pbar,
                    **kwargs,
            ):
                res = None
                for out in g:
                    res = _append_response(res, out)
                outputs.append(res)
        finally:
            if pbar:
                pbar.close()  # noqa
        if is_single:
            return outputs[0]
        return outputs

    def stream_infer(
        self,
        prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
        gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
        do_preprocess: bool = True,
        adapter_name: Optional[str] = None,
        stream_response: bool = True,
        **kwargs,
    ):
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
        return self.infer(
            prompts,
            gen_config,
            do_preprocess,
            adapter_name,
            stream_response,
            multiplex=True,
            **kwargs,
        )

    async def _get_prompt_input(
        self,
        prompt: str,
        do_preprocess: bool,
        sequence_start: bool,
        adapter_name: str,
        tools: Optional[List[object]] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ):
        if do_preprocess:
            # use adapter's chat template if possible
            chat_template = self.chat_template
            if adapter_name in MODELS.module_dict:
                chat_template = MODELS.module_dict[adapter_name]()
        else:
            chat_template = BaseChatTemplate()
        prompt = chat_template.messages2prompt(prompt, sequence_start, tools=tools, enable_thinking=enable_thinking)
        if prompt is None:
            raise ValueError(
                f"You are using base template to handle chat task. Please specify a `--chat-template` name chosen from `lmdeploy list` if you want to use OpenAI messages input."  # noqa
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
        stream_response: bool = True,
        sequence_start: bool = True,
        sequence_end: bool = True,  # no interactive mode by default
        step: int = 0,
        do_preprocess: bool = True,
        adapter_name: Optional[str] = None,
        skip_stop_tokens: bool = True,
        rewind_stop_tokens: bool = False,
        input_ids: Optional[List] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ):
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
            logger.warning(f"GenerationConfig: {gen_config}")
            logger.warning('Since v0.6.0, lmdeploy add `do_sample` in '
                           'GenerationConfig. It defaults to False, meaning greedy '
                           'decoding. Please set `do_sample=True` if sampling '
                           ' decoding is needed')
            # greedy decode
            gen_config.top_k = 1
            # avoid unnecessary process
            gen_config.temperature = 1.0
            gen_config.repetition_penalty = 1.0
        # set random if it is not set and sequence_start is True
        elif gen_config.random_seed is None and sequence_start:
            gen_config.random_seed = random.getrandbits(64)
        if gen_config.n > 1:
            logger.ERROR(f"n({gen_config.n}) > 1 hasn't been supported yet. "
                         f"Fallback to 1")
            gen_config.n = 1
        if messages:
            prompt = messages
            self.request_logger.log_prompt(session_id=session_id, prompt=prompt)
            prompt_input = await self._get_prompt_input(
                prompt,
                do_preprocess,
                sequence_start,
                adapter_name,
                tools=tools,
                enable_thinking=enable_thinking,
            )
            prompt = prompt_input['prompt']
            input_ids = prompt_input['input_ids']
            self.request_logger.log_inputs(
                session_id=session_id,
                prompt=prompt,
                prompt_token_ids=input_ids,
                gen_config=gen_config,
                adapter_name=adapter_name,
            )
            logger.info(f"session={session_id}, "
                        f"history_tokens={self.id2step[session_id]}, "
                        f"input_tokens={len(input_ids)}, "
                        f"max_new_tokens={gen_config.max_new_tokens}, "
                        f"seq_start={sequence_start}, seq_end={sequence_end}, "
                        f"step={step}, prep={do_preprocess}")
        else:
            # TODO(lvhan) VLM doesn't support input_ids as an argument.
            # Figure out a graceful way to handle the invalid input
            prompt_input = dict(input_ids=input_ids)
        if gen_config.max_new_tokens is None:
            # for interactive endpoint, will try maximum possible token num
            gen_config.max_new_tokens = max(128, self.session_len - self.id2step[session_id] - len(input_ids))
        elif (self.id2step[session_id] + len(input_ids) + gen_config.max_new_tokens > self.session_len):
            gen_config.max_new_tokens = max(self.session_len - self.id2step[session_id] - len(input_ids), 128)
            logger.error(f"Truncate max_new_tokens to {gen_config.max_new_tokens}")
        if (self.id2step[session_id] + len(input_ids) + gen_config.max_new_tokens > self.session_len):
            logger.error(f"run out of tokens. session={session_id}.")
            yield GenOut('', self.id2step[session_id], len(input_ids), 0, 'length')
            if sequence_end is True and sequence_start is False:
                await self.end_session(session_id)
            return

        def is_error(status):
            return status not in [ResponseType.SUCCESS, ResponseType.FINISH]

        # used to skip / rewind stop words in interactive mode
        stop_ids = []
        if skip_stop_tokens and not gen_config.ignore_eos:
            stop_ids = gen_config.stop_token_ids or []

        metrics_processor.increment_total_requests()
        async with self.model_inst(session_id) as inst:
            token_ids = input_ids.copy()
            history_len = self.id2step[session_id]
            input_len = len(input_ids)
            output_len, gen_len = 0, 0
            state = DetokenizeState(len(input_ids))
            start_ids_offset = state.ids_offset
            response = ''
            finish_reason = None
            async with self.safe_run(
                    inst,
                    session_id=session_id,
                    **prompt_input,
                    gen_config=gen_config,
                    adapter_name=adapter_name,
                    stream_output=stream_response,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=history_len,
            ) as gen:
                prev_len = 0
                hit_stop_token = 0
                req_state = RequestState(prompt_len=input_len)  # per-requst state
                async for outputs in gen:
                    iteration_stats = IterationStats()  # per-iteration stats
                    # decode res
                    if is_error(outputs.status):
                        break

                    output_len = outputs.num_token
                    metrics_processor.queue_update((input_len, prev_len, outputs, req_state, iteration_stats))

                    if hit_stop_token or prev_len == output_len:
                        continue

                    # This assumes the engine will stop when stop token is hit
                    if output_len and outputs.token_ids[-1] in stop_ids:
                        hit_stop_token = 1
                        # one token and it's been skipped
                        if output_len == prev_len + 1:
                            continue

                    mask = slice(prev_len - output_len, output_len - hit_stop_token)

                    token_ids += outputs.token_ids[mask]
                    gen_len = len(token_ids) - input_len

                    prev_len = output_len

                    ids_offset = state.ids_offset
                    response, state = self.tokenizer.detokenize_incrementally(
                        token_ids,
                        state,
                        skip_special_tokens=gen_config.skip_special_tokens,
                        spaces_between_special_tokens=gen_config.spaces_between_special_tokens,
                    )
                    res = token_ids[ids_offset:]

                    out = GenOut(
                        response,
                        history_len,
                        input_len,
                        gen_len,
                        finish_reason,
                        token_ids=res,
                        cache_block_ids=outputs.cache_block_ids,
                    )

                    if outputs.logprobs is not None:
                        log_offset = ids_offset - start_ids_offset
                        out.logprobs = outputs.logprobs[log_offset:]
                    if outputs.last_hidden_state is not None:
                        out.last_hidden_state = outputs.last_hidden_state
                        if hit_stop_token:
                            out.last_hidden_state = out.last_hidden_state[:-hit_stop_token]
                    if outputs.logits is not None:
                        out.logits = outputs.logits
                        if hit_stop_token:
                            out.logits = out.logits[:-hit_stop_token]

                    yield out
                # end of generator loop
                metrics_processor.increment_finished_requests()

                if not is_error(outputs.status):
                    finish_reason = ('length' if gen_len >= gen_config.max_new_tokens else 'stop')
                    # utf-8 char at the end means it's a potential unfinished
                    # byte sequence
                    if not response.endswith('�'):
                        # avoid returning the last response twice
                        response = ''
                    logger.info(f"session {session_id} finished, reason "
                                f'"{finish_reason}", input_tokens '
                                f'{len(input_ids)}, output_tokens {gen_len}')
                    yield GenOut(response,
                                 self.id2step[session_id],
                                 len(input_ids),
                                 gen_len,
                                 finish_reason,
                                 token_ids=[],
                                 cache_block_ids=outputs.cache_block_ids)
                else:
                    logger.error(f"session {session_id} finished, "
                                 'reason "error"')
                    yield GenOut(
                        response='internal error happened',
                        history_token_len=self.id2step[session_id],
                        input_token_len=len(input_ids),
                        generate_token_len=0,
                        finish_reason='error',
                        token_ids=[],
                    )
            # update step
            if sequence_end:
                self.id2step[session_id] = 0
                if self.backend == 'pytorch':
                    # manually end pytorch session
                    await inst.async_end(session_id)
            else:
                if rewind_stop_tokens:
                    # rewind the step to the token before the stop token
                    output_len = gen_len
                self.id2step[session_id] += input_len + output_len

    def _run(self, fn=None, coro=None, loop=None):
        assert (fn or coro) and not (fn and coro)
        loop = loop or self.internal_thread.loop
        if fn:

            async def _coro():
                return fn()

            coro = _coro()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def session(self, gen_config: GenerationConfig = None):
        return Session(
            self._run(fn=lambda: next(self._session_id)).result(),
            engine=self,
            gen_config=gen_config,
        )

    def chat(
        self,
        prompt: str,
        session=None,
        gen_config: Optional[GenerationConfig] = None,
        stream_response=False,
        **kwargs,
    ) -> Union[Session, Iterator]:
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

        generator = self.infer(
            prompt,
            gen_config,
            sequence_start=sequence_start,
            sequence_end=False,
            session_id=session._id,
            stream_response=stream_response,
            multiplex=True,
        )

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
            logger.warning(f"Invalid Free session {session_id}.")

    def p2p_initialize(self, init_request: DistServeInitRequest):
        return self.engine.p2p_initialize(init_request)

    def p2p_connect(self, conn_request: List[DistServeConnectionRequest]):
        return self.engine.p2p_connect(conn_request)

    """ DistServe Async Engine API End """
