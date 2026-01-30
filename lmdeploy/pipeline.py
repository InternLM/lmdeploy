# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import atexit
import concurrent.futures
import os
from contextlib import closing
from functools import partial
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Dict, Iterator, List, Tuple

import torch
import tqdm
from typing_extensions import deprecated

from .archs import autoget_backend_config, get_task
from .messages import GenerationConfig, PytorchEngineConfig, Response, SpeculativeConfig, TurbomindEngineConfig
from .model import ChatTemplateConfig
from .serve.processors import MultimodalProcessor
from .utils import get_logger, get_model

if TYPE_CHECKING:
    from PIL.Image import Image

    from .serve.managers import Session

logger = get_logger('lmdeploy')


class Pipeline:
    """Pipeline - User-facing API layer for inference."""

    def __init__(self,
                 model_path: str,
                 backend_config: TurbomindEngineConfig | PytorchEngineConfig | None = None,
                 chat_template_config: ChatTemplateConfig | None = None,
                 log_level: str = 'WARNING',
                 max_log_len: int | None = None,
                 speculative_config: SpeculativeConfig | None = None,
                 **kwargs):
        """Initialize Pipeline.

        Args:
            model_path: Path to the model.
            backend_config: Backend configuration.
            chat_template_config: Chat template configuration.
            log_level: Log level.
            max_log_len: Max number of prompt characters or prompt tokens being printed in log.
            speculative_config: Speculative decoding configuration.
            **kwargs: Additional keyword arguments.
        """

        os.environ.setdefault('TM_LOG_LEVEL', log_level)
        logger.setLevel(log_level)

        # Download model if the path does not exist locally
        if not os.path.exists(model_path):
            download_dir = backend_config.download_dir if backend_config else None
            revision = backend_config.revision if backend_config else None
            model_path = get_model(model_path, download_dir, revision)

        # Download speculative model if the path does not exist locally
        if speculative_config and speculative_config.model and not os.path.exists(speculative_config.model):
            download_dir = backend_config.download_dir if backend_config else None
            speculative_config.model = get_model(speculative_config.model, download_dir)

        # Create inference engine
        _, pipeline_class = get_task(model_path)
        backend, backend_config = autoget_backend_config(model_path, backend_config)
        self.async_engine = pipeline_class(model_path,
                                           backend=backend,
                                           backend_config=backend_config,
                                           chat_template_config=chat_template_config,
                                           max_log_len=max_log_len,
                                           speculative_config=speculative_config,
                                           **kwargs)
        self.internal_thread = _EventLoopThread(daemon=True)
        self.limiter: asyncio.Semaphore = None
        self.session_mgr = self.async_engine.session_mgr
        self.backend_config = self.async_engine.backend_config
        self.async_engine.start_loop(self.internal_thread.loop, use_async_api=False)

    def infer(self,
              prompts: List[str] | str | List[Dict] | List[List[Dict]] | Tuple | List[Tuple],
              gen_config: GenerationConfig | List[GenerationConfig] | None = None,
              do_preprocess: bool = True,
              adapter_name: str | None = None,
              use_tqdm: bool = False,
              **kwargs):
        """Inference prompts.

        Args:
            prompts: Prompts to inference. It can be a single prompt, a list of prompts, a list of tuples, or a tuple.
                Tuple can be (prompt, image or [images]) or (image or [images], prompt).
            gen_config(GenerationConfig | List[GenerationConfig] | None): Generation configuration(s).
            do_preprocess(bool): Whether to pre-process messages.
            adapter_name(str | None): Adapter name.
            use_tqdm(bool): Whether to use progress bar.
            **kwargs(dict): Additional keyword arguments.
        """
        is_single = self._is_single(prompts)
        # format prompts to openai message format, which is a list of dicts
        prompts = MultimodalProcessor.format_prompts(prompts)
        pbar = tqdm.tqdm(total=len(prompts)) if use_tqdm else None
        outputs = []
        try:
            requests = self._request_generator(prompts,
                                               gen_config=gen_config,
                                               do_preprocess=do_preprocess,
                                               adapter_name=adapter_name,
                                               stream_response=False,
                                               **kwargs)
            for g in self._infer(requests, multiplex=False, pbar=pbar):
                res = None
                for out in g:
                    res = res.extend(out) if res else out
                outputs.append(res)
        finally:
            if pbar: pbar.close()  # noqa
        if is_single:
            return outputs[0]
        return outputs

    @deprecated('This method is deprecated. Please use "Pipeline.infer" instead.')
    def batch_infer(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def stream_infer(self,
                     prompts: List[str] | str | List[Dict] | List[List[Dict]] | Tuple | List[Tuple],
                     sessions: 'Session' | List['Session'] | None = None,
                     gen_config: GenerationConfig | List[GenerationConfig] | None = None,
                     do_preprocess: bool = True,
                     adapter_name: str | None = None,
                     stream_response: bool = True,
                     **kwargs):
        """Stream inference.

        Args:
            prompts(List[str] | str | List[Dict] | List[List[Dict]] | Tuple | List[Tuple]): Prompts to inference.
                It can be a single prompt, a list of prompts, a list of tuples, or a tuple.
                Tuple can be (prompt, image or [images]) or (image or [images], prompt).
            sessions(Session | List[Session] | None): Sessions. Each of which corresponds to a prompt.
            gen_config(GenerationConfig | List[GenerationConfig] | None): Generation configuration(s).
            do_preprocess(bool): Whether to pre-process messages.
            adapter_name(str | None): Adapter name.
            stream_response(bool): Whether to stream the response. If True, the generator will stream the response.
                Otherwise, the generator will run until finish and return the final response. This argument
                is introduced to support the streaming and non-streaming modes of Pipeline.chat.
            **kwargs(dict): Additional keyword arguments.

        Returns:
            Generator: A generator that yields the output (i.e. instance of class `Response`) of the inference.
        """
        prompts = MultimodalProcessor.format_prompts(prompts)
        requests = self._request_generator(prompts,
                                           sessions=sessions,
                                           gen_config=gen_config,
                                           do_preprocess=do_preprocess,
                                           adapter_name=adapter_name,
                                           stream_response=stream_response,
                                           **kwargs)
        return self._infer(requests, multiplex=True)

    def close(self):
        """Close the pipeline."""
        self.internal_thread.close()
        self.async_engine.close()

    def chat(self,
             prompt: str | Tuple[str, 'Image' | List['Image']],
             session=None,
             gen_config: GenerationConfig | None = None,
             stream_response=False,
             adapter_name=None,
             **kwargs) -> 'Session' | Iterator:
        """Chat.

        Args:
            prompt (str): prompt
            session (Session): the chat session
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            stream_response (bool): whether to stream the response.
            adapter_name (str): adapter name.
            **kwargs (dict): additional keyword arguments.
        """
        if session is None:
            session = self.session_mgr.get()
        session.update(prompt=prompt, response=None)

        prompt = MultimodalProcessor.format_prompts(prompt)

        sequence_start = session.step == 0
        generator = self.stream_infer(prompts=prompt,
                                      sessions=session,
                                      gen_config=gen_config,
                                      stream_response=stream_response,
                                      adapter_name=adapter_name,
                                      multiplex=True,
                                      sequence_start=sequence_start,
                                      sequence_end=False,
                                      step=session.step)

        def _gen():
            resp = None
            try:
                for out in generator:
                    resp = resp.extend(out) if resp else out
                    yield out
            except:  # noqa
                self._run(coro=session.async_abort())
                raise
            else:
                session.response = resp
                session.step += resp.generate_token_len + resp.input_token_len
                session.history.append((session.prompt, resp.text))

        if stream_response:
            return _gen()
        else:
            # run the generator until finish
            with closing(_gen()) as gen:
                for _ in gen:
                    pass
            session.generator = None

        return session

    def session(self) -> 'Session':
        """Create a new session."""
        return self.session_mgr.get()

    def get_reward_score(self, input_ids: List) -> List[float]:
        """Get reward score.

        Args:
            input_ids(List): a list of token_id or a list of token_id list or token_id tensor
        Return:
            reward score in a list. If the input_ids is a list of token_id, the return value
            is still a list with length 1.
        """
        supported_reward_models = ['InternLM2ForRewardModel', 'Qwen2ForRewardModel']
        arch = self.async_engine.arch
        if arch not in supported_reward_models:
            raise ValueError(f'{arch} is not in reward model list: {supported_reward_models}')
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(isinstance(x, List) for x in input_ids)
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids
        logits = self._run(coro=self.async_engine.async_get_logits(input_ids=input_ids)).result()
        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    def get_ppl(self, input_ids: List[int] | List[List[int]]) -> List[float]:
        """Get perplexity scores given a list of input tokens that have to be
        of the same length.

        Args:
            input_ids (List[int] | List[List[int]]): the batch of input token ids

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert isinstance(input_ids, List)
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        assert all(len(_) > 1 for _ in input_ids)

        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.async_engine.hf_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (vocab_size * 4)
        sizes = [len(_) for _ in input_ids]
        result = []
        sorted_index_values = sorted(list(enumerate(sizes)), key=lambda x: x[1], reverse=True)
        sizes = [value for index, value in sorted_index_values]
        indices = [index for index, value in sorted_index_values]
        logger.info(f'sorted sizes: {sizes}')
        logger.info(f'sorted indices: {indices}')
        for (start, end) in self._batch_iterator(sizes, max_input_len):
            logger.info(f'start: {start}, end: {end}')
            if start == end:
                _input_ids = input_ids[indices[start]]
                res = self._get_long_text_ppl(input_ids=_input_ids, max_input_len=max_input_len)
                result.append(res)
            else:
                _input_ids = [input_ids[indices[i]] for i in range(start, end)]
                res = self._get_ppl(
                    input_ids=_input_ids,
                    max_input_len=max_input_len,
                )
                result.extend(res)
        output = list(range(len(result)))
        for index, sorted_index in enumerate(indices):
            output[sorted_index] = result[index]
        return output

    def __call__(self,
                 prompts: List[str] | str | List[Dict] | List[List[Dict]],
                 gen_config: GenerationConfig | List[GenerationConfig] | None = None,
                 **kwargs):
        return self.infer(prompts, gen_config=gen_config, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @deprecated('This method is deprecated. Please use "AsyncEngine.generate" instead.')
    async def generate(self, *args, **kwargs):
        """Generate responses as an async generator.

        This method delegates to async_engine.generate and forwards all yielded values.
        """
        async for item in self.async_engine.generate(*args, **kwargs):
            yield item

    @staticmethod
    def _is_single(prompts):
        """Check if prompts is a single prompt."""
        return (isinstance(prompts, str) or (isinstance(prompts, tuple) and len(prompts) == 2)
                or (isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], Dict)))

    def _request_generator(self,
                           prompts: List[str] | str | List[Dict] | List[List[Dict]],
                           sessions: List['Session'] | 'Session' | None = None,
                           gen_config: GenerationConfig | List[GenerationConfig] | None = None,
                           **kwargs):
        """Generate requests."""
        is_single = self._is_single(prompts)
        prompts = [prompts] if is_single else prompts

        if sessions is None:
            sessions = [self.session_mgr.get() for _ in prompts]
        elif isinstance(sessions, list):
            sessions = sessions
        else:
            sessions = [sessions]

        if len(prompts) != len(sessions):
            raise ValueError(f'prompts and sessions should have the same length. '
                             f'Got {len(prompts)} prompts and {len(sessions)} sessions')

        if gen_config is None:
            gen_configs = [GenerationConfig()] * len(prompts)
        elif isinstance(gen_config, list):
            gen_configs = gen_config
        else:
            gen_configs = [gen_config] * len(prompts)

        if len(prompts) != len(gen_configs):
            raise ValueError(f'input gen_config length differs from the length of prompts. '
                             f'Got {len(prompts)} prompts and {len(gen_configs)} gen_configs')

        for prompt, gen_cfg, session in zip(prompts, gen_configs, sessions):
            # Use session_id is for backward compatibility. We will remove it in the future.
            # Since AsyncEngine.generate defines session_id in the argument lists, here we
            # use session_id to pass the session to the AsyncEngine.generate. It's
            yield dict(session_id=session, messages=prompt, gen_config=gen_cfg, **kwargs)

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
                gen = self.async_engine.generate(**req)
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

    def _run(self, fn=None, coro=None):
        assert (fn or coro) and not (fn and coro)
        loop = self.internal_thread.loop
        if fn:

            async def _coro():
                return fn()

            coro = _coro()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _batch_iterator(self, sizes, max_value):
        """Return an iterator that calculates intervals (start, end) of a
        descend-order list, in which the sum of values in the range is the
        maximum number not less than max_value. By "the sum of values",

        here it means $$len(sizes[start:end]) * sizes[start]$$
        """
        i = 0
        while i < len(sizes):
            current_sum = 0
            start_index = i

            while i < len(sizes) and current_sum + sizes[start_index] <= max_value:
                current_sum += sizes[start_index]
                i += 1

            yield (start_index, i)
            if i > start_index:
                continue
            else:
                i += 1

    def _get_long_text_ppl(self, input_ids, max_input_len):
        assert all(isinstance(_, int) for _ in input_ids)
        seq_len = len(input_ids)
        assert seq_len > max_input_len
        logger.info(f'get long text ppl: seq_len {seq_len}')

        losses = []
        target_counts = []
        for i in range(0, seq_len, max_input_len):
            token_ids = input_ids[i:i + max_input_len]
            step = [i]
            # shift token_ids by 1 to the left
            target_ids = input_ids[i + 1:i + 1 + max_input_len]
            loss = self._get_ppl(input_ids=[token_ids],
                                 max_input_len=len(token_ids),
                                 target_ids=[target_ids],
                                 steps=step,
                                 sequence_start=(i == 0),
                                 sequence_end=False)
            losses.extend(loss)
            target_counts.append(len(target_ids))
        losses = [loss * target_count for loss, target_count in zip(losses, target_counts)]
        loss_sum = sum(losses)
        target_count = sum(target_counts)
        return loss_sum / target_count

    def _get_ppl(self,
                 input_ids,
                 max_input_len,
                 target_ids=None,
                 steps=None,
                 sequence_start: bool = True,
                 sequence_end: bool = True):
        assert (isinstance(input_ids, List) and all(isinstance(_, List) for _ in input_ids))
        assert steps is None or len(steps) == len(input_ids)
        assert target_ids is None or len(target_ids) == len(input_ids)

        lens = [len(_) for _ in input_ids]
        total_len = sum(lens)
        assert sum(lens) <= max_input_len

        logger.info(f'get_ppl: bs: {len(input_ids)}, lens: {lens}, '
                    f'total_len: {total_len}, steps: {steps}')
        torch.cuda.empty_cache()

        logits = self._run(coro=self.async_engine.async_get_logits(
            input_ids=input_ids, steps=steps, sequence_start=sequence_start, sequence_end=sequence_end)).result()
        padding_token_id = -100
        if target_ids is None:
            target_ids = [x[1:] + [padding_token_id] for x in input_ids]
        else:
            target_ids = [
                target_ids[i] + [padding_token_id] if len(target_ids[i]) < len(input_ids[i]) else target_ids[i]
                for i in range(len(input_ids))
            ]
        target_ids = [torch.Tensor(torch.LongTensor(_target_ids)) for _target_ids in target_ids]

        result = []
        for _logits, _target_ids in zip(logits, target_ids):
            _logits = _logits.float()
            vocab_size = _logits.shape[-1]
            _target_ids = _target_ids.to(_logits.device)
            target_mask = _target_ids != padding_token_id
            # compute cross entropy loss
            flat_logits = _logits.contiguous().view(-1, vocab_size)
            flat_target_ids = _target_ids.contiguous().view(-1)
            flat_loss_matrix = torch.nn.functional.cross_entropy(flat_logits,
                                                                 flat_target_ids,
                                                                 reduction='none',
                                                                 ignore_index=padding_token_id)
            loss = flat_loss_matrix.sum()
            target_count = target_mask.sum()
            result.append(loss.item() / target_count.item())
        logger.info(f'ppl result: {result}')
        return result


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
