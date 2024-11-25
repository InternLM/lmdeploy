# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import queue
import time
from threading import Thread
from typing import Dict, List, Optional, Union

import torch
from PIL.Image import Image

from lmdeploy.messages import (PytorchEngineConfig, TurbomindEngineConfig,
                               VisionConfig)
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.builder import load_vl_model

logger = get_logger('lmdeploy')


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    """raise exception on finish."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        raise e


class Record:
    """Batching manager."""

    def __init__(self, thread_safe):
        self.thread_safe = thread_safe
        self.number = []
        self.waiting = []
        self.kwargs = []
        self.done = []
        self.res_que = []
        self.total = 0

    def enqueue(self, images: List[Image], kwargs: List[Dict],
                que: Union[queue.Queue, asyncio.Queue]):
        """add ith request to manager."""
        self.number.append(len(images))
        self.waiting.extend(images)
        self.kwargs.extend(kwargs)
        self.res_que.append(que)
        self.total += len(images)
        self.log('received', len(images))

    def dequeue(self, max_batch_size):
        """try to dequeue max batch size images."""
        inputs = self.waiting[:max_batch_size]
        kwargs = self.kwargs[:max_batch_size]
        self.waiting = self.waiting[max_batch_size:]
        self.kwargs = self.kwargs[max_batch_size:]
        self.total -= len(inputs)
        self.log('process', len(inputs))
        return inputs, kwargs

    def notify(self):
        """set result if request i is finished."""
        if len(self.number) == 0 or self.number[0] > len(self.done):
            return False
        num_images = self.number.pop(0)
        outputs = self.done[:num_images]
        self.done = self.done[num_images:]
        que = self.res_que.pop(0)
        self.log('done', num_images)
        if self.thread_safe:
            que._loop.call_soon_threadsafe(que.put_nowait, outputs)
        else:
            que.put_nowait(outputs)
        return True

    def log(self, task: str, num: int):
        logger.info(f'ImageEncoder {task} {num} images, '
                    f'left {self.total} images.')


class ImageEncoder:
    """Image encoder."""

    def __init__(
        self,
        model_path: str,
        backend: str,
        vision_config: VisionConfig = None,
        backend_config: Optional[Union[TurbomindEngineConfig,
                                       PytorchEngineConfig]] = None,
    ):
        self.model = load_vl_model(model_path,
                                   backend,
                                   backend_config=backend_config)
        if vision_config is None:
            vision_config = VisionConfig()
        self.vision_config = vision_config
        self.max_batch_size = vision_config.max_batch_size
        torch.cuda.empty_cache()
        self._que = asyncio.Queue()
        if vision_config.thread_safe:
            self._create_thread_safe_task()

    def _create_thread_safe_task(self):
        """thread safe loop task."""
        self._loop = asyncio.new_event_loop()

        def _work_thread():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._forward_loop())

        thread = Thread(target=_work_thread, daemon=True)
        thread.start()

    def _create_event_loop_task(self):
        """event loop task."""
        task = asyncio.get_event_loop().create_task(self._forward_loop())
        self._loop_task = task
        self._loop = task.get_loop()

    @property
    def req_que(self):
        if self.vision_config.thread_safe:
            return self._que
        if self._loop_task is None:
            self._create_event_loop_task()
        if asyncio.get_event_loop() != self._loop:
            raise RuntimeError('Current event loop is different from'
                               ' the one bound to loop task!')
        return self._que

    async def _forward_loop(self):
        """working loop to process images."""
        logger.info('start ImageEncoder._forward_loop')
        record = Record(self.vision_config.thread_safe)
        while True:
            while record.total == 0 or (self._que.qsize() and
                                        record.total < self.max_batch_size):
                while self._que.qsize() == 0:
                    await asyncio.sleep(0.01)
                item = await self._que.get()
                record.enqueue(item[0], item[1], item[2])
            inputs, kwargs = record.dequeue(self.max_batch_size)
            future = asyncio.get_event_loop().run_in_executor(
                None, self.forward, inputs, kwargs)
            future.add_done_callback(_raise_exception_on_finish)
            outputs = await future
            record.done.extend(outputs)
            while record.notify():
                pass

    def forward(self, messages: List[List[Dict]]) -> Dict:
        # messages in batch
        assert all(isinstance(message, List) for message in messages)

        time_start = time.perf_counter()
        outputs, n_image = self.model.forward(messages)
        if isinstance(outputs[0], torch.Tensor):
            outputs = [x.cpu() for x in outputs]
        time_end = time.perf_counter()
        logger.info(f'ImageEncoder forward {n_image} images, '
                    f'cost {time_end - time_start:.3f}s')
        return outputs

    def infer(self, messages: List[Dict]) -> Dict:
        """perform vision encoding to get a dict, in which there are input_ids,
        embeddings, embedding_ranges and so on. They will be used by turbomind
        engine. The key in the dict must be the same defined in turbmoind
        engine's infer API.

        Args:
            messages (List[Dict]): user's input in GPT4V format
        """
        assert isinstance(messages, List)
        assert all(isinstance(item, Dict) for item in messages)

        return self.forward(messages)

    async def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """preprocess multimodal data in the messages.

        This function is designed to gather all images along with their respective parameters from the
        messages and compile them into a single list. It then appends an entry to the messages list,
        formatted as a dictionary `dict(role='images', conrtent=List[Dict])`.
        Additionally, it invokes the preprocess method of the vision model to handle image preprocessing.
        The subsequent results are integrated into the messages list, with the specific approach left to
        the vision model's discretion. For instance, the results may be appended directly to the messages
        list or merged into each individual image item.
        Args:
            messages (List[Dict]): a list of message. For instance,
            [
                {'role': 'user', 'content': 'string'},
                {'role': 'assistant', 'content': 'string'},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'string'},
                        {
                            'type': 'image',
                            'image': pillow.Image,
                            'key1': 'value1',
                            ...
                        },
                        {...},
                    ]
                },
                {...}
            ]
        Returns:
            [
                {'role': 'user', 'content': 'string'},
                {'role': 'assistant', 'content': 'string'},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'string'},
                        {
                            'type': 'image',
                            'image': pillow.Image,
                            'key1': 'value1',
                            ...
                        },
                        {...},
                    ],
                },
                {...},
                {'role': 'images', 'images': List[Tuple]}
            ]
        """  # noqa
        # collect all images in the `messages` to a list and insert it
        # to `messages` as a new entry
        images = []
        for i, message in enumerate(messages):
            content = message['content']
            if not isinstance(content, List):
                continue
            images.extend([
                (x['image'],
                 {k: v
                  for k, v in x.items() if k not in {'type', 'image'}})
                for x in content if x['type'] == 'image'
            ])
        messages.append(dict(role='images', content=images))
        result = self.model.preprocess(messages)
        return result

    async def async_infer(self, messages: List[Dict]) -> List[Dict]:
        """get multimodal embedding.

        Args:
            messages (List[Dict]): a list of message, which is the output
            of `preprocess()`
        """
        return self.model.forward(messages)

    async def wrap_for_pytorch(self, messages: List[Dict], chat_template,
                               tokenizer, sequence_start) -> List[Dict]:
        """
        Args:
            messages (List[Dict]): a list of message, which is supposed to be
                the output of `preprocess`
        Returns:
            a dict which will be passed to pytorch engine_instance's forward.
            The dict is like the following:
            Dict(
                'prompt': 'the prompt after applying chat template'
                'input_ids': [],
                'multimodal': {
                    'pixel_values': torch.Tensor,
                    ...
                ]
            )
        """
        result = self.model.to_pytorch(messages, chat_template, tokenizer,
                                       sequence_start)
        # clear data
        for i, message in enumerate(messages):
            if isinstance(message['content'], List):
                messages[i]['preprocess'] = None
        return result

    async def wrap_for_turbomind(self, messages: List[Dict], chat_template,
                                 tokenizer, sequence_start) -> Dict:
        """
        Args:
            messages (List[Dict]): a list of message, which is supposed to be
                the output of `async_infer`
        Returns:
            a dict which will be passed to pytorch engine_instance's forward.
            The dict is like the following:
            Dict(
                'prompt': 'the prompt after applying chat template'
                'input_ids': [],
                'input_embeddings': list[torch.Tensor],
                'input_embedding_ranges': list[torch.Tensor],
                ...
        """
        result = self.model.to_turbomind(messages, chat_template, tokenizer,
                                         sequence_start)
        # clear data
        for i, message in enumerate(messages):
            if isinstance(message['content'], List):
                messages[i]['preprocess'] = None
                messages[i]['forward'] = None
        return result
