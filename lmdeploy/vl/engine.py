# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import queue
import time
from threading import Thread
from typing import List, Optional, Union

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
        self.done = []
        self.res_que = []
        self.total = 0

    def enqueue(self, images: List[Image], que: Union[queue.Queue,
                                                      asyncio.Queue]):
        """add ith request to manager."""
        self.number.append(len(images))
        self.waiting.extend(images)
        self.res_que.append(que)
        self.total += len(images)
        self.log('received', len(images))

    def dequeue(self, max_batch_size):
        """try to dequeue max batch size images."""
        inputs = self.waiting[:max_batch_size]
        self.waiting = self.waiting[max_batch_size:]
        self.total -= len(inputs)
        self.log('process', len(inputs))
        return inputs

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

    def __init__(self,
                 model_path: str,
                 vision_config: VisionConfig = None,
                 backend_config: Optional[Union[TurbomindEngineConfig,
                                                PytorchEngineConfig]] = None):
        self.model = load_vl_model(model_path, backend_config=backend_config)
        if vision_config is None:
            vision_config = VisionConfig()
        self.vision_config = vision_config
        self.max_batch_size = vision_config.max_batch_size
        torch.cuda.empty_cache()
        self._que: asyncio.Queue = None
        self._loop_task: asyncio.Task = None
        if vision_config.thread_safe:
            self._create_thread_safe_task()

    def _create_thread_safe_task(self):
        """thread safe loop task."""
        self._loop = asyncio.new_event_loop()

        def _work_thread():
            asyncio.set_event_loop(self._loop)
            self._que = asyncio.Queue()
            self._loop.run_until_complete(self._forward_loop())

        thread = Thread(target=_work_thread, daemon=True)
        thread.start()
        self._loop_thread = thread

    def _create_event_loop_task(self):
        """event loop task."""
        task = asyncio.get_event_loop().create_task(self._forward_loop())
        self._loop_task = task
        self._loop = task.get_loop()

    @property
    def req_que(self):
        if self.vision_config.thread_safe:
            return self._que
        if self._que is None:
            self._que = asyncio.Queue()
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
                    await asyncio.sleep(0)
                item = await self._que.get()
                record.enqueue(item[0], item[1])
            inputs = record.dequeue(self.max_batch_size)
            future = asyncio.get_event_loop().run_in_executor(
                None, self.forward, inputs)
            future.add_done_callback(_raise_exception_on_finish)
            outputs = await future
            record.done.extend(outputs)
            while record.notify():
                pass

    def forward(self, inputs: List[Image]):
        """Model forward."""
        time_start = time.perf_counter()
        outputs = self.model.forward(inputs)
        if isinstance(outputs[0], torch.Tensor):
            outputs = [x.cpu() for x in outputs]
        time_end = time.perf_counter()
        logger.info(f'ImageEncoder forward {len(inputs)} images, '
                    f'cost {time_end - time_start:.3f}s')
        return outputs

    def infer(self, inputs: List[Image]):
        """infer."""
        results = self.forward(inputs)
        return results

    async def async_infer(self, inputs: List[Image]):
        """async infer."""
        outputs = asyncio.Queue()
        item = (inputs, outputs)
        if self.vision_config.thread_safe:
            self._loop.call_soon_threadsafe(self._que.put_nowait, item)
        else:
            self.req_que.put_nowait(item)
        results = await outputs.get()
        return results
