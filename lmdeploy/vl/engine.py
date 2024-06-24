# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import queue
import time
from typing import List, Optional, Union

import torch
from PIL.Image import Image

from lmdeploy.messages import (PytorchEngineConfig, TurbomindEngineConfig,
                               VisionConfig)
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.builder import load_vl_model

logger = get_logger('lmdeploy')


class Record:
    """Batching manager."""

    def __init__(self):
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
        que.put_nowait(outputs)
        self.log('done', num_images)
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
        self.max_batch_size = (1 if vision_config is None else
                               vision_config.max_batch_size)
        torch.cuda.empty_cache()
        self._que: asyncio.Queue = None
        self._loop_task: asyncio.Task = None

    @property
    def req_que(self):
        if self._que is None:
            self._que = asyncio.Queue()
        if self._loop_task is None:
            task = asyncio.get_event_loop().create_task(self._forward_loop())
            self._loop_task = task
        assert asyncio.get_event_loop() == self._loop_task.get_loop()
        return self._que

    async def _forward_loop(self):
        """working loop to process images."""
        logger.info('start ImageEncoder._forward_loop')
        record = Record()
        while True:
            while record.total == 0 or (self._que.qsize() and
                                        record.total < self.max_batch_size):
                while self._que.qsize() == 0:
                    await asyncio.sleep(0)
                item = await self._que.get()
                record.enqueue(item[0], item[1])
            inputs = record.dequeue(self.max_batch_size)
            outputs = await asyncio.get_event_loop().run_in_executor(
                None, self.forward, inputs)
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
        self.req_que.put_nowait(item)
        results = await outputs.get()
        return results
