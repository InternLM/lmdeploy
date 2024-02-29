# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import queue
from threading import Thread
from typing import List, Union

from PIL.Image import Image

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.builder import load_vl_model

logger = get_logger('lmdeploy')


class Record:

    def __init__(self):
        self.number = []
        self.waiting = []
        self.done = []
        self.res_que = []
        self.total = 0

    def enqueue(self, images: List[Image], que: Union[queue.Queue,
                                                      asyncio.Queue]):
        self.number.append(len(images))
        self.waiting.extend(images)
        self.res_que.append(que)
        self.total += len(images)

    def dequeue(self, max_batch_size):
        inputs = self.waiting[:max_batch_size]
        self.waiting = self.waiting[max_batch_size:]
        self.total -= len(inputs)
        return inputs

    def nofify(self):
        if len(self.number) == 0 or self.number[0] < len(self.done):
            return False
        num_images = self.number.pop(0)
        outputs = self.done[:num_images]
        self.done = self.done[num_images:]
        que = self.res_que.pop(0)
        if isinstance(que, queue.Queue):
            que.put(outputs)
        else:
            que._loop.call_soon_threadsafe(que.put_nowait, outputs)
        return True


class ImageEncoder:
    """Image encoder."""

    def __init__(self, model_path: str, max_batch_size: int = 16):
        self.model = load_vl_model(model_path)
        self.max_batch_size = max_batch_size
        self.loop = asyncio.new_event_loop()
        self.work_thread = self._start_work_thread()

    def _start_work_thread(self):

        def _work_thread():
            asyncio.set_event_loop(self.loop)
            self.que = asyncio.Queue()
            self.loop.run_until_complete(self._forward_loop())

        thread = Thread(target=_work_thread, daemon=True)
        thread.start()
        return thread

    async def _forward_loop(self):
        logger.info('start ImageEncoder._forward_loop')
        record = Record()
        while True:
            while record.total == 0 or (self.que.qsize() and
                                        record.total < self.max_batch_size):
                item = await self.que.get()
                record.enqueue(item[0], item[1])
            inputs = record.dequeue(self.max_batch_size)
            outputs = self.model.forward(inputs)
            record.done.extend(outputs)
            while record.nofify():
                pass

    def forward(self, inputs: List[Image]):
        """Model forward."""
        return self.model.forward(inputs)

    def infer(self, inputs: List[Image]):
        """infer."""
        outputs = queue.Queue()
        item = (inputs, outputs)
        self.loop.call_soon_threadsafe(self.que.put_nowait, item)
        results = outputs.get()
        return results

    async def async_infer(self, inputs: List[Image]):
        """async infer."""
        outputs = asyncio.Queue()
        item = (inputs, outputs)
        self.loop.call_soon_threadsafe(self.que.put_nowait, item)
        results = await outputs.get()
        return results
