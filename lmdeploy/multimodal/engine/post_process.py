# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class PostProcessor():

    def __init__(self, model_agent):
        print('=> PostProcessor init')
        self.model_agent = model_agent
        self._loop_task = None

        # session_id -> future
        self._future_store: Dict[int, asyncio.Future] = {}

    def add_future(self, session_id, messages, future):
        self._future_store[session_id] = (messages, future)

    def start_loop(self):
        if not hasattr(self, '_loop_task') or self._loop_task is None:
            logger.info('Starting PostProcessor loop')
            self._loop_task = asyncio.create_task(self.async_loop())

    def post_process(self, messages: List[Dict]):
        # TODO: implement model-specific post process logic
        return messages

    async def async_loop(self):
        while True:
            out = await self.model_agent._post_proc_que.get()
            print(f'=> PostProcessor got data: {out}')

            out = self.post_process(out)
            print(f'=> PostProcessor post-processed data: {out}')

            session_id = out.pop('session_id', None)
            print(f'=> PostProcessor session_id: {session_id}')
            messages, future = self._future_store.pop(session_id, None)
            print(messages)
            # add out as an attri named 'encoder_result' to messages

            messages[0]['encoder_result'] = out
            if future and not future.done():
                print(f'=> PostProcessor setting future result: {messages}')
                future.set_result(messages)

    def close(self):
        """Cancel the background loop task."""
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            logger.info('PostProcessor loop cancelled.')
