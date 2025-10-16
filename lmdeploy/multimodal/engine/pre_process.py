# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List

from lmdeploy.utils import get_logger
from lmdeploy.vl.utils import load_image

logger = get_logger('lmdeploy')


class PreProcessor():

    def __init__(self, model_agent, post_processor):
        print('=> PreProcessor init')
        self._in_que = asyncio.Queue()
        self.model_agent = model_agent
        self.post_processor = post_processor

        self._loop_task = None

    @staticmethod
    def collect_images(messages):
        """Gather all images along with their respective parameters from the
        messages and compile them into a single list. Each image is converted
        to RGB color space.

        Args:
            messages (List[Tuple[Image, Dict]]): a list of images with their
                corresponding parameters
        """  # noqa
        images = []
        for message in messages:
            content = message['content']
            if not isinstance(content, List):
                continue
            images.extend([(x['image'], {
                k: v
                for k, v in x.items() if k not in {'type', 'image'}
            }) for x in content if x['type'] == 'image'])
        return images

    @classmethod
    async def async_convert_to_pil_images(cls, messages: List[Dict]) -> List[Dict]:
        """Scan the provided messages to find image URLs or base64-encoded
        image data. Loads the images into Pillow image objects.

        Args:
            messages (List[Dict]): a user request of GPT4V message format
        """
        if isinstance(messages, Dict):
            messages = [messages]
        assert isinstance(messages, List)

        out_messages = [None] * len(messages)

        def _inner_call(i, in_messages, out_messages):
            role = in_messages[i]['role']
            content = in_messages[i]['content']
            assert role in ['system', 'user', 'assistant'], \
                f'unsupported role "{role}"'
            if role != 'user' or isinstance(content, str):
                # the content is a user's prompt or an assistant's prompt,
                # returning it directly
                out_messages[i] = in_messages[i]
                return
            # the role is a user and the content is a list, in which there
            # might be image_url or image_data
            assert isinstance(content, List)
            message = dict(role=role, content=[])
            for item in content:
                # image url or base64-encoded image data
                if item['type'] == 'image_url':
                    """
                    convert the following item:
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': 'image url or base64-encoded image data',
                            'key': 'value'  # parameters used in image processing
                            ...
                        }
                    }
                    to:
                    {
                        'type': 'image',
                        'image': Pillow.Image,
                        'key': 'value'   # parameters used in image processing
                        ...
                    }
                    """  # noqa
                    data = item['image_url'].copy()
                    try:
                        url = data.pop('url')
                        image = load_image(url)
                        data.update(type='image', image=image)
                        message['content'].append(data)
                    except KeyError:
                        logger.error(f'invalid format {message}')
                elif item['type'] == 'image_data':
                    """
                    convert the following item:
                    {
                        'type': 'image_data',
                        'image_data': {
                            'data': Pillow.Image,
                            'key': 'value'  # parameters used in image processing
                            ...
                        }
                    }
                    to:
                    {
                        'type': 'image',
                        'image': Pillow.Image,
                        'key': 'value'   # parameters used in image processing
                        ...
                    }
                    """  # noqa
                    data = item['image_data'].copy()
                    try:
                        image = data.pop('data')
                        data.update(type='image', image=image)
                        message['content'].append(data)
                    except KeyError:
                        logger.error(f'invalid format {message}')
                elif item['type'] == 'text':
                    message['content'].append(item)
                else:
                    logger.error(f'unexpected content type {message}')
            out_messages[i] = message

        await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(None, _inner_call, i, messages, out_messages)
            for i in range(len(messages))
        ])
        return out_messages

    def start_loop(self):
        """Creates a task for the given coroutine."""
        if not hasattr(self, '_loop_task') or self._loop_task is None:
            logger.info('Starting PreProcessor loop')
            self._loop_task = asyncio.create_task(self.async_loop())

    async def async_loop(self):
        while True:
            session_id, messages = await self._in_que.get()

            messages = await self.async_convert_to_pil_images(messages)
            print(f'after convert msg: {messages}')

            proc_inputs = self.model_agent.model.preprocess(messages)
            print(f'after preproc msg: {proc_inputs}')

            # TODO: process to get token ids, image mask

            self.model_agent._pre_in_que.put_nowait((session_id, proc_inputs))

    def process(self, session_id, messages, future):
        if messages is not None:
            self._in_que.put_nowait((session_id, messages))

            self.post_processor.add_future(session_id, messages, future)

    def close(self):
        """Cancel the background loop task."""
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            logger.info('PreProcessor loop cancelled.')
