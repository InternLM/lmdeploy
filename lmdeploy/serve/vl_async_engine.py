# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Optional, Union

from lmdeploy.pytorch.check_env import try_import_deeplink
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.utils import get_logger
from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.vl.templates import VLPromptType, get_vl_prompt_template
from lmdeploy.vl.utils import load_image

logger = get_logger('lmdeploy')


class VLAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self, model_path: str, **kwargs) -> None:
        vision_config = kwargs.pop('vision_config', None)
        backend_config = kwargs.get('backend_config', None)
        self.backend = kwargs['backend']
        if self.backend == 'pytorch':
            try_import_deeplink(backend_config.device_type)
        self.vl_encoder = ImageEncoder(model_path,
                                       self.backend,
                                       vision_config,
                                       backend_config=backend_config)
        super().__init__(model_path, **kwargs)
        if self.model_name == 'base':
            raise RuntimeError(
                'please specify chat template as guided in https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#set-chat-template'  # noqa: E501
            )
        self.vl_prompt_template = get_vl_prompt_template(
            model_path, self.chat_template, self.model_name)

    def _convert_prompts(self,
                         prompts: Union[VLPromptType, List[Dict],
                                        List[VLPromptType], List[List[Dict]]]):
        """convert prompts to openai GPT4V format."""
        if isinstance(prompts, str) or isinstance(prompts, tuple):
            _prompts = self.vl_prompt_template.prompt_to_messages(prompts)
        elif isinstance(prompts[0], tuple) or isinstance(prompts[0], str):
            _prompts = [
                self.vl_prompt_template.prompt_to_messages(x) for x in prompts
            ]
        else:
            _prompts = prompts
        return _prompts

    async def _get_prompt_input(self,
                                messages: Union[str, List[Dict]],
                                do_preprocess: bool,
                                sequence_start: bool,
                                adapter_name: str,
                                tools: Optional[List[object]] = None,
                                **kwargs):
        """process messages and return the required data for the inference
        engines. Refer to pytorch.engine.EngineInstance.async_stream_infer and
        turbomind.TurboMindInstance.async_stream_infer for the argument
        specification.

        Args:
            messages
            do_preprocess
            sequence_start
            adapter_name
            tools
        Returns:
        """
        if isinstance(messages, str):
            return super(self)._get_prompt_input(messages, do_preprocess,
                                                 sequence_start, adapter_name,
                                                 tools, **kwargs)
        elif isinstance(messages, List):
            has_multimodal_input = any(
                isinstance(message['content'], list) and any(
                    item['type'] in ['image_url', 'image_data']
                    for item in message['content']) for message in messages)
            if not has_multimodal_input:
                return super(self)._get_prompt_input(messages, do_preprocess,
                                                     sequence_start,
                                                     adapter_name, tools,
                                                     **kwargs)
        else:
            raise RuntimeError(f'unsupported messages {messages}')

        messages = await self.async_convert_to_pil_images(messages)
        results = await self.vl_encoder.preprocess(messages)
        if self.backend == 'turbomind':
            results = await self.vl_encoder.async_infer(results)
            results = await self.vl_encoder.wrap_for_turbomind(
                results, self.chat_template, sequence_start)
        elif self.backend == 'pytorch':
            results = await self.vl_encoder.wrap_for_pytorch(
                results, self.chat_template, sequence_start)
        return results

    @classmethod
    async def async_convert_to_pil_images(cls,
                                          messages: List[Dict]) -> List[Dict]:
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
            asyncio.get_event_loop().run_in_executor(None, _inner_call, i,
                                                     messages, out_messages)
            for i in range(len(messages))
        ])
        return out_messages

    def batch_infer(self, prompts: Union[VLPromptType, List[Dict],
                                         List[VLPromptType], List[List[Dict]]],
                    **kwargs):
        """Inference a batch of prompts."""
        prompts = self._convert_prompts(prompts)
        return super().batch_infer(prompts, **kwargs)

    def stream_infer(self, prompts: Union[VLPromptType, List[Dict],
                                          List[VLPromptType],
                                          List[List[Dict]]], **kwargs):
        """Inference a batch of prompts with stream mode."""
        prompts = self._convert_prompts(prompts)
        return super().stream_infer(prompts, **kwargs)

    def __call__(self, prompts: Union[VLPromptType, List[Dict],
                                      List[VLPromptType], List[List[Dict]]],
                 **kwargs):
        """Inference a batch of prompts."""
        return super().__call__(prompts, **kwargs)

    def chat(self, prompts: VLPromptType, **kwargs):
        """chat."""
        _prompts = self._convert_prompts(prompts)
        sess = super().chat(_prompts, **kwargs)

        # recover prompts & history
        sess._prompt = prompts
        last_round = sess.history[-1]
        sess.history[-1] = (prompts, last_round[-1])
        return sess
