# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import PIL

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.model import BaseChatTemplate
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.utils import get_logger, try_import_deeplink
from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.vl.utils import load_image

logger = get_logger('lmdeploy')

VLPromptType = Union[str, Tuple[str, PIL.Image.Image], Tuple[str, List[PIL.Image.Image]]]


class VLAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self,
                 model_path: str,
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                 backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
                 vision_config: Optional[VisionConfig] = None,
                 **kwargs) -> None:
        if backend == 'pytorch':
            try_import_deeplink(backend_config.device_type)
        if backend_config and backend_config.enable_prefix_caching:
            backend_config.enable_prefix_caching = False
            logger.warning('Prefix caching is disabled since LMDeploy hasn\'t support in on VL models yet')
        self.vl_encoder = ImageEncoder(model_path, backend, vision_config, backend_config=backend_config)
        super().__init__(model_path, backend=backend, backend_config=backend_config, **kwargs)
        if self.model_name == 'base':
            raise RuntimeError(
                'please specify chat template as guided in https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#set-chat-template'  # noqa: E501
            )

    @classmethod
    def _convert_prompts(cls, prompts: Union[VLPromptType, List[Dict], List[VLPromptType], List[List[Dict]]]):
        """Convert prompts to openai GPT4V format."""
        if isinstance(prompts, str) or isinstance(prompts, tuple):
            _prompts = cls.prompt_to_messages(prompts)
        elif isinstance(prompts[0], tuple) or isinstance(prompts[0], str):
            _prompts = [cls.prompt_to_messages(x) for x in prompts]
        else:
            _prompts = prompts
        return _prompts

    async def _get_prompt_input(self,
                                messages: Union[str, List[Dict]],
                                do_preprocess: bool,
                                sequence_start: bool,
                                adapter_name: str,
                                tools: Optional[List[object]] = None,
                                chat_template_kwargs: Optional[Dict] = None,
                                mm_processor_kwargs: Optional[Dict[str, Any]] = None,
                                **kwargs):
        """Process messages and return the required data for the inference
        engines.

        Refer to pytorch.engine.EngineInstance.async_stream_infer and turbomind.TurboMindInstance.async_stream_infer for
        the argument specification.
        """
        if isinstance(messages, str):
            return await super()._get_prompt_input(messages,
                                                   do_preprocess,
                                                   sequence_start,
                                                   adapter_name,
                                                   tools=tools,
                                                   chat_template_kwargs=chat_template_kwargs,
                                                   **kwargs)
        elif isinstance(messages, List):
            has_multimodal_input = any(
                isinstance(message['content'], list) and any(item['type'] in ['image_url', 'image_data']
                                                             for item in message['content']) for message in messages)
            if not has_multimodal_input:
                return await super()._get_prompt_input(messages,
                                                       do_preprocess,
                                                       sequence_start,
                                                       adapter_name,
                                                       tools,
                                                       chat_template_kwargs=chat_template_kwargs,
                                                       **kwargs)
        else:
            raise RuntimeError(f'unsupported messages {messages}')

        chat_template = self.chat_template if do_preprocess else BaseChatTemplate()
        messages = await self.async_convert_to_pil_images(messages)
        results = await self.vl_encoder.preprocess(messages, mm_processor_kwargs)
        if self.backend == 'turbomind':
            # for tm engine, this module perform vision embedding after image
            # preprocessing. It utilizes the hf model's vision embeddings
            # functions and returns the input_ids, input_embeddings,
            # embedding_ranges and so on. All the returned values are passed
            # to tm engine for token generation
            results = await self.vl_encoder.async_infer(results)
            results = await self.vl_encoder.wrap_for_turbomind(results,
                                                               chat_template,
                                                               self.tokenizer,
                                                               sequence_start,
                                                               tools=tools,
                                                               chat_template_kwargs=chat_template_kwargs)
        elif self.backend == 'pytorch':
            # for pt engine, this module only conduct the image preprocessing
            # It leaves the vision embedding to the pt engine
            results = await self.vl_encoder.wrap_for_pytorch(results,
                                                             chat_template,
                                                             self.tokenizer,
                                                             sequence_start,
                                                             tools=tools,
                                                             chat_template_kwargs=chat_template_kwargs)
        return results

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

    def batch_infer(self, prompts: Union[VLPromptType, List[Dict], List[VLPromptType], List[List[Dict]]], *args,
                    **kwargs):
        """Inference a batch of prompts."""
        prompts = self._convert_prompts(prompts)
        return super().batch_infer(prompts, *args, **kwargs)

    def stream_infer(self, prompts: Union[VLPromptType, List[Dict], List[VLPromptType], List[List[Dict]]], *args,
                     **kwargs):
        """Inference a batch of prompts with stream mode."""
        prompts = self._convert_prompts(prompts)
        return super().stream_infer(prompts, *args, **kwargs)

    def __call__(self, prompts: Union[VLPromptType, List[Dict], List[VLPromptType], List[List[Dict]]], *args, **kwargs):
        """Inference a batch of prompts."""
        return super().__call__(prompts, *args, **kwargs)

    def close(self):
        if hasattr(self, 'vl_encoder'):
            del self.vl_encoder
            super().close()

    def chat(self, prompts: VLPromptType, *args, **kwargs):
        """chat."""
        _prompts = self._convert_prompts(prompts)
        sess = super().chat(_prompts, *args, **kwargs)

        # recover prompts & history
        sess._prompt = prompts
        if sess.history:
            last_round = sess.history[-1]
            sess.history[-1] = (prompts, last_round[-1])
        return sess

    @classmethod
    def prompt_to_messages(cls, prompt: VLPromptType):
        """Convert prompt to GTP4V format."""
        messages = {
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': '',
            }]
        }
        if isinstance(prompt, str):
            messages['content'][0]['text'] = prompt
        else:
            prompt, images = prompt
            if not isinstance(images, list):
                images = [images]
            messages['content'][0]['text'] = prompt
            for image in images:
                # 'image_url': means url or local path to image.
                # 'image_data': means PIL.Image.Image object.
                if isinstance(image, str):
                    image = load_image(image)
                    item = {'type': 'image_data', 'image_data': {'data': image}}
                elif isinstance(image, PIL.Image.Image):
                    item = {'type': 'image_data', 'image_data': {'data': image}}
                else:
                    raise ValueError('image should be a str(url/path) or PIL.Image.Image')

                messages['content'].append(item)

        return [messages]
