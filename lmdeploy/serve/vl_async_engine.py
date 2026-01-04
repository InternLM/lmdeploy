# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Literal, Optional, Tuple, Union

import PIL

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.multimodal_processor import MultimodalProcessor
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
        # Update prompt_processor to support multimodal processing
        self.prompt_processor = MultimodalProcessor(self.tokenizer,
                                                    self.chat_template,
                                                    vl_encoder=self.vl_encoder,
                                                    backend=backend)
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

    @classmethod
    async def async_convert_to_pil_images(cls, messages: List[Dict]) -> List[Dict]:
        """Convert messages to PIL images.

        Delegates to MultimodalProcessor.
        """
        return await MultimodalProcessor.async_convert_to_pil_images(messages)

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
