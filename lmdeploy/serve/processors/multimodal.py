# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Any, Literal

import PIL

from lmdeploy.model import MODELS, BaseChatTemplate
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.media.audio import AudioMediaIO
from lmdeploy.vl.media.connection import load_from_url
from lmdeploy.vl.media.image import ImageMediaIO
from lmdeploy.vl.media.time_series import TimeSeriesMediaIO
from lmdeploy.vl.media.video import VideoMediaIO

logger = get_logger('lmdeploy')


class MultimodalProcessor:
    """Processor for handling prompt preprocessing, message content merging,
    and multimodal processing."""

    def __init__(self,
                 tokenizer: Tokenizer,
                 chat_template: BaseChatTemplate,
                 vl_encoder=None,
                 backend: str | None = None):
        """Initialize MultimodalProcessor.

        Args:
            tokenizer: Tokenizer instance for encoding prompts.
            chat_template: Chat template instance for message processing.
            vl_encoder: Optional ImageEncoder instance for multimodal processing.
            backend: Optional backend name ('turbomind' or 'pytorch') for multimodal processing.
        """
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.vl_encoder = vl_encoder
        self.backend = backend

    @staticmethod
    def merge_message_content(msg: dict) -> dict:
        """Merge multimodal content blocks and ensure content field exists.

        This function normalizes message content to match vLLM's behavior:
        1. Missing content field -> add content='' (empty string)
        2. None content -> convert to content='' (empty string)
        3. String content -> return as-is
        4. List content (multimodal) -> merge all text blocks with newline separator

        Args:
            msg: A message dict with 'role' and optionally 'content' field

        Returns:
            A message dict with 'content' field guaranteed to exist

        Note:
            This implementation is based on vLLM's content processing logic.
            vLLM uses "\n".join() to merge multiple text blocks from multimodal content.

        References:
            - vLLM content normalization:
              https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/chat_utils.py
              See _parse_chat_message_content() and _parse_chat_message_content_parts()
            - vLLM text merging logic:
              text_prompt = "\n".join(texts)
        """
        # If content is missing or None, convert to empty string (matches vLLM behavior)
        # This prevents Jinja2 template errors when rendering chat templates
        if 'content' not in msg or msg['content'] is None:
            result = dict(msg)
            result['content'] = ''
            return result

        # If content is already a string, return as-is
        if isinstance(msg['content'], str):
            return msg

        # If content is a list, merge all text blocks into a single string
        # This matches vLLM's behavior: text_prompt = "\n".join(texts)
        content_parts = []
        for block in msg['content']:
            if isinstance(block, dict) and block.get('type') == 'text':
                content_parts.append(block.get('text', ''))
        merged_content = '\n'.join(content_parts)

        # Preserve all other fields in the message (e.g., tool_calls)
        result = dict(msg)
        result['content'] = merged_content
        return result

    @staticmethod
    def _parse_multimodal_item(i: int, in_messages: list[dict], out_messages: list[dict], media_io_kwargs: dict[str,
                                                                                                                Any]):
        """Synchronous helper to parse a single multimodal message item."""
        role = in_messages[i]['role']
        content = in_messages[i]['content']

        if role != 'user' or isinstance(content, str):
            out_messages[i] = in_messages[i]
            return

        assert isinstance(content, list)
        out_message = dict(role=role, content=[])

        for item in content:
            item_type = item.get('type')
            if item_type == 'text':
                out_message['content'].append(item)
                continue

            item_params = item.get(item_type, {}).copy()
            data_src = item_params.pop('url', None) or item_params.pop('data', None)

            if item_type == 'image_data':
                modality = Modality.IMAGE
                data = data_src
            elif item_type == 'image_url':
                modality = Modality.IMAGE
                img_io = ImageMediaIO(**media_io_kwargs.get('image', {}))
                data = load_from_url(data_src, img_io)
            elif item_type == 'video_url':
                modality = Modality.VIDEO
                vid_io = VideoMediaIO(image_io=ImageMediaIO(), **media_io_kwargs.get('video', {}))
                data, metadata = load_from_url(data_src, vid_io)
                item_params['video_metadata'] = metadata
            elif item_type == 'audio_url':
                modality = Modality.AUDIO
                audio_io = AudioMediaIO(**media_io_kwargs.get('audio', {}))
                data = load_from_url(data_src, audio_io)
            elif item_type == 'time_series_url':
                modality = Modality.TIME_SERIES
                ts_io = TimeSeriesMediaIO(**media_io_kwargs.get('time_series', {}))
                data = load_from_url(data_src, ts_io)
            else:
                raise NotImplementedError(f'unknown type: {item_type}')

            out_message['content'].append({'type': modality, 'data': data, **item_params})

        out_messages[i] = out_message

    @staticmethod
    async def async_parse_multimodal_item(messages: list[dict],
                                          media_io_kwargs: dict[str, Any] | None = None) -> list[dict]:
        """Convert user-input multimodal data into GPT4V message format."""
        if isinstance(messages, dict):
            messages = [messages]
        assert isinstance(messages, list)

        out_messages = [None] * len(messages)
        media_io_kwargs = media_io_kwargs or {}
        loop = asyncio.get_event_loop()

        await asyncio.gather(*[
            loop.run_in_executor(None, MultimodalProcessor._parse_multimodal_item, i, messages, out_messages,
                                 media_io_kwargs) for i in range(len(messages))
        ])
        return out_messages

    async def get_prompt_input(self,
                               prompt: str | list[dict],
                               do_preprocess: bool,
                               sequence_start: bool,
                               adapter_name: str,
                               tools: list[object] | None = None,
                               reasoning_effort: Literal['low', 'medium', 'high'] | None = None,
                               chat_template_kwargs: dict | None = None,
                               media_io_kwargs: dict[str, Any] | None = None,
                               mm_processor_kwargs: dict[str, Any] | None = None,
                               **kwargs):
        """Process prompt and return prompt string and input_ids.

        Handles both text-only and multimodal prompts. If multimodal input is detected
        and vl_encoder is available, processes images accordingly.

        Args:
            prompt: Input prompt as string or list of message dicts.
            do_preprocess: Whether to apply chat template preprocessing.
            sequence_start: Indicator for starting a sequence.
            adapter_name: Adapter name for selecting chat template.
            tools: Optional list of tools.
            reasoning_effort: Optional reasoning effort level.
            chat_template_kwargs: Optional kwargs for chat template.
            media_io_kwargs: Optional kwargs for media IO operations.
            mm_processor_kwargs: Optional kwargs for multimodal processor.
            **kwargs: Additional keyword arguments.

        Returns:
            dict with 'prompt' (str) and 'input_ids' (list[int]) keys for text-only,
            or dict with multimodal data for multimodal prompts.
        """
        # Handle string input
        if isinstance(prompt, str):
            return await self._get_text_prompt_input(prompt=prompt,
                                                     do_preprocess=do_preprocess,
                                                     sequence_start=sequence_start,
                                                     adapter_name=adapter_name,
                                                     tools=tools,
                                                     reasoning_effort=reasoning_effort,
                                                     chat_template_kwargs=chat_template_kwargs,
                                                     **kwargs)

        # Handle list input
        elif isinstance(prompt, list):
            # Check if multimodal input exists
            has_multimodal_input = self._has_multimodal_input(prompt)

            # If no multimodal input or no vl_encoder, use text-only processing
            if not has_multimodal_input or self.vl_encoder is None:
                return await self._get_text_prompt_input(prompt=prompt,
                                                         do_preprocess=do_preprocess,
                                                         sequence_start=sequence_start,
                                                         adapter_name=adapter_name,
                                                         tools=tools,
                                                         reasoning_effort=reasoning_effort,
                                                         chat_template_kwargs=chat_template_kwargs,
                                                         **kwargs)

            # Process multimodal input
            return await self._get_multimodal_prompt_input(messages=prompt,
                                                           do_preprocess=do_preprocess,
                                                           sequence_start=sequence_start,
                                                           adapter_name=adapter_name,
                                                           tools=tools,
                                                           chat_template_kwargs=chat_template_kwargs,
                                                           media_io_kwargs=media_io_kwargs,
                                                           mm_processor_kwargs=mm_processor_kwargs,
                                                           **kwargs)
        else:
            raise RuntimeError(f'unsupported prompt type: {type(prompt)}')

    @staticmethod
    def format_prompts(prompts: Any) -> list[dict]:
        """Format prompts."""
        if not isinstance(prompts, list):
            prompts = [prompts]
        # str or batch of str
        if all(isinstance(prompt, str) for prompt in prompts):
            return prompts
        if (MultimodalProcessor._is_openai_message(prompts)
                or all(MultimodalProcessor._is_openai_message(prompt) for prompt in prompts)):
            return prompts
        if all(MultimodalProcessor._is_str_images_pair(prompt) for prompt in prompts):
            # batch of (prompt, image or [images]) or (image or [images], prompt) ->
            # [[openai_gpt4v_message], [openai_gpt4v_message], ...]
            return [[MultimodalProcessor._re_format_prompt_images_pair(prompt)] for prompt in prompts]
        raise ValueError(f'Unsupported prompts: {prompts}. Only support str, openai message format, '
                         'or (prompt, image or [images]) or (image or [images], prompt) pair.')

    @staticmethod
    def _is_openai_message(message) -> bool:
        """Check if the message conforms to openai message format."""
        return isinstance(message, list) and all(isinstance(msg, dict) for msg in message)

    @staticmethod
    def _is_str_images_pair(message) -> bool:
        """Check if the message is a (prompt, image or [images]) or (image or
        [images], prompt) pair."""
        if not (isinstance(message, tuple) and len(message) == 2):
            return False
        _1, _2 = message
        if MultimodalProcessor._is_image(_1) or MultimodalProcessor._is_image_list(_1):
            _1, _2 = _2, _1
        return isinstance(_1, str) and (MultimodalProcessor._is_image(_2) or MultimodalProcessor._is_image_list(_2))

    @staticmethod
    def _is_image(obj) -> bool:
        # image or image url or base64-encoded image data
        return (isinstance(obj, PIL.Image.Image)
                or isinstance(obj, str) and (obj.startswith('http') or obj.startswith('data:image')))

    @staticmethod
    def _is_image_list(obj) -> bool:
        return isinstance(obj, list) and all(MultimodalProcessor._is_image(img) for img in obj)

    @staticmethod
    def _re_format_prompt_images_pair(prompt: tuple) -> dict:
        """Reformat the prompt to openai message format."""
        from lmdeploy.vl import load_image

        messages = {'role': 'user', 'content': []}
        prompt, images = prompt
        prompt_first = True
        if MultimodalProcessor._is_image(prompt) or MultimodalProcessor._is_image_list(prompt):
            prompt, images = images, prompt
            prompt_first = False
        image_contents = []
        images = images if isinstance(images, list) else [images]
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
            image_contents.append(item)

        if prompt_first:
            messages['content'].append({'type': 'text', 'text': prompt})
            messages['content'].extend(image_contents)
        else:
            messages['content'].extend(image_contents)
            messages['content'].append({'type': 'text', 'text': prompt})
        return messages

    def _has_multimodal_input(self, messages: list[dict]) -> bool:
        """Check if messages contain multimodal input (images)."""
        multimodal_types = ['image_url', 'image_data', 'video_url', 'audio_url', 'time_series_url']
        return any(
            isinstance(message.get('content'), list) and any(
                item.get('type') in multimodal_types for item in message['content']) for message in messages)

    async def _get_text_prompt_input(self,
                                     prompt: str | list[dict],
                                     do_preprocess: bool,
                                     sequence_start: bool,
                                     adapter_name: str,
                                     tools: list[object] | None = None,
                                     reasoning_effort: Literal['low', 'medium', 'high'] | None = None,
                                     chat_template_kwargs: dict | None = None,
                                     **kwargs):
        """Process text-only prompt and return prompt string and input_ids."""
        # Change multimodal data to openai text messages
        if isinstance(prompt, list):
            prompt = [self.merge_message_content(msg) for msg in prompt]
        if do_preprocess:
            # use adapter's chat template if possible
            chat_template = self.chat_template
            if adapter_name in MODELS.module_dict:
                chat_template = MODELS.module_dict[adapter_name]()
        else:
            chat_template = BaseChatTemplate()
        chat_template_kwargs = chat_template_kwargs or {}
        prompt = chat_template.messages2prompt(prompt,
                                               sequence_start,
                                               tools=tools,
                                               reasoning_effort=reasoning_effort,
                                               **chat_template_kwargs)
        if prompt is None:
            raise ValueError(
                f'You are using base template to handle chat task. Please specify a `--chat-template` name chosen from `lmdeploy list` if you want to use OpenAI messages input.'  # noqa
            )
        input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)
        return {'prompt': prompt, 'input_ids': input_ids}

    async def _get_multimodal_prompt_input(self,
                                           messages: list[dict],
                                           do_preprocess: bool,
                                           sequence_start: bool,
                                           adapter_name: str,
                                           tools: list[object] | None = None,
                                           chat_template_kwargs: dict | None = None,
                                           media_io_kwargs: dict[str, Any] | None = None,
                                           mm_processor_kwargs: dict[str, Any] | None = None,
                                           **kwargs):
        """Process multimodal prompt and return processed data for inference
        engines."""
        chat_template = self.chat_template if do_preprocess else BaseChatTemplate()
        messages = await self.async_parse_multimodal_item(messages, media_io_kwargs)
        results = await self.vl_encoder.preprocess(messages, mm_processor_kwargs)

        if self.backend == 'turbomind':
            # for tm engine, this module perform vision embedding after image
            # preprocessing. It utilizes the hf model's vision embeddings
            # functions and returns the input_ids, input_embeddings,
            # embedding_ranges and so on. All the returned values are passed
            # to tm engine for token generation
            results = await self.vl_encoder.async_infer(results)
            results = await self.vl_encoder.wrap_for_turbomind(messages=results,
                                                               chat_template=chat_template,
                                                               tokenizer=self.tokenizer,
                                                               sequence_start=sequence_start,
                                                               tools=tools,
                                                               chat_template_kwargs=chat_template_kwargs)
        elif self.backend == 'pytorch':
            # for pt engine, this module only conduct the image preprocessing
            # It leaves the vision embedding to the pt engine
            results = await self.vl_encoder.wrap_for_pytorch(messages=results,
                                                             chat_template=chat_template,
                                                             tokenizer=self.tokenizer,
                                                             sequence_start=sequence_start,
                                                             tools=tools,
                                                             chat_template_kwargs=chat_template_kwargs)
        return results
